"""VM reward scoring server — FastAPI endpoint.

Loads a single VM (scalar-head or vocab_lowrank) and exposes a HTTP scoring
API. Designed to run in a lightweight Docker container (no vLLM required).

Endpoints
---------
GET  /health
    {"status": "ok", "vm_type": "scalar|vocab_lowrank", "model": "..."}

POST /score
    Request : {"instruction": "...", "response": "..."}
    Response: {"score": 3.14}

POST /score_batch
    Request : [{"instruction": "...", "response": "..."}, ...]
    Response: [{"score": 3.14}, ...]
    (processed sequentially — GPU memory stays bounded)

Usage
-----
    python scripts/vm_score_server.py \\
        --vm_type  scalar \\
        --vm_model /workspace/models/VM-Qwen3-4B-merged-for-vllm \\
        --device   cuda:0 \\
        --port     8001

    python scripts/vm_score_server.py \\
        --vm_type  vocab_lowrank \\
        --vm_model /workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-bt-20260704-merged \\
        --device   cuda:0 \\
        --port     8002
"""

import argparse
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def _tokenize(tokenizer, messages, add_generation_prompt=False):
    """apply_chat_template → plain list of ints (works across transformers versions)."""
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if hasattr(result, "input_ids"):
        result = result.input_ids
    if isinstance(result, torch.Tensor):
        result = result.flatten().tolist()
    return result


# ---------------------------------------------------------------------------
# Scalar head VM
# ---------------------------------------------------------------------------

class ScalarVM:
    """Qwen3ForSequenceClassification — last-token scalar reward."""

    def __init__(self, model_path: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            num_labels=1,
        )
        self.model.eval()
        print(f"[scalar] loaded from {model_path}", flush=True)

    @torch.inference_mode()
    def score(self, instruction: str, response: str) -> float:
        messages = [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": response},
        ]
        ids = _tokenize(self.tokenizer, messages)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_tensor)
        return float(out.logits.squeeze().item())


# ---------------------------------------------------------------------------
# Vocab_lowrank VM
# ---------------------------------------------------------------------------

class VocabLowrankVM:
    """Qwen3ForCausalLM + low-rank head (score_A / score_B).

    Response score = mean over response token positions of
        score_A(hidden[t]) @ score_B  indexed at  token[t+1]
    — consistent with the ARM training objective (h[t] → reward for token[t+1]).
    """

    def __init__(self, model_path: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        # Load score_A / score_B from safetensors (skipped by AutoModelForCausalLM).
        score_A_w = score_B_w = None
        for st in sorted(Path(model_path).glob("*.safetensors")):
            tensors = load_file(st, device="cpu")
            if "score_A.weight" in tensors:
                score_A_w = tensors["score_A.weight"]
            if "score_B.weight" in tensors:
                score_B_w = tensors["score_B.weight"]
            if score_A_w is not None and score_B_w is not None:
                break

        if score_A_w is None or score_B_w is None:
            raise ValueError(f"score_A.weight / score_B.weight not found in {model_path}")

        rank       = score_A_w.shape[0]
        vocab_size = score_B_w.shape[0]
        hidden_dim = score_A_w.shape[1]

        self.score_A = nn.Linear(hidden_dim, rank,       bias=False).to(device=device, dtype=torch.float32)
        self.score_B = nn.Linear(rank,       vocab_size, bias=False).to(device=device, dtype=torch.float32)
        self.score_A.weight.data.copy_(score_A_w.float())
        self.score_B.weight.data.copy_(score_B_w.float())
        self.vocab_size = vocab_size

        print(f"[vocab_lowrank] loaded rank={rank} vocab={vocab_size} hidden={hidden_dim}", flush=True)

    @torch.inference_mode()
    def score(self, instruction: str, response: str) -> float:
        messages = [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": response},
        ]
        full_ids   = _tokenize(self.tokenizer, messages)
        prompt_ids = _tokenize(self.tokenizer,
                               [{"role": "user", "content": instruction}],
                               add_generation_prompt=True)

        prompt_len = len(prompt_ids)
        seq_len    = len(full_ids)

        if prompt_len >= seq_len - 1:
            return 0.0

        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        out    = self.model(input_tensor, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]  # (seq_len, hidden)

        # ARM alignment: h[t] scores token[t+1]
        # t_start = last prompt position; next token = first response token
        t_start = prompt_len - 1
        t_end   = seq_len - 2  # inclusive — last position whose next token is a response token

        if t_end < t_start:
            return 0.0

        h_slice   = hidden[t_start : t_end + 1].float()             # (resp_len, hidden)
        next_toks = input_tensor[0, t_start + 1 : t_end + 2]        # (resp_len,)
        next_toks = next_toks.clamp(max=self.vocab_size - 1)

        h_a          = self.score_A(h_slice)                          # (resp_len, rank)
        vocab_scores = self.score_B(h_a)                              # (resp_len, vocab_size)
        token_values = vocab_scores[
            torch.arange(len(next_toks), device=self.device), next_toks
        ]
        return float(token_values.mean().item())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app   = FastAPI(title="VM Score Server")
_vm   = None        # loaded at startup
_meta = {}          # vm_type, model_path


class ScoreRequest(BaseModel):
    instruction: str
    response: str

class ScoreResponse(BaseModel):
    score: float


@app.get("/health")
def health():
    return {"status": "ok", **_meta}


@app.post("/score", response_model=ScoreResponse)
def score_one(req: ScoreRequest):
    try:
        s = _vm.score(req.instruction, req.response)
        return ScoreResponse(score=s)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_batch", response_model=List[ScoreResponse])
def score_batch(reqs: List[ScoreRequest]):
    results = []
    for req in reqs:
        try:
            s = _vm.score(req.instruction, req.response)
            results.append(ScoreResponse(score=s))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _vm, _meta

    p = argparse.ArgumentParser()
    p.add_argument("--vm_type",  required=True, choices=["scalar", "vocab_lowrank"])
    p.add_argument("--vm_model", required=True, help="path to merged VM checkpoint")
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--port",     type=int, default=8001)
    args = p.parse_args()

    print(f"Loading {args.vm_type} VM from {args.vm_model} on {args.device} ...", flush=True)
    t0 = time.time()
    if args.vm_type == "scalar":
        _vm = ScalarVM(args.vm_model, args.device)
    else:
        _vm = VocabLowrankVM(args.vm_model, args.device)
    print(f"Model ready in {time.time()-t0:.1f}s", flush=True)

    _meta["vm_type"]    = args.vm_type
    _meta["model_path"] = args.vm_model

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

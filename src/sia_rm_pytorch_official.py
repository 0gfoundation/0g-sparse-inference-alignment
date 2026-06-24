"""
SIA RM Server — loads strictly using the official ValueModel.from_pretrained, bypassing the vllm serve path.

Difference from src/sia_rm_server.py (which uses a custom _ValueModelWrapper implementation):
This server directly imports /workspace/SIA/git/SIA/src/value_model/model.ValueModel
and loads base + LoRA + token_reward_head entirely following the official code, with no conversion.

The /score endpoint is fully compatible with the existing sia_rm_server.py; the main LLM server
does not need any code changes — just point --rm_url at this server.

Startup:
  python src/sia_rm_pytorch_official.py \\
    --rm /workspace/SIA/models/Qwen3-4B \\
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \\
    --device cuda:0 --port 8001
"""
import argparse
import sys
import time
from typing import Optional

# Add the official SIA repo path
sys.path.insert(0, '/workspace/SIA/git/SIA')

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.value_model.model import ValueModel


app = FastAPI()

# Globals
VM = None
TOK = None
DEVICE = "cuda:0"


class ScoreRequest(BaseModel):
    user_content: str
    response_so_far: str
    candidate_texts: list[str]
    request_id: Optional[str] = None


class ScoreTokenIdsRequest(BaseModel):
    """Fully aligned with the vllm /classify (path A direct-token-ids) payload schema:
    input = list[list[int]], each inner list = one complete candidate sequence (prefix + gen + cand).
    request_id is used only for dual-VM log correlation.
    """
    input: list[list[int]]
    request_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "backend": "pytorch-official-value-model"}


@app.post("/score")
def score(req: ScoreRequest):
    """Same schema as sia_rm_server.py, but loads using the official ValueModel, returns raw logit (no sigmoid)."""
    try:
        return _score_impl(req)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[RM-error] {type(e).__name__}: {e}\n{tb}", flush=True)
        n = len(req.candidate_texts) if req.candidate_texts else 5
        return {"scores": [0.0] * n, "error": f"{type(e).__name__}: {e}"}


def _score_impl(req: ScoreRequest):
    """
    Tokenize prefix + response_so_far + candidate (no <|im_end|> close tag, format after Fix #1)
    separately, run through ValueModel forward, and take the token_reward at the last non-pad
    position indicated by attention_mask as the raw logit.
    """
    t0 = time.time()

    # 1. Build prefix (render user_msg + fake assistant SENTINEL via chat_template, then split)
    convs = [
        {"role": "user", "content": req.user_content},
        {"role": "assistant", "content": "ZSIASENTINELZ"},
    ]
    text = TOK.apply_chat_template(convs, tokenize=False)
    prefix, _ = text.split("ZSIASENTINELZ", 1)
    if TOK.bos_token and prefix.startswith(TOK.bos_token):
        prefix = prefix[len(TOK.bos_token):]

    # 2. Build 5 formatted_texts (consistent with _score_candidates_vllm after Fix #1, no suffix)
    formatted = [prefix + req.response_so_far + ct for ct in req.candidate_texts]

    # 3. Batch tokenize, padding to longest
    enc = TOK(formatted, return_tensors='pt', padding=True,
              add_special_tokens=False).to(DEVICE)

    # 4. Forward through official ValueModel — note: official forward uses token_rewards[:, -1]
    #    without looking at attention_mask. We use attention_mask to find the last non-pad position,
    #    to correctly handle variable-length inputs within a batch (get token_rewards from
    #    backbone.forward then pick manually).
    with torch.no_grad():
        # ValueModel.forward returns token_rewards (batch, seq) and logits (batch, 1) = [:, -1]
        # But [:, -1] is wrong for padded inputs; we manually pick the last non-pad position
        # from token_rewards using the mask
        out = VM(input_ids=enc.input_ids,
                 attention_mask=enc.attention_mask)
        # token_rewards shape: (batch, seq)
        token_rewards = out.token_rewards
        # Find the last non-pad position for each row
        seq_lens = enc.attention_mask.sum(dim=1) - 1   # (batch,)
        idx = torch.arange(token_rewards.size(0), device=DEVICE)
        last_rewards = token_rewards[idx, seq_lens]    # (batch,) raw logit

    scores = last_rewards.float().cpu().tolist()

    t_total = time.time() - t0
    return {"scores": scores, "elapsed_ms": round(t_total * 1000, 2)}


@app.post("/score_token_ids")
def score_token_ids(req: ScoreTokenIdsRequest):
    """Direct token-ids version: input = list[list[int]] of complete candidate sequences.

    Usage: dual-VM comparison — accepts the exact same payload as vllm /classify
    (path A direct-token-ids) (LLM-generated token_ids + candidate cand_id), runs official
    ValueModel.forward + takes the reward at the last non-pad token, returns raw logit list[float].

    Difference from /score: completely skips the apply_chat_template / re-tokenize path,
    forwarding directly with the token_ids from upstream — 100% aligned with the vllm /classify input distribution.
    """
    try:
        t0 = time.time()
        token_id_lists = req.input
        if not token_id_lists:
            return {"scores": [], "elapsed_ms": 0.0}

        # Pad to longest
        max_len = max(len(ids) for ids in token_id_lists)
        pad_id = TOK.pad_token_id
        input_ids = torch.full(
            (len(token_id_lists), max_len), pad_id,
            dtype=torch.long, device=DEVICE,
        )
        attention_mask = torch.zeros(
            (len(token_id_lists), max_len),
            dtype=torch.long, device=DEVICE,
        )
        for i, ids in enumerate(token_id_lists):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=DEVICE)
            attention_mask[i, :n] = 1

        with torch.no_grad():
            out = VM(input_ids=input_ids, attention_mask=attention_mask)
            token_rewards = out.token_rewards
            seq_lens = attention_mask.sum(dim=1) - 1   # (batch,)
            idx = torch.arange(token_rewards.size(0), device=DEVICE)
            last_rewards = token_rewards[idx, seq_lens]

        scores = last_rewards.float().cpu().tolist()
        t_total = time.time() - t0
        return {"scores": scores, "elapsed_ms": round(t_total * 1000, 2)}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[RM-token-ids-error] {type(e).__name__}: {e}\n{tb}", flush=True)
        n = len(req.input) if req.input else 5
        return {"scores": [0.0] * n, "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rm",      required=True,
                   help="Base model path (e.g. /workspace/SIA/models/Qwen3-4B)")
    p.add_argument("--rm_lora", required=True,
                   help="LoRA + token_reward_head directory "
                        "(e.g. /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base)")
    p.add_argument("--device",  default="cuda:0")
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=8001)
    args = p.parse_args()

    global VM, TOK, DEVICE
    DEVICE = args.device

    print(f"[RM-server] Loading official ValueModel.from_pretrained:")
    print(f"  base: {args.rm}")
    print(f"  lora: {args.rm_lora}")
    print(f"  device: {DEVICE}")
    VM = ValueModel.from_pretrained(
        base_model_path=args.rm,
        model_path=args.rm_lora,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).to(DEVICE).eval()

    TOK = AutoTokenizer.from_pretrained(args.rm)
    if TOK.pad_token_id is None:
        TOK.pad_token_id = TOK.eos_token_id

    print(f"[RM-server] VM ready. Starting FastAPI on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

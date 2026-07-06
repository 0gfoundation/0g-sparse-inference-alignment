"""Offline VM comparison: scalar head vs vocab_lowrank.

Scores each (instruction, output) pair from a scored JSON file using both VMs,
then computes Spearman correlation against the ground-truth Skywork rewards.

Usage:
    python scripts/compare_vm_offline.py \
        --data    exp/alpaca-vl30b-farma-bt-2048-seq-20260705/nosia_scored.json \
        --scalar  /workspace/models/VM-Qwen3-4B-merged-for-vllm \
        --vocab   /workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-bt-20260704-merged \
        --device  cuda:0
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Qwen3ForCausalLM
from scipy.stats import spearmanr


def _tokenize(tokenizer, messages, add_generation_prompt=False):
    """apply_chat_template → plain list of ints (works across transformers versions)."""
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # Newer transformers may return BatchEncoding; extract input_ids list.
    if hasattr(result, "input_ids"):
        result = result.input_ids
    if isinstance(result, torch.Tensor):
        result = result.flatten().tolist()
    return result


# ---------------------------------------------------------------------------
# Vocab_lowrank VM loader (pure HuggingFace, no vLLM)
# ---------------------------------------------------------------------------

class VocabLowrankVM:
    """Qwen3ForCausalLM + score_A / score_B low-rank head for offline response scoring.

    Response score = mean over response token positions of
        score_A(hidden_t) @ score_B  indexed at  tokens[t+1]
    — the TD value accumulated across the response.
    """

    def __init__(self, model_path: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        # Load score_A / score_B from safetensors (ignored by Qwen3ForCausalLM.load_weights).
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
            raise ValueError(f"score_A.weight or score_B.weight not found in {model_path}")

        rank      = score_A_w.shape[0]
        vocab_size = score_B_w.shape[0]
        hidden_size = score_A_w.shape[1]

        self.score_A = nn.Linear(hidden_size, rank,       bias=False).to(device=device, dtype=torch.float32)
        self.score_B = nn.Linear(rank,        vocab_size, bias=False).to(device=device, dtype=torch.float32)
        self.score_A.weight.data.copy_(score_A_w.float())
        self.score_B.weight.data.copy_(score_B_w.float())
        self.vocab_size = vocab_size

        print(f"[vocab_lowrank] rank={rank} vocab_size={vocab_size} hidden={hidden_size}", flush=True)

    @torch.inference_mode()
    def score(self, instruction: str, output: str) -> float:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        full_ids    = _tokenize(self.tokenizer, messages)
        prompt_ids  = _tokenize(self.tokenizer, [{"role": "user", "content": instruction}],
                                add_generation_prompt=True)
        prompt_len  = len(prompt_ids)
        seq_len     = len(full_ids)

        if prompt_len >= seq_len - 1:
            return 0.0

        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        out = self.model(input_tensor, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]  # (seq_len, hidden)

        # t_start..t_end: positions whose NEXT token is a response token
        t_start = prompt_len - 1
        t_end   = seq_len - 2  # inclusive

        if t_end < t_start:
            return 0.0

        h_slice    = hidden[t_start:t_end + 1].float()         # (resp_len, hidden)
        next_toks  = input_tensor[0, t_start + 1:t_end + 2]    # (resp_len,)
        # clamp in case vocab_size mismatch
        next_toks  = next_toks.clamp(max=self.vocab_size - 1)

        h_a           = self.score_A(h_slice)                   # (resp_len, rank)
        vocab_scores  = self.score_B(h_a)                       # (resp_len, vocab_size)
        token_values  = vocab_scores[
            torch.arange(len(next_toks), device=self.device), next_toks
        ]
        return float(token_values.mean().item())


# ---------------------------------------------------------------------------
# Scalar head VM loader
# ---------------------------------------------------------------------------

class ScalarVM:
    """Qwen3ForSequenceClassification for offline response scoring."""

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

    @torch.inference_mode()
    def score(self, instruction: str, output: str) -> float:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        ids = _tokenize(self.tokenizer, messages)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_tensor)
        return float(out.logits.squeeze().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True)
    p.add_argument("--scalar", required=True, help="Qwen3ForSequenceClassification checkpoint")
    p.add_argument("--vocab",  required=True, help="vocab_lowrank checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit",  type=int, default=0, help="0 = all")
    args = p.parse_args()

    data = json.load(open(args.data))
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} items from {args.data}", flush=True)

    print("Loading scalar head VM ...", flush=True)
    scalar_vm = ScalarVM(args.scalar, args.device)

    print("Loading vocab_lowrank VM ...", flush=True)
    vocab_vm = VocabLowrankVM(args.vocab, args.device)

    skywork_scores, scalar_scores, vocab_scores = [], [], []

    for i, item in enumerate(data):
        instr  = item["instruction"]
        output = item["output"]
        if item.get("reward") is None:
            continue
        sw     = float(item["reward"])

        sc = scalar_vm.score(instr, output)
        vc = vocab_vm.score(instr, output)

        skywork_scores.append(sw)
        scalar_scores.append(sc)
        vocab_scores.append(vc)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(data)}] skywork={sw:6.2f}  scalar={sc:+.4f}  vocab={vc:+.4f}",
                  flush=True)

    rho_scalar, p_scalar = spearmanr(skywork_scores, scalar_scores)
    rho_vocab,  p_vocab  = spearmanr(skywork_scores, vocab_scores)
    rho_cross,  _        = spearmanr(scalar_scores,  vocab_scores)

    print()
    print("=" * 62)
    print(f"  Scalar head   vs Skywork : ρ = {rho_scalar:+.4f}  (p={p_scalar:.3g})")
    print(f"  Vocab_lowrank vs Skywork : ρ = {rho_vocab:+.4f}  (p={p_vocab:.3g})")
    print(f"  Scalar  vs Vocab_lowrank : ρ = {rho_cross:+.4f}  (agreement between VMs)")
    print("=" * 62)

    pairs_total = pairs_scalar = pairs_vocab = 0
    n = len(skywork_scores)
    for i in range(n):
        for j in range(i + 1, n):
            if skywork_scores[i] == skywork_scores[j]:
                continue
            pairs_total += 1
            prefer_i = skywork_scores[i] > skywork_scores[j]
            if (scalar_scores[i] > scalar_scores[j]) == prefer_i:
                pairs_scalar += 1
            if (vocab_scores[i] > vocab_scores[j]) == prefer_i:
                pairs_vocab += 1

    if pairs_total:
        print(f"  Pairwise accuracy  (N={pairs_total} pairs):")
        print(f"    Scalar head   : {pairs_scalar/pairs_total:.1%}")
        print(f"    Vocab_lowrank : {pairs_vocab/pairs_total:.1%}")
        print("=" * 62)


if __name__ == "__main__":
    main()

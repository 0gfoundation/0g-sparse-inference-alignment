"""
VM backbone-only generation (no SIA, no head intervention).

Loads the VM as Qwen3ForCausalLM (overriding Qwen3ForSequenceClassification
stored in config.json). With tie_word_embeddings=True, lm_head is automatically
tied to embed_tokens — no weight is missing. score*.weight keys are unexpected
and produce a warning, which is suppressed here for clarity.

Goal: rule out backbone weight differences as the cause of vocab_lowrank
underperformance vs scalar in SIA evaluation.

Usage
-----
    python scripts/vm_backbone_gen.py \
        --vm /workspace/models/VM-Qwen3-4B-merged-for-vllm \
        --label scalar \
        --output exp/alpaca-vm-backbone-20260706/scalar_backbone.json \
        --limit 200 --max_tokens 512 --batch_size 8
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM


def load_model(vm_path: str, device: str):
    print(f"[backbone] 加载 tokenizer: {vm_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(vm_path, trust_remote_code=True)
    tok.padding_side = "left"

    print(f"[backbone] 加载模型为 Qwen3ForCausalLM ...", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*score.*")
        warnings.filterwarnings("ignore", message=".*unexpected.*")
        warnings.filterwarnings("ignore", message=".*regex.*")
        model = Qwen3ForCausalLM.from_pretrained(
            vm_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    model.eval()
    print(f"[backbone] 模型加载完成，耗时 {time.time()-t0:.1f}s", flush=True)
    return tok, model


@torch.inference_mode()
def generate_batch(tok, model, instructions: list, max_tokens: int, temperature: float) -> list:
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # disable <think> prefix in assistant turn
        )
        for instr in instructions
    ]
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=1024).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=tok.eos_token_id,
    )
    return [tok.decode(out[i, prompt_len:], skip_special_tokens=True)
            for i in range(len(instructions))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vm",          required=True)
    p.add_argument("--label",       default="vm")
    p.add_argument("--output",      required=True)
    p.add_argument("--data",        default="data/alpaca_eval/alpaca_eval.json")
    p.add_argument("--limit",       type=int,   default=200)
    p.add_argument("--max_tokens",  type=int,   default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--device",      default="cuda:0")
    args = p.parse_args()

    questions = json.load(open(args.data))
    if args.limit:
        questions = questions[:args.limit]
    print(f"[backbone] {args.label}: {len(questions)} 条，"
          f"max_tokens={args.max_tokens}  batch_size={args.batch_size}", flush=True)

    tok, model = load_model(args.vm, args.device)

    results = []
    n_err = 0
    t0 = time.time()

    for batch_start in range(0, len(questions), args.batch_size):
        batch = questions[batch_start : batch_start + args.batch_size]
        instrs = [q["instruction"] for q in batch]

        try:
            outputs = generate_batch(tok, model, instrs, args.max_tokens, args.temperature)
            err = False
        except Exception as e:
            outputs = [f"ERROR: {e}"] * len(batch)
            n_err += len(batch)
            err = True

        for q, out in zip(batch, outputs):
            results.append({
                "instruction": q["instruction"],
                "output":      out,
                "dataset":     q.get("dataset", "alpaca_eval"),
                "generator":   f"{args.label}_backbone",
            })

        done = len(results)
        if done % 20 == 0 or done >= len(questions) or err:
            elapsed = time.time() - t0
            eta = (len(questions) - done) / (done / elapsed) if elapsed > 0 else 0
            tag = " [ERROR]" if err else ""
            print(f"[backbone] {args.label}: {done}/{len(questions)}"
                  f"  {elapsed:.0f}s 已用  {eta:.0f}s 剩余  errors={n_err}{tag}",
                  flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), ensure_ascii=False, indent=2)
    total = time.time() - t0
    print(f"[backbone] {args.label} 完成: {len(results)} 条"
          f"  errors={n_err}  总耗时={total:.0f}s ({total/60:.1f}min)"
          f"  → {args.output}", flush=True)


if __name__ == "__main__":
    main()

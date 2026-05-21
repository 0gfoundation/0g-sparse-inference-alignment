"""
将当前 RM（Qwen3-4B + LoRA + token_reward_head.pt）转成 vLLM 可直接加载的
HuggingFace 标准 SequenceClassification checkpoint。

转换流程：
  1. 加载 Qwen3-4B 作为 Qwen3ForSequenceClassification (num_labels=1)
  2. 加载 LoRA adapter → merge_and_unload()
  3. 加载 token_reward_head.pt（独立的 Linear(2560, 1)）
  4. 把 head 权重拷进 model.score 层
  5. 保存为标准 HF checkpoint（含 tokenizer）
  6. Sanity check：用 wrapper-style forward 算 vs 用标准 forward 算，对比 score

bias 处理：
  - Qwen3ForSequenceClassification.score 默认 bias=False
  - token_reward_head 通常有 bias 但值很小（实测 -0.0087）
  - SIA 把 5 个 candidate 的 reward 加到 logits 后采样，常数偏移完全不影响 ranking
  - 默认丢掉 bias 以保持模型架构干净（vLLM 加载更顺）
  - 如果 |bias| > 0.1 会 warn；可以 --keep_bias 强行保留

之后 vLLM serve：
  vllm serve <output_dir> --task token_classify \\
    --enable-prefix-caching --gpu-memory-utilization 0.3 --port 8001

Usage:
  python scripts/convert_rm_for_vllm.py \\
    --rm /workspace/SIA/models/Qwen3-4B \\
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \\
    --output /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
)


def parse_args():
    p = argparse.ArgumentParser(description="Convert RM to vLLM-compatible HF checkpoint")
    p.add_argument("--rm",       required=True,
                   help="Qwen3-4B base 模型路径")
    p.add_argument("--rm_lora",  required=True,
                   help="LoRA + token_reward_head 所在目录（含 lora_weights/ 子目录和 token_reward_head.pt）")
    p.add_argument("--output",   required=True,
                   help="输出 HF checkpoint 目录")
    p.add_argument("--dtype",    default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="保存权重 dtype")
    p.add_argument("--keep_bias", action="store_true",
                   help="保留 token_reward_head 的 bias（修改 score 层为 bias=True）。"
                        "默认丢掉 bias 以保持模型架构纯净，vLLM 加载更稳。")
    p.add_argument("--skip_verify", action="store_true",
                   help="跳过保存后的 sanity check（默认会做）")
    return p.parse_args()


def load_token_reward_head(rm_lora_path: str):
    head_path = os.path.join(rm_lora_path, "token_reward_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"token_reward_head.pt not found at {head_path}")
    state = torch.load(head_path, map_location='cpu', weights_only=True)
    head_w = state['token_reward_head']
    print(f"[convert] loaded token_reward_head:")
    print(f"          weight shape: {tuple(head_w['weight'].shape)}  dtype: {head_w['weight'].dtype}")
    print(f"          bias   shape: {tuple(head_w['bias'].shape)}  value: {head_w['bias'].item():.6f}")
    return head_w


def main():
    args = parse_args()
    target_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[args.dtype]

    print("=" * 70)
    print(f"RM base    : {args.rm}")
    print(f"RM LoRA dir: {args.rm_lora}")
    print(f"Output     : {args.output}")
    print(f"Dtype      : {args.dtype}")
    print(f"Keep bias  : {args.keep_bias}")
    print("=" * 70)

    # 0) 检查路径
    lora_subdir = os.path.join(args.rm_lora, "lora_weights")
    if not os.path.exists(lora_subdir):
        sys.exit(f"[convert] FATAL: LoRA dir not found: {lora_subdir}")

    # 1) 加载 base model as SequenceClassification (num_labels=1)
    print("\n[convert] Step 1/5: Loading base as Qwen3ForSequenceClassification(num_labels=1)...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm,
        num_labels=1,
        torch_dtype=target_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # 此时 model.score 是 Linear(hidden_size, 1, bias=False)，权重随机
    # 接下来会被 token_reward_head 覆盖

    # 2) 应用 LoRA 并 merge
    print("\n[convert] Step 2/5: Loading LoRA adapter and merging...")
    from peft import PeftModel
    base_model = PeftModel.from_pretrained(base_model, lora_subdir)
    base_model = base_model.merge_and_unload()
    print(f"[convert] LoRA merged. Model class: {type(base_model).__name__}")

    # 3) 加载 token_reward_head.pt
    print("\n[convert] Step 3/5: Loading token_reward_head.pt...")
    head_w = load_token_reward_head(args.rm_lora)

    # 4) 拷进 model.score
    print("\n[convert] Step 4/5: Copying token_reward_head → model.score...")
    score_module = base_model.score
    if not isinstance(score_module, nn.Linear):
        sys.exit(f"[convert] FATAL: model.score is not nn.Linear, got {type(score_module)}")

    expected_weight_shape = (1, score_module.weight.shape[1])
    if tuple(head_w['weight'].shape) != expected_weight_shape:
        sys.exit(
            f"[convert] FATAL: token_reward_head weight shape {tuple(head_w['weight'].shape)} "
            f"!= expected score shape {expected_weight_shape}"
        )

    bias_val = head_w['bias'].item()
    print(f"          token_reward_head.bias = {bias_val:.6f}")

    if args.keep_bias:
        if score_module.bias is None:
            # 重建 score 层带 bias
            in_features = score_module.weight.shape[1]
            print(f"          --keep_bias set: rebuilding score as Linear({in_features}, 1, bias=True)")
            new_score = nn.Linear(in_features, 1, bias=True, dtype=target_dtype)
            new_score.weight.data.copy_(head_w['weight'].to(target_dtype))
            new_score.bias.data.copy_(head_w['bias'].to(target_dtype))
            base_model.score = new_score
        else:
            score_module.weight.data.copy_(head_w['weight'].to(target_dtype))
            score_module.bias.data.copy_(head_w['bias'].to(target_dtype))
    else:
        # 默认：只拷 weight，丢 bias
        if abs(bias_val) > 0.1:
            print(f"          ⚠️  bias magnitude {abs(bias_val):.4f} > 0.1, dropping it anyway "
                  f"(constant offset doesn't affect SIA candidate ranking; "
                  f"pass --keep_bias if you really need it)")
        else:
            print(f"          bias magnitude {abs(bias_val):.6f} negligible; dropping it "
                  f"(keeps score layer compatible with Qwen3 standard architecture)")
        score_module.weight.data.copy_(head_w['weight'].to(target_dtype))

    # 5) 保存
    print(f"\n[convert] Step 5/5: Saving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    # 让 SequenceClassification forward 能处理 batch padding：必须设 pad_token_id
    tok = AutoTokenizer.from_pretrained(args.rm, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base_model.config.pad_token_id = tok.pad_token_id
    print(f"[convert] set model.config.pad_token_id = {tok.pad_token_id}")

    base_model.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    print(f"[convert] Saved model + tokenizer to {args.output}")

    if args.skip_verify:
        print("\n[convert] --skip_verify set, skipping sanity check.")
        _print_next_steps(args.output)
        return

    # 6) Sanity check
    print("\n" + "=" * 70)
    print("[verify] Comparing wrapper-style forward vs standard SequenceClassification forward")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sample_texts = [
        "The capital of France is Paris.",
        "Hello, how are you today? I'm doing great, thanks for asking.",
    ]

    enc = tok(sample_texts, return_tensors="pt", padding=True).to(device)
    base_model = base_model.to(device).eval()

    # 6a) "wrapper-style"：手动 backbone → token_reward_head（fp32）→ pick last valid token
    backbone = base_model.model if hasattr(base_model, "model") else base_model
    head_w_fp32 = head_w['weight'].to(device).float()
    head_b_fp32 = head_w['bias'].to(device).float() if args.keep_bias else None

    with torch.no_grad():
        backbone_out = backbone(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        hidden = backbone_out.last_hidden_state  # (batch, seq, hidden)
        # F.linear(hidden_fp32, head_w_fp32, head_b_fp32) → (batch, seq, 1)
        tok_rewards = hidden.float() @ head_w_fp32.T  # (batch, seq, 1)
        if head_b_fp32 is not None:
            tok_rewards = tok_rewards + head_b_fp32
        # pick last valid position per batch
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        wrapper_scores = tok_rewards[torch.arange(tok_rewards.size(0), device=device), seq_lens].squeeze(-1)
    print(f"[verify] wrapper-style scores: {wrapper_scores.tolist()}")

    # 6b) 用标准 SequenceClassification.forward
    with torch.no_grad():
        out = base_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
    standard_scores = out.logits.squeeze(-1)
    print(f"[verify] standard SeqCls scores: {standard_scores.tolist()}")

    diff = (wrapper_scores - standard_scores).abs()
    print(f"[verify] diff: max={diff.max().item():.6f}  mean={diff.mean().item():.6f}")

    # 容忍 bfloat16 精度差异（wrapper 用 fp32 算 head，标准 forward 用 target_dtype）
    tol = 0.05 if target_dtype == torch.bfloat16 else 0.001
    if diff.max().item() > tol:
        print(f"[verify] ⚠️  Max diff {diff.max().item():.6f} > tol {tol}. "
              f"This is usually just bfloat16 numerical noise, but inspect if suspicious.")
    else:
        print(f"[verify] ✅ Scores match within tol={tol}. Conversion looks correct.")

    if not args.keep_bias and abs(bias_val) > 0.01:
        print(f"[verify] note: bias was dropped (value={bias_val:.4f}). "
              f"Scores differ by ~{abs(bias_val):.4f} from original wrapper "
              f"but ranking among candidates is identical.")

    _print_next_steps(args.output)


def _print_next_steps(output: str):
    print("\n" + "=" * 70)
    print("Next steps")
    print("=" * 70)
    print(f"1) Start vLLM serve:")
    print(f"   vllm serve {output} \\")
    print(f"     --task token_classify \\")
    print(f"     --enable-prefix-caching \\")
    print(f"     --gpu-memory-utilization 0.3 \\")
    print(f"     --port 8001")
    print()
    print(f"2) Test with curl:")
    print(f"   curl -s http://localhost:8001/pooling \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\"model\": \"{os.path.basename(output)}\", \"input\": [\"hello world\"], \"task\": \"token_classify\"}}' | jq .")
    print()
    print(f"3) Or test offline with Python:")
    print(f"   python -c \"")
    print(f"   from vllm import LLM")
    print(f"   llm = LLM(model='{output}', task='token_classify')")
    print(f"   out = llm.encode(['hello world'])")
    print(f"   print(out[0].outputs.data)\"")


if __name__ == "__main__":
    main()

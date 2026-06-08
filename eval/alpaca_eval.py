"""
AlpacaEval generation client. 跑 805 条 helpful 指令 → save JSON.
跟 eval/mmlu_eval.py 一样是个 OpenAI-compatible HTTP 客户端, 跟 SIA server
或 raw vllm serve 都能搭。

用法:
    python eval/alpaca_eval.py \\
        --base_url http://localhost:8000/v1 \\
        --model 0GM-1.0-35B-A3B-0427 \\
        --output exp/alpaca_results.json \\
        --max_tokens 256 \\
        --temperature 1.0

可选 sampling params (跟 mmlu_eval.py 一致):
    --top_p, --top_k, --repetition_penalty
"""
import argparse
import json
import time
from pathlib import Path

import requests


def chat_completion(base_url, model, instruction, *,
                    max_tokens=256, temperature=1.0,
                    top_p=None, top_k=None, repetition_penalty=None,
                    disable_thinking=False, no_think_prompt=False,
                    ban_think_token=False, brief_instruction=False,
                    sia_weight=None, sia_topk=None, sia_entropy_threshold=None,
                    timeout=700):
    """
    POST 到 /v1/chat/completions (常规, 走 chat_template) 或者
    /v1/completions (raw prompt 模式, 跳过 chat_template — no_think_prompt=True 用此路径)。

    no_think_prompt 强制 prompt 完全不进入 Qwen3 thinking 模式:
      - 走 /v1/completions, 完全跳过 chat_template
      - 用 Human:\\n{instr}\\nAssistant:\\n raw 文本 (跟官方 SIA 14B A1 evaluate.py 一致)
      - 模型在这种 non-Qwen-chat 格式下不会 emit <think> (实测验证)
      - 比 --disable_thinking (chat_template 仍塞空 <think></think> 块且模型仍可能受
        thinking-pattern 训练先验影响) 更彻底
    """
    sampling_kwargs = {}
    for k, v in [("top_p", top_p), ("top_k", top_k),
                 ("repetition_penalty", repetition_penalty)]:
        if v is not None:
            sampling_kwargs[k] = v

    # brief_instruction 模式: 在 user content 前加一句简短指令, 让模型 think 和 answer 都简短
    if brief_instruction:
        instruction = (
            "Please keep both your reasoning (inside <think>) and your final answer "
            "concise and to the point. Avoid unnecessary elaboration.\n\n"
            + instruction
        )

    if no_think_prompt:
        # raw /v1/completions, 跳过 chat_template, 用 Human/Assistant raw 文本
        # (跟官方 14B A1 evaluate.py 完全一致, non-Qwen-chat 格式让模型不进入 thinking)
        prompt = f"Human:\n{instruction}\nAssistant:\n"
        if ban_think_token:
            # 即使 prompt 无 <think>, 0GM 训练先验仍可能自发 emit <think>。
            # 用 vllm bad_words 把 <think> / </think> token 在 sampler 端禁掉。
            sampling_kwargs["bad_words"] = ["<think>", "</think>"]
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **sampling_kwargs,
        }
        resp = requests.post(
            f"{base_url.rstrip('/')}/completions",
            json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
        n_tok = data.get("usage", {}).get("completion_tokens", -1)
        return text, n_tok

    # 默认路径: /v1/chat/completions
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        **sampling_kwargs,
    }
    if disable_thinking:
        # Qwen3 / 0GM chat template 支持 enable_thinking=false:
        # 在 assistant 起始处插入 '<think>\\n\\n</think>\\n\\n' (空 thinking block),
        # 不是真的无 think tag (要彻底去掉用 --no_think_prompt 走 raw completions)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    sia_extra = {}
    if sia_weight is not None:
        sia_extra["sia_weight"] = sia_weight
    if sia_topk is not None:
        sia_extra["sia_topk"] = sia_topk
    if sia_entropy_threshold is not None:
        sia_extra["sia_entropy_threshold"] = sia_entropy_threshold
    payload.update(sia_extra)
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    n_tok = data.get("usage", {}).get("completion_tokens", -1)
    return text, n_tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset",
                   default="/workspace/SIA/data/alpaca_eval/alpaca_eval.json")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="只跑前 N 条 (default 全 805)")
    p.add_argument("--max_tokens", type=int, default=256,
                   help="跟官方 evaluate.sh 一致 (默认 256)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--disable_thinking", action="store_true",
                   help="给 chat_template 传 enable_thinking=false, 让模型跳过 "
                        "<think>...</think> 直接输出答案 (Qwen3 / 0GM 系列支持). "
                        "注意: 实际是在 prompt 塞空 <think></think> 块, 不是真的无 think tag")
    p.add_argument("--no_think_prompt", action="store_true",
                   help="走 raw /v1/completions, prompt 完全没有 <think> 标记 "
                        "(比 --disable_thinking 更彻底, 避开 chat template 强塞)")
    p.add_argument("--ban_think_token", action="store_true",
                   help="(仅 --no_think_prompt 时生效) 通过 vllm bad_words 在 sampler 层 "
                        "禁止 <think>/</think> token 被采样, 强制模型不进 thinking")
    p.add_argument("--brief_instruction", action="store_true",
                   help="在 user 指令前 prepend 简短性要求: 让 thinking 和 final answer 都简短, "
                        "缩短生成长度同时仍保留 thinking 结构")
    p.add_argument("--sia_weight", type=float, default=None,
                   help="per-request SIA weight 覆盖 (0.0 = 完全关闭 SIA, 退化为纯 vLLM; "
                        "留空 = 沿用 server 启动时的 --weight 全局默认值)")
    p.add_argument("--sia_topk", type=int, default=None,
                   help="per-request SIA topk 覆盖 (留空 = 沿用全局 --topk)")
    p.add_argument("--sia_entropy_threshold", type=float, default=None,
                   help="per-request entropy gate 覆盖 (0 = 每 token 都干预; 留空 = 沿用全局)")
    args = p.parse_args()

    data = json.load(open(args.dataset))
    if args.limit:
        data = data[:args.limit]
    print(f"loaded {len(data)} prompts from {args.dataset}")
    print(f"base_url    : {args.base_url}")
    print(f"model       : {args.model}")
    print(f"max_tokens  : {args.max_tokens}")
    print(f"temperature : {args.temperature}")
    print(f"top_p / top_k / rep_penalty: "
          f"{args.top_p} / {args.top_k} / {args.repetition_penalty}")
    print(f"disable_thinking: {args.disable_thinking}")
    print(f"no_think_prompt : {args.no_think_prompt}")
    print(f"sia_weight / sia_topk / sia_entropy_threshold: "
          f"{args.sia_weight} / {args.sia_topk} / {args.sia_entropy_threshold}")
    print(f"output      : {args.output}")
    print("-" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t0_total = time.time()
    for idx, row in enumerate(data):
        instr = row["instruction"]
        t0 = time.time()
        try:
            text, ntok = chat_completion(
                args.base_url, args.model, instr,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                disable_thinking=args.disable_thinking,
                no_think_prompt=args.no_think_prompt,
                ban_think_token=args.ban_think_token,
                brief_instruction=args.brief_instruction,
                sia_weight=args.sia_weight,
                sia_topk=args.sia_topk,
                sia_entropy_threshold=args.sia_entropy_threshold,
            )
        except Exception as e:
            text, ntok = f"ERROR: {e}", -1
        elapsed = time.time() - t0
        results.append({
            "id": idx + 1,
            "instruction": instr,
            "output": text,
            "tokens": ntok,
            "elapsed": elapsed,
            # 沿用官方 measure_reward.py 期望的字段名 (兼容现有打分逻辑):
            "prompt": f"Human:\n{instr}\nAssistant:\n",
            "result": f"Human:\n{instr}\nAssistant:\n{text}",
        })
        if (idx + 1) % 25 == 0 or idx == 0:
            total_elapsed = time.time() - t0_total
            tps = sum(r.get("tokens", 0) or 0 for r in results) / total_elapsed
            avg_lat = total_elapsed / (idx + 1)
            print(f"[{idx+1}/{len(data)}] last={elapsed:.1f}s "
                  f"avg={avg_lat:.1f}s/q  cum_tps={tps:.1f} tok/s")
            # 中途保存
            json.dump(results, open(args.output, "w"),
                      ensure_ascii=False, indent=2)

    json.dump(results, open(args.output, "w"),
              ensure_ascii=False, indent=2)
    total_elapsed = time.time() - t0_total
    total_tokens = sum(r.get("tokens", 0) or 0 for r in results)
    n_err = sum(1 for r in results if str(r["output"]).startswith("ERROR:"))
    print()
    print(f"=== summary ===")
    print(f"  saved {len(results)} results to {args.output}")
    print(f"  errors: {n_err}")
    print(f"  total tokens: {total_tokens}")
    print(f"  total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  throughput: {total_tokens/total_elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()

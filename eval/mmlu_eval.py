#!/usr/bin/env python3
"""
MMLU-Redux 生成式评测脚本

对 baseline（vLLM 原生 server）和 SIA server 使用完全相同的评测逻辑，
通过 --base_url 切换，结果可直接对比。

数据集：edinburgh-dawg/mmlu-redux（30 个子学科，每科约 100 题）
打分：exact_match（从模型输出中提取 A/B/C/D，与标准答案字符串匹配）

依赖：
  pip install datasets requests tqdm

用法：
  # 快速冒烟测试（每科只跑 3 题）
  python eval/mmlu_eval.py \\
    --base_url http://localhost:8000/v1 \\
    --model Qwen3-14B \\
    --output results/test.json \\
    --limit 3

  # baseline 完整评测
  python eval/mmlu_eval.py \\
    --base_url http://localhost:8000/v1 \\
    --model Qwen3-14B \\
    --output results/Qwen3-14B_noSIA_mmlu.json

  # SIA 完整评测（先启动 SIA server，其余参数相同）
  python eval/mmlu_eval.py \\
    --base_url http://localhost:8000/v1 \\
    --model Qwen3-14B \\
    --output results/Qwen3-14B_SIA_mmlu.json

  # 只跑部分子学科
  python eval/mmlu_eval.py \\
    --base_url http://localhost:8000/v1 \\
    --model Qwen3-14B \\
    --subjects anatomy astronomy college_biology \\
    --output results/partial.json
"""

import argparse
import time
import json
import re
import sys
from pathlib import Path

import requests
from datasets import load_dataset


# ---------------------------------------------------------------------------
# edinburgh-dawg/mmlu-redux 实际可用的 30 个子学科
# ---------------------------------------------------------------------------
ALL_SUBJECTS = [
    "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "conceptual_physics",
    "econometrics", "electrical_engineering", "formal_logic",
    "global_facts", "high_school_chemistry", "high_school_geography",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "high_school_us_history",
    "human_aging", "logical_fallacies", "machine_learning", "miscellaneous",
    "philosophy", "professional_accounting", "professional_law",
    "public_relations", "virology",
]

CHOICE_LETTERS = ["A", "B", "C", "D"]

# 系统提示：允许 Qwen3 thinking，但要求简洁推理，答案以 "Answer: X" 结尾
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following multiple choice question. "
    "You may think before answering, but keep your reasoning concise and under 500 tokens. "
    "End your response with exactly: Answer: X "
    "(where X is A, B, C, or D)."
)


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def format_prompt(question: str, choices: list[str]) -> str:
    lines = [question]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 答案提取
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """
    从模型输出中提取 A/B/C/D。
    优先全文匹配 "Answer: X"（含 thinking 内部），再从 </think> 后提取独立字母行。
    """
    # 全文优先：匹配 Answer: + 可选空格 + A/B/C/D（不紧跟其他字母）
    m = re.search(r"[Aa]nswer:\s*\*{0,2}([ABCD])\*{0,2}(?![a-zA-Z])", text)
    if m:
        return m.group(1).upper()

    # 兜底：</think> 之后的独立字母行
    after_think = text.split("</think>", 1)[-1] if "</think>" in text else ""
    after_think = after_think.strip()
    m = re.search(r"(?:^|\n)\s*([ABCD])\s*(?:\.|,|\s|$)", after_think)
    if m:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# HTTP 调用
# ---------------------------------------------------------------------------

def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "dummy",
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> tuple[str, int]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", -1)
        return text, tokens
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e


# ---------------------------------------------------------------------------
# 单个子学科评测
# ---------------------------------------------------------------------------

def evaluate_subject(
    base_url: str,
    model: str,
    subject: str,
    api_key: str = "dummy",
    limit: int | None = None,
    temperature: float = 1.0,
) -> dict:
    try:
        dataset = load_dataset(
            "edinburgh-dawg/mmlu-redux", subject, split="test"
        )
    except Exception as e:
        print(f"  [WARN] Failed to load subject '{subject}': {e}", file=sys.stderr)
        return {"accuracy": None, "correct": 0, "total": 0, "details": []}

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    details = []

    for item in dataset:
        question = item["question"]
        choices = item["choices"]          # list of 4 strings
        answer_idx = item["answer"]        # integer 0-3
        answer_letter = CHOICE_LETTERS[answer_idx]

        prompt = format_prompt(question, choices)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        def log(s=""):
            print(s, flush=True)

        log(f"\n{'='*60}")
        log(f"[{subject}] Q{len(details)+1}  answer={answer_letter}")
        log(prompt)
        log(f"{'─'*60}")

        t0 = time.time()
        tokens = -1
        try:
            response_text, tokens = chat_completion(
                base_url, model, messages, api_key=api_key,
                max_tokens=2048, temperature=temperature,
            )
            predicted = extract_answer(response_text)
            is_correct = predicted == answer_letter
        except Exception as e:
            response_text = f"ERROR: {e}"
            predicted = None
            is_correct = False
        elapsed = time.time() - t0

        log(response_text)
        log(f"→ truth={answer_letter}  predicted={predicted}  correct={is_correct}  tokens={tokens}  latency={elapsed:.1f}s")

        correct += int(is_correct)
        details.append({
            "question": question,
            "choices": choices,
            "answer": answer_letter,
            "predicted": predicted,
            "response": response_text,
            "correct": is_correct,
            "tokens": tokens if tokens >= 0 else 0,
        })

    total = len(details)
    accuracy = correct / total if total > 0 else 0.0
    total_tokens = sum(d.get("tokens", 0) or 0 for d in details)
    return {"accuracy": accuracy, "correct": correct, "total": total, "total_tokens": total_tokens, "details": details}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MMLU-Redux 生成式评测（baseline / SIA 通用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base_url", default="http://localhost:8000/v1",
                        help="Server base URL（默认 http://localhost:8000/v1）")
    parser.add_argument("--model", required=True,
                        help="模型名称（需与 server 暴露的 model id 一致）")
    parser.add_argument("--output", required=True,
                        help="结果保存路径（JSON）")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="指定子学科列表，默认跑全部 57 个")
    parser.add_argument("--limit", type=int, default=None,
                        help="每个子学科限制题数，用于快速测试")
    parser.add_argument("--api_key", default="dummy",
                        help="API key（vLLM 不校验，传任意字符串即可）")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度（默认 1.0）")
    args = parser.parse_args()

    subjects = args.subjects or ALL_SUBJECTS
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"base_url    : {args.base_url}")
    print(f"model       : {args.model}")
    print(f"subjects    : {len(subjects)} 个")
    print(f"limit       : {args.limit or '无'}")
    print(f"temperature : {args.temperature}")
    print(f"output      : {output_path}")
    print("-" * 60)

    per_subject: dict[str, dict] = {}
    total_correct = 0
    total_count = 0
    total_tokens = 0
    eval_start = time.time()

    for subject in subjects:
        result = evaluate_subject(
            args.base_url, args.model, subject,
            api_key=args.api_key, limit=args.limit,
            temperature=args.temperature,
        )
        per_subject[subject] = result
        if result["accuracy"] is not None:
            total_correct += result["correct"]
            total_count += result["total"]
            total_tokens += result.get("total_tokens", 0)
            print(
                f"  {subject:<45} {result['correct']:>3}/{result['total']:<3}"
                f"  acc={result['accuracy']:.3f}",
                flush=True,
            )

    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    # 保存结果（details 单独存，summary 放顶层方便查看）
    output = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "subjects": subjects,
            "limit": args.limit,
        },
        "summary": {
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_count": total_count,
            "per_subject": {
                s: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
                for s, v in per_subject.items()
            },
        },
        "details": per_subject,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - eval_start
    avg_latency = total_elapsed / total_count if total_count > 0 else 0.0
    avg_tokens = total_tokens / total_count if total_count > 0 else 0.0
    tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
    print("-" * 60)
    print(f"Overall accuracy: {overall_acc:.4f}  ({total_correct}/{total_count})")
    print(f"Total latency   : {total_elapsed:.1f}s  avg={avg_latency:.1f}s/q")
    print(f"Avg token length: {avg_tokens:.1f} tokens/q  (total={total_tokens})")
    print(f"Throughput      : {tps:.1f} tokens/s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

"""SIA generation client — sends AlpacaEval questions to two SIA servers,
collects text responses, and saves JSON compatible with measure_alpaca_reward.py.

Both servers run concurrently (one thread each) so total wall-clock time equals
the slower server, not their sum.

Usage
-----
    python scripts/vm_sia_client.py \\
        --scalar http://localhost:8001 \\
        --vocab  http://localhost:8002 \\
        --output_dir exp/alpaca-vm-comparison-20260706

    # Custom generation settings:
    python scripts/vm_sia_client.py \\
        --scalar http://localhost:8001 \\
        --vocab  http://localhost:8002 \\
        --output_dir exp/alpaca-vm-comparison-20260706 \\
        --max_tokens 512 --temperature 0.7 --limit 200

    # After the client finishes, score with Skywork (on the host or in the container):
    python scripts/measure_alpaca_reward.py \\
        --input_file  exp/alpaca-vm-comparison-20260706/scalar_sia.json \\
        --output_file exp/alpaca-vm-comparison-20260706/scalar_sia_scored.json \\
        --rm /workspace/models/Skywork-Reward-V2-Llama-3.1-8B \\
        --device cuda:0

    python scripts/measure_alpaca_reward.py \\
        --input_file  exp/alpaca-vm-comparison-20260706/vocab_sia.json \\
        --output_file exp/alpaca-vm-comparison-20260706/vocab_sia_scored.json \\
        --rm /workspace/models/Skywork-Reward-V2-Llama-3.1-8B \\
        --device cuda:0

Output format (compatible with measure_alpaca_reward.py)
---------------------------------------------------------
    [{"instruction": "...", "output": "...", "dataset": "...", "generator": "..."}, ...]
"""

import argparse
import json
import time
import threading
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _health(url: str, label: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
        print(f"  [{label}] {url}/health → {r.json()}", flush=True)
        return True
    except Exception as e:
        print(f"  [{label}] health FAILED: {e}", flush=True)
        return False


def _chat(base_url: str, model_id: str, instruction: str, *,
          max_tokens: int, temperature: float, timeout: int = 600) -> str:
    """POST /v1/chat/completions → assistant text content."""
    payload = {
        "model":    model_id,
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Per-server generation loop
# ---------------------------------------------------------------------------

def generate_all(base_url: str, model_id: str, questions: list,
                 max_tokens: int, temperature: int,
                 label: str, results: list) -> None:
    """Generate responses for all questions sequentially.

    Writes dicts into `results` in order.
    Called in its own thread.
    """
    t0 = time.time()
    for i, q in enumerate(questions):
        instr = q["instruction"]
        try:
            output = _chat(
                base_url, model_id, instr,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            err = False
        except Exception as e:
            output = f"ERROR: {e}"
            err = True

        results.append({
            "instruction": instr,
            "output":      output,
            "dataset":     q.get("dataset", "alpaca_eval"),
            "generator":   model_id,
        })

        done = i + 1
        if done % 10 == 0 or done == len(questions) or err:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(questions) - done) / rate if rate > 0 else 0
            status = " [ERROR]" if err else ""
            print(f"  [{label}] {done}/{len(questions)}  "
                  f"{elapsed:.0f}s elapsed  {eta:.0f}s ETA{status}",
                  flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scalar",     default="http://localhost:8001",
                   help="scalar-head SIA server URL")
    p.add_argument("--vocab",      default="http://localhost:8002",
                   help="vocab_lowrank SIA server URL")
    p.add_argument("--data",
                   default="data/alpaca_eval/alpaca_eval.json",
                   help="AlpacaEval dataset path")
    p.add_argument("--output_dir", required=True,
                   help="directory where scalar_sia.json and vocab_sia.json are saved")
    p.add_argument("--limit",      type=int, default=200,
                   help="number of questions (default 200; 0 = all 805)")
    p.add_argument("--max_tokens", type=int, default=512,
                   help="generation max tokens (default 512)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature (default 1.0)")
    p.add_argument("--scalar_model", default="Qwen3-4B-SIA-scalar",
                   help="model_id string for the scalar server (matches --model_id in server)")
    p.add_argument("--vocab_model",  default="Qwen3-4B-SIA-vocab",
                   help="model_id string for the vocab server")
    p.add_argument("--tag", default="sia",
                   help="output filename tag: scalar_{tag}.json / vocab_{tag}.json (default: sia)")
    args = p.parse_args()

    # ── Load questions ────────────────────────────────────────────────────────
    questions = json.load(open(args.data))
    if args.limit:
        questions = questions[: args.limit]
    print(f"已加载 {len(questions)} 条问题，来源：{args.data}", flush=True)

    # ── Health checks ─────────────────────────────────────────────────────────
    print("\n健康检查：")
    ok_s = _health(args.scalar, "scalar")
    ok_v = _health(args.vocab,  "vocab ")
    if not (ok_s and ok_v):
        print("\n错误：一个或两个服务器无法连接。")
        print("  请先启动容器：")
        print("    docker compose -f docker-compose.vm_sia.yml up -d")
        print("  等待两个 /health 返回 200 后再运行此脚本：")
        print("    docker compose -f docker-compose.vm_sia.yml logs -f")
        return

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scalar_path = out_dir / f"scalar_{args.tag}.json"
    vocab_path  = out_dir / f"vocab_{args.tag}.json"

    # ── Generate concurrently ─────────────────────────────────────────────────
    scalar_results: list = []
    vocab_results:  list = []

    t_scalar = threading.Thread(
        target=generate_all,
        args=(args.scalar, args.scalar_model, questions,
              args.max_tokens, args.temperature, "scalar", scalar_results),
        daemon=True,
    )
    t_vocab = threading.Thread(
        target=generate_all,
        args=(args.vocab, args.vocab_model, questions,
              args.max_tokens, args.temperature, "vocab ", vocab_results),
        daemon=True,
    )

    t0 = time.time()
    print(f"\n开始并发生成：{len(questions)} 条问题，两个服务器同时跑 ...")
    print(f"  scalar_model={args.scalar_model}  max_tokens={args.max_tokens}  "
          f"temperature={args.temperature}\n")

    t_scalar.start()
    t_vocab.start()
    t_scalar.join()
    t_vocab.join()

    total_time = time.time() - t0
    print(f"\n两个服务器全部完成，总耗时 {total_time:.0f}s（{total_time/60:.1f} 分钟）", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    json.dump(scalar_results, open(scalar_path, "w"), ensure_ascii=False, indent=2)
    json.dump(vocab_results,  open(vocab_path,  "w"), ensure_ascii=False, indent=2)
    print(f"  已保存：{scalar_path}")
    print(f"  已保存：{vocab_path}")

    # ── Error summary ─────────────────────────────────────────────────────────
    n_err_s = sum(1 for r in scalar_results if str(r["output"]).startswith("ERROR:"))
    n_err_v = sum(1 for r in vocab_results  if str(r["output"]).startswith("ERROR:"))
    print(f"\n  错误数 — scalar: {n_err_s}/{len(scalar_results)}  "
          f"vocab: {n_err_v}/{len(vocab_results)}")

    # ── Next-step instructions ────────────────────────────────────────────────
    rm_path = "/workspace/models/Skywork-Reward-V2-Llama-3.1-8B"
    print(f"""
下一步：用 Skywork 对两组输出打分（需要在有 Skywork 模型的机器上运行）。

Option A — in the Docker container:
  docker exec -it vm-sia-comparison \\
    /opt/venv-vl30b/bin/python /workspace/sia-repo/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \\
      --input_file  {scalar_path} \\
      --output_file {out_dir}/scalar_sia_scored.json \\
      --rm {rm_path} --device cuda:0

  docker exec -it vm-sia-comparison \\
    /opt/venv-vl30b/bin/python /workspace/sia-repo/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \\
      --input_file  {vocab_path} \\
      --output_file {out_dir}/vocab_sia_scored.json \\
      --rm {rm_path} --device cuda:0

Option B — on the host (with transformers venv active):
  python scripts/measure_alpaca_reward.py \\
      --input_file  {scalar_path} \\
      --output_file {out_dir}/scalar_sia_scored.json \\
      --rm {rm_path} --device cuda:0

  python scripts/measure_alpaca_reward.py \\
      --input_file  {vocab_path} \\
      --output_file {out_dir}/vocab_sia_scored.json \\
      --rm {rm_path} --device cuda:0
""")


if __name__ == "__main__":
    main()

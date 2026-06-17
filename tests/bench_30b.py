"""
30B SIA 服务压测

两种扫描模式：
  1. Concurrency Sweep: input≈256 tokens, max_out=128, 并发 1/2/4/8/16
  2. Input-Length Sweep: 变化 input 长度，超过 max_model_len=2048 的组跳过

指标：TTFT均值/p99, ITL均值, Req Latency均值, Output tok/s, Req/s

依赖：aiohttp（pip install aiohttp）

用法：
    python tests/bench_30b.py                     # 两个 sweep（SIA 开启）
    python tests/bench_30b.py --no-sia            # SIA 关闭（纯 vLLM 基线）
    python tests/bench_30b.py --compare           # SIA vs noSIA 对比
    python tests/bench_30b.py --only-concurrency  # 只跑并发扫描
    python tests/bench_30b.py --only-input        # 只跑输入长度扫描
    python tests/bench_30b.py --rounds 3          # 每档重复轮数（默认 3）
"""
import argparse
import asyncio
import json
import statistics
import sys
import time

try:
    import aiohttp
except ImportError:
    print("缺少依赖：pip install aiohttp", file=sys.stderr)
    sys.exit(1)

URL_DEFAULT = "http://localhost:8000"
MAX_MODEL_LEN = 2048

# 基准句子：约 10 tokens / 45 chars
_PHRASE = "The quick brown fox jumps over the lazy dog. "
_PHRASE_TOKENS = 10

# ── Concurrency Sweep ──────────────────────────────────────────────────────
CONC_SWEEP_CONCURRENCIES = [1, 2, 4, 8, 16]
CONC_SWEEP_INPUT_TOKENS = 256
CONC_SWEEP_OUTPUT_TOKENS = 128

# ── Input-Length Sweep ─────────────────────────────────────────────────────
# (target_input_tokens, concurrency)
INPUT_SWEEP_CONFIGS = [
    (256,  2),
    (512,  2),
    (1024, 2),
    (1792, 1),
    (2048, 1),   # make_prompt 自动缩减到安全范围（含 output+template 后 ≤ 2048）
    (4096, 1),   # > MAX_MODEL_LEN → 自动跳过
]
INPUT_SWEEP_OUTPUT_TOKENS = 64

_TEMPLATE_OVERHEAD = 50


def make_prompt(target_input_tokens: int, reserve_output: int = 0) -> str:
    """构造约 target_input_tokens 的用户消息。
    只有当 target + reserve 接近 MAX_MODEL_LEN 时才缩减 prompt。
    """
    if target_input_tokens + reserve_output + _TEMPLATE_OVERHEAD > MAX_MODEL_LEN:
        effective = max(1, MAX_MODEL_LEN - _TEMPLATE_OVERHEAD - reserve_output)
    else:
        effective = max(1, target_input_tokens - _TEMPLATE_OVERHEAD)
    reps = max(1, int(effective / _PHRASE_TOKENS))
    return _PHRASE * reps + "Summarize the above text in one sentence."


def pct(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p), len(s) - 1)]


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    no_sia: bool = False,
) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if no_sia:
        payload["sia_weight"] = 0

    t0 = time.perf_counter()
    token_times: list = []
    prompt_tokens = 0
    completion_tokens = 0
    status = 0
    error = None

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            status = resp.status
            if status != 200:
                body = await resp.text()
                return {"status": status, "error": body[:300]}

            buf = b""
            done = False
            async for chunk_bytes in resp.content.iter_any():
                if done:
                    break
                buf += chunk_bytes
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    line = raw_line.strip().decode(errors="replace")
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content") or ""
                        if content:
                            token_times.append(time.perf_counter())

                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as exc:
        error = str(exc)

    if not token_times:
        return {"status": status or 0, "error": error or "no tokens received"}

    t_end = time.perf_counter()
    ttft_ms = (token_times[0] - t0) * 1000
    latency_ms = (t_end - t0) * 1000
    itl_ms = (
        statistics.mean(
            (token_times[i] - token_times[i - 1]) * 1000
            for i in range(1, len(token_times))
        )
        if len(token_times) > 1
        else 0.0
    )

    return {
        "status": 200,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def run_batch(
    url: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    rounds: int,
    no_sia: bool,
) -> tuple:
    """返回 (ok_results, is_overflow, wall_s)"""
    results = []
    async with aiohttp.ClientSession() as session:
        t_start = time.perf_counter()
        for _ in range(rounds):
            batch = await asyncio.gather(
                *[one_request(session, url, prompt, max_tokens, no_sia)
                  for _ in range(concurrency)]
            )
            results.extend(batch)
        wall_s = time.perf_counter() - t_start

    ok = [r for r in results if r.get("status") == 200]
    is_overflow = any(
        r.get("status") == 400 and
        any(kw in r.get("error", "").lower() for kw in ("context", "length", "token"))
        for r in results
    )
    return ok, is_overflow, wall_s


def compute_metrics(ok: list, wall_s: float) -> dict:
    if not ok:
        return {}
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
    itls  = [r["itl_ms"]  for r in ok if r.get("itl_ms") and r["itl_ms"] > 0]
    lats  = [r["latency_ms"] for r in ok if r.get("latency_ms")]
    pts   = [r["prompt_tokens"] for r in ok]
    cts   = [r["completion_tokens"] for r in ok]
    return {
        "n":          len(ok),
        "input_mean": int(statistics.mean(pts)) if pts else 0,
        "out_mean":   int(statistics.mean(cts)) if cts else 0,
        "ttft_mean":  statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_p99":   pct(ttfts, 0.99),
        "itl_mean":   statistics.mean(itls) if itls else 0.0,
        "lat_mean":   statistics.mean(lats) if lats else 0.0,
        "out_tps":    sum(cts) / wall_s if wall_s > 0 else 0.0,
        "req_s":      len(ok) / wall_s if wall_s > 0 else 0.0,
    }


_COL = "  {conc:>4}  {inp:>7}  {out:>6}  {ttft_m:>10}  {ttft_p99:>9}  {itl:>9}  {lat:>9}  {tps:>10}  {rps:>6}"
_HDR = "  {:>4}  {:>7}  {:>6}  {:>10}  {:>9}  {:>9}  {:>9}  {:>10}  {:>6}".format(
    "Conc", "Input", "Output", "TTFT mean", "TTFT p99", "ITL mean",
    "Req Lat", "Out tok/s", "Req/s"
)
_SEP = "  " + "─" * (len(_HDR) - 2)


def print_table(rows: list, title: str):
    print(f"\n{'═' * len(_HDR)}")
    print(f"  {title}")
    print(_SEP)
    print(_HDR)
    print(_SEP)
    for row in rows:
        if row.get("skip"):
            reason = row.get("reason", "超过 max_model_len")
            print(f"  {row['conc']:>4}  {row['target']:>7}  {'—':>6}  SKIP ({reason})")
        else:
            m = row["m"]
            print(_COL.format(
                conc=row["conc"],
                inp=m["input_mean"],
                out=m["out_mean"],
                ttft_m=f"{m['ttft_mean']:.0f}ms",
                ttft_p99=f"{m['ttft_p99']:.0f}ms",
                itl=f"{m['itl_mean']:.1f}ms",
                lat=f"{m['lat_mean']:.0f}ms",
                tps=f"{m['out_tps']:.1f}",
                rps=f"{m['req_s']:.2f}",
            ))
    print(_SEP)


async def do_concurrency_sweep(url: str, rounds: int, no_sia: bool):
    label = "noSIA" if no_sia else "SIA"
    prompt = make_prompt(CONC_SWEEP_INPUT_TOKENS, CONC_SWEEP_OUTPUT_TOKENS)
    rows = []
    print(f"\n[{label}] Concurrency Sweep — 进行中...")
    for conc in CONC_SWEEP_CONCURRENCIES:
        print(f"  conc={conc:>2}  ({conc * rounds} 请求)...", end=" ", flush=True)
        ok, overflow, wall_s = await run_batch(
            url, prompt, CONC_SWEEP_OUTPUT_TOKENS, conc, rounds, no_sia
        )
        if overflow:
            print("SKIP")
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS, "skip": True, "reason": "overflow"})
        elif not ok:
            print("FAIL (0 ok)")
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS, "skip": True, "reason": "all failed"})
        else:
            m = compute_metrics(ok, wall_s)
            print(f"ok={m['n']}/{conc * rounds}  TTFT={m['ttft_mean']:.0f}ms  ITL={m['itl_mean']:.1f}ms")
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS, "m": m})

    print_table(
        rows,
        f"Concurrency Sweep [{label}]  "
        f"(target_input={CONC_SWEEP_INPUT_TOKENS}, max_out={CONC_SWEEP_OUTPUT_TOKENS})"
    )


async def do_input_sweep(url: str, rounds: int, no_sia: bool):
    label = "noSIA" if no_sia else "SIA"
    rows = []
    print(f"\n[{label}] Input-Length Sweep — 进行中...")
    for (target_input, conc) in INPUT_SWEEP_CONFIGS:
        tag = f"input={target_input} conc={conc}"
        if target_input > MAX_MODEL_LEN:
            print(f"  {tag}  SKIP (target > max_model_len={MAX_MODEL_LEN})")
            rows.append({"conc": conc, "target": target_input, "skip": True,
                         "reason": f"> max_model_len={MAX_MODEL_LEN}"})
            continue

        prompt = make_prompt(target_input, INPUT_SWEEP_OUTPUT_TOKENS)
        print(f"  {tag}  ({conc * rounds} 请求)...", end=" ", flush=True)
        ok, overflow, wall_s = await run_batch(
            url, prompt, INPUT_SWEEP_OUTPUT_TOKENS, conc, rounds, no_sia
        )
        if overflow:
            print("SKIP (overflow)")
            rows.append({"conc": conc, "target": target_input, "skip": True, "reason": "overflow"})
        elif not ok:
            print("FAIL")
            rows.append({"conc": conc, "target": target_input, "skip": True, "reason": "all failed"})
        else:
            m = compute_metrics(ok, wall_s)
            print(f"ok={m['n']}/{conc * rounds}  TTFT={m['ttft_mean']:.0f}ms  ITL={m['itl_mean']:.1f}ms")
            rows.append({"conc": conc, "target": target_input, "m": m})

    print_table(
        rows,
        f"Input-Length Sweep [{label}]  (max_out={INPUT_SWEEP_OUTPUT_TOKENS})"
    )


async def check_health(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(description="30B SIA 压测")
    parser.add_argument("--url", default=URL_DEFAULT)
    parser.add_argument("--rounds", type=int, default=3, help="每档重复轮数（默认 3）")
    parser.add_argument("--no-sia", action="store_true", help="关闭 SIA（sia_weight=0）")
    parser.add_argument("--compare", action="store_true", help="SIA vs noSIA 对比")
    parser.add_argument("--only-concurrency", action="store_true", help="只跑并发扫描")
    parser.add_argument("--only-input", action="store_true", help="只跑输入长度扫描")
    args = parser.parse_args()

    print(f"目标服务: {args.url}")
    if not await check_health(args.url):
        print("❌ /health 不可达，服务未启动？", file=sys.stderr)
        sys.exit(1)
    print("✅ /health OK")

    run_conc  = not args.only_input
    run_input = not args.only_concurrency
    modes     = [False, True] if args.compare else [args.no_sia]

    for no_sia in modes:
        if run_conc:
            await do_concurrency_sweep(args.url, args.rounds, no_sia)
        if run_input:
            await do_input_sweep(args.url, args.rounds, no_sia)

    print("\n压测完成。")


if __name__ == "__main__":
    asyncio.run(main())

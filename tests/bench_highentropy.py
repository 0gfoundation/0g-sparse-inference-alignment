"""
High-Entropy SIA Stress Test

Uses a counterfactual reasoning prompt that produces ~18% SIA intervention
rate (vs ~4% from the phrase-repetition baseline), giving a more realistic
picture of SIA overhead under genuine reasoning workloads.

Prompt: "If the printing press had never been invented..."
Output cap: 512 tokens (longer than the 304-token default to let the model
finish its reasoning before hitting max_tokens).

Usage:
    python tests/bench_highentropy.py                    # stress test, SIA on
    python tests/bench_highentropy.py --no-sia           # noSIA baseline
    python tests/bench_highentropy.py --compare          # SIA vs noSIA
    python tests/bench_highentropy.py --stress-start 32  # start from conc=32
    python tests/bench_highentropy.py --fail-fast        # stop on first error
    python tests/bench_highentropy.py --stress-max-conc 256
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
MAX_MODEL_LEN = 32768

PROMPT = (
    "If the printing press had never been invented, how might the development "
    "of science and democracy have differed? Reason carefully through the counterfactual."
)
MAX_TOKENS = 512

# ── Stress test tuning ────────────────────────────────────────────────────────
_STRESS_DROP_THRESHOLD   = 0.60
_STRESS_DROP_CONSECUTIVE = 2
_STRESS_ERROR_THRESHOLD  = 0.50
_STRESS_OOM_KEYWORDS     = ("cuda out of memory", "oom", "out of memory",
                             "cudaerror", "device-side assert")


# ── Per-request streaming call ────────────────────────────────────────────────

async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    no_sia: bool = False,
) -> dict:
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
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

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
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
        return {"status": 0, "error": "timeout"}
    except Exception as exc:
        return {"status": 0, "error": str(exc)}

    if not token_times:
        return {"status": 0, "error": f"no tokens received (http={status})"}

    t_end = time.perf_counter()
    ttft_ms = (token_times[0] - t0) * 1000
    latency_ms = (t_end - t0) * 1000
    itl_ms = (
        statistics.mean(
            (token_times[i] - token_times[i - 1]) * 1000
            for i in range(1, len(token_times))
        )
        if len(token_times) > 1 else 0.0
    )
    return {
        "status": 200,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_batch(url: str, concurrency: int, rounds: int, no_sia: bool):
    results = []
    async with aiohttp.ClientSession() as session:
        t_start = time.perf_counter()
        for _ in range(rounds):
            batch = await asyncio.gather(
                *[one_request(session, url, no_sia) for _ in range(concurrency)]
            )
            results.extend(batch)
        wall_s = time.perf_counter() - t_start
    ok = [r for r in results if r.get("status") == 200]
    return ok, results, wall_s


# ── Metrics ───────────────────────────────────────────────────────────────────

def pct(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p), len(s) - 1)]


def compute_metrics(ok: list, wall_s: float) -> dict:
    if not ok:
        return {}
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
    itls  = [r["itl_ms"]  for r in ok if r.get("itl_ms") and r["itl_ms"] > 0]
    lats  = [r["latency_ms"] for r in ok if r.get("latency_ms")]
    pts   = [r.get("prompt_tokens", 0) for r in ok]
    cts   = [r.get("completion_tokens", 0) for r in ok]
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


# ── Table printing ────────────────────────────────────────────────────────────

_COL = "  {conc:>4}  {inp:>6}  {out:>6}  {ttft_m:>10}  {ttft_p99:>9}  {itl:>9}  {lat:>9}  {tps:>10}  {rps:>6}  {tpm:>7}"
_HDR = "  {:>4}  {:>6}  {:>6}  {:>10}  {:>9}  {:>9}  {:>9}  {:>10}  {:>6}  {:>7}".format(
    "Conc", "Input", "Output", "TTFT mean", "TTFT p99", "ITL mean",
    "Req Lat", "Out tok/s", "Req/s", "TPM"
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
            print(f"  {row['conc']:>4}  {'—':>6}  {'—':>6}  SKIP ({row.get('reason', '')})")
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
                tpm=f"{m['out_tps']*60:.0f}",
            ))
    print(_SEP)


# ── Concurrency escalation ────────────────────────────────────────────────────

def _next_conc(conc: int) -> int:
    if conc < 16:
        return conc * 2
    raw = conc * 1.5
    if raw < 100:
        return max(conc + 1, int(round(raw / 4) * 4))
    return max(conc + 1, int(round(raw / 16) * 16))


def _is_oom(results: list) -> bool:
    return any(
        any(kw in (r.get("error") or "").lower() for kw in _STRESS_OOM_KEYWORDS)
        for r in results
    )


def _is_server_down(results: list) -> bool:
    return bool(results) and all(r.get("status", 0) == 0 for r in results)


# ── Stress test ───────────────────────────────────────────────────────────────

async def do_stress_test(
    url: str,
    start_conc: int,
    max_conc: int,
    rounds: int,
    no_sia: bool,
    fail_fast: bool = False,
):
    label = "noSIA" if no_sia else "SIA"
    rows: list = []
    peak_tps = 0.0
    drop_count = 0
    stop_reason = ""
    conc = start_conc

    print(f"\n[{label}] High-Entropy Stress Test")
    print(f"  prompt: \"{PROMPT[:80]}...\"")
    print(f"  max_tokens={MAX_TOKENS}  start_conc={conc}  max_conc={max_conc}  rounds={rounds}")
    print(f"  停止条件：tok/s < 峰值×{_STRESS_DROP_THRESHOLD:.0%}（连续{_STRESS_DROP_CONSECUTIVE}档）"
          f" 或 失败率>{_STRESS_ERROR_THRESHOLD:.0%} 或 服务崩溃\n")

    while True:
        if conc > max_conc:
            stop_reason = f"已达上限 conc={max_conc}（可用 --stress-max-conc 调整）"
            break
        total_reqs = conc * rounds
        print(f"  conc={conc:>4}  ({total_reqs} 请求)...", end=" ", flush=True)

        try:
            ok, all_results, wall_s = await run_batch(url, conc, rounds, no_sia)
        except Exception as exc:
            print(f"CRASH ({exc})")
            rows.append({"conc": conc, "skip": True, "reason": f"异常: {exc}"})
            stop_reason = str(exc)
            break

        fail_rate = 1.0 - len(ok) / max(len(all_results), 1)

        if _is_server_down(all_results):
            print("CRASH (服务不可达)")
            rows.append({"conc": conc, "skip": True, "reason": "服务崩溃/不可达"})
            stop_reason = "服务崩溃/不可达"
            break

        if _is_oom(all_results):
            print("OOM")
            rows.append({"conc": conc, "skip": True, "reason": "CUDA OOM"})
            stop_reason = "CUDA OOM"
            break

        err_threshold = 0.0 if fail_fast else _STRESS_ERROR_THRESHOLD
        if fail_rate > err_threshold:
            print(f"FAIL (失败率 {fail_rate:.0%}，ok={len(ok)}/{total_reqs})")
            rows.append({"conc": conc, "skip": True,
                         "reason": f"失败率 {fail_rate:.0%}"})
            stop_reason = f"失败率 {fail_rate:.0%} > {err_threshold:.0%}"
            break

        m = compute_metrics(ok, wall_s)
        tps = m.get("out_tps", 0.0)
        rows.append({"conc": conc, "m": m})

        peak_tps = max(peak_tps, tps)
        drop_flag = peak_tps > 0 and tps < peak_tps * _STRESS_DROP_THRESHOLD
        drop_count = drop_count + 1 if drop_flag else 0

        status_tag = f"↓{tps/peak_tps:.0%}" if drop_flag else "✓"
        print(f"ok={len(ok)}/{total_reqs}  ITL={m['itl_mean']:.1f}ms  "
              f"tok/s={tps:.1f}  TPM={tps*60:.0f}  {status_tag}  "
              f"(fail={fail_rate:.0%})")

        if drop_count >= _STRESS_DROP_CONSECUTIVE:
            stop_reason = (f"tok/s={tps:.1f} < 峰值{peak_tps:.1f}×"
                           f"{_STRESS_DROP_THRESHOLD:.0%}，连续 {drop_count} 档")
            break

        next_c = _next_conc(conc)
        if next_c <= conc:
            next_c = conc + max(4, conc // 4)
        if next_c > max_conc:
            stop_reason = f"已达上限 conc={max_conc}（可用 --stress-max-conc 调整）"
            break
        conc = next_c

    print_table(
        rows,
        f"High-Entropy Stress Test [{label}]  (max_out={MAX_TOKENS})"
    )

    best = max(
        (r for r in rows if not r.get("skip")),
        key=lambda r: r["m"].get("out_tps", 0),
        default=None,
    )
    if best:
        bm = best["m"]
        print(f"\n  🏆 峰值吞吐：conc={best['conc']}  tok/s={bm['out_tps']:.1f}"
              f"  ITL={bm['itl_mean']:.1f}ms"
              f"  input={bm['input_mean']}tok  output={bm['out_mean']}tok")
    if stop_reason:
        print(f"  🛑 停止原因：{stop_reason}\n")


# ── Health check ──────────────────────────────────────────────────────────────

async def check_health(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{url}/health",
                             timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
    except Exception:
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="High-entropy SIA stress test (counterfactual reasoning prompt)"
    )
    parser.add_argument("--url", default=URL_DEFAULT)
    parser.add_argument("--no-sia", action="store_true", help="关闭 SIA（noSIA 基线）")
    parser.add_argument("--compare", action="store_true", help="SIA vs noSIA 对比")
    parser.add_argument("--stress-start", type=int, default=1,
                        help="起始并发数（默认 1）")
    parser.add_argument("--stress-rounds", type=int, default=2,
                        help="每档轮数（默认 2）")
    parser.add_argument("--stress-max-conc", type=int, default=512,
                        help="并发上限（默认 512）")
    parser.add_argument("--fail-fast", action="store_true",
                        help="任意请求失败立即停止")
    args = parser.parse_args()

    print(f"目标服务: {args.url}")
    if not await check_health(args.url):
        print("❌ /health 不可达，服务未启动？", file=sys.stderr)
        sys.exit(1)
    print("✅ /health OK")

    modes = [False, True] if args.compare else [args.no_sia]
    for no_sia in modes:
        await do_stress_test(
            args.url,
            start_conc=args.stress_start,
            max_conc=args.stress_max_conc,
            rounds=args.stress_rounds,
            no_sia=no_sia,
            fail_fast=args.fail_fast,
        )


if __name__ == "__main__":
    asyncio.run(main())

"""
35B SIA 服务压测 v2

两种扫描模式：
  1. Concurrency Sweep: input≈512 tokens, max_out=304, 并发 1/2/4/8/16
  2. Input-Length Sweep: 变化 input 长度，超过 max_model_len 的组跳过

极限压力模式：
  3. Stress Test: 从 conc=16 起按 ~1.5× 梯度递增，直到吞吐严重下滑
     或服务崩溃为止，测出最大可承载并发数。

指标：TTFT均值/p99, ITL均值, Req Latency均值, Output tok/s, Req/s

依赖：aiohttp（pip install aiohttp）

用法：
    python tests/bench_35b.py                     # 两个 sweep（SIA 开启）
    python tests/bench_35b.py --no-sia            # SIA 关闭（纯 vLLM 基线）
    python tests/bench_35b.py --compare           # SIA vs noSIA 对比
    python tests/bench_35b.py --only-concurrency  # 只跑并发扫描
    python tests/bench_35b.py --only-input        # 只跑输入长度扫描
    python tests/bench_35b.py --rounds 3          # 每档重复轮数（默认 3）
    python tests/bench_35b.py --stress            # 极限并发压测（自动递增至崩溃）
    python tests/bench_35b.py --stress --no-sia   # noSIA 极限压测
    python tests/bench_35b.py --stress --stress-start 32   # 从 conc=32 开始
    python tests/bench_35b.py --stress --stress-rounds 1   # 每档只跑 1 轮（更快）
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

# 基准句子：约 10 tokens / 45 chars（实测自 test_long_context.py）
_PHRASE = "The quick brown fox jumps over the lazy dog. "
_PHRASE_TOKENS = 10

# ── Concurrency Sweep ──────────────────────────────────────────────────────
CONC_SWEEP_CONCURRENCIES = [1, 2, 4, 8, 16]
CONC_SWEEP_INPUT_TOKENS = 512
CONC_SWEEP_OUTPUT_TOKENS = 304

# ── Input-Length Sweep ─────────────────────────────────────────────────────
# (target_input_tokens, concurrency)
INPUT_SWEEP_CONFIGS = [
    (512,   2),
    (2048,  2),
    (4096,  2),
    (16384, 1),
    (32768, 1),
    (65536, 1),   # > MAX_MODEL_LEN → 自动跳过
    (94000, 1),   # > MAX_MODEL_LEN → 自动跳过
]
INPUT_SWEEP_OUTPUT_TOKENS = 128

_TEMPLATE_OVERHEAD = 50  # chat template 约 50 tokens


def make_prompt(target_input_tokens: int, reserve_output: int = 0) -> str:
    """构造约 target_input_tokens 的用户消息。
    只有当 target + reserve 接近 MAX_MODEL_LEN 时才缩减 prompt，
    否则按 target 原值构造（小 input 不需要扣 reserve）。
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
        error = "timeout"
    except Exception as exc:
        error = str(exc)

    if not token_times:
        # 不管 HTTP 状态如何，没有收到任何 token 就算失败（status=0 → 不进 ok 列表）
        return {"status": 0, "error": f"no tokens received (http={status})"}

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


_COL = "  {conc:>4}  {inp:>7}  {out:>6}  {ttft_m:>10}  {ttft_p99:>9}  {itl:>9}  {lat:>9}  {tps:>10}  {rps:>6}  {tpm:>7}"
_HDR = "  {:>4}  {:>7}  {:>6}  {:>10}  {:>9}  {:>9}  {:>9}  {:>10}  {:>6}  {:>7}".format(
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
                tpm=f"{m['out_tps']*60:.0f}",
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


# ── Stress Test ────────────────────────────────────────────────────────────

# 停止条件参数
_STRESS_DROP_THRESHOLD   = 0.60   # tok/s 跌至峰值 60% 以下 → 判定严重下滑
_STRESS_DROP_CONSECUTIVE = 2      # 连续 N 档下滑才停（避免单点抖动误判）
_STRESS_ERROR_THRESHOLD  = 0.50   # 单档请求失败率超过 50% → 判定服务崩溃
_STRESS_OOM_KEYWORDS     = ("cuda out of memory", "oom", "out of memory",
                             "cuDAError", "device-side assert")


def _next_conc(conc: int) -> int:
    """递增策略：
    conc < 16  → 2×（1→2→4→8→16，覆盖低并发基线）
    conc >= 16 → ~1.5×，凑整到 4 的倍数（<100）或 16 的倍数（≥100）
    """
    if conc < 16:
        return conc * 2
    raw = conc * 1.5
    if raw < 100:
        return max(conc + 1, int(round(raw / 4) * 4))
    return max(conc + 1, int(round(raw / 16) * 16))


def _is_oom(results: list) -> bool:
    for r in results:
        err = r.get("error", "") or ""
        if any(kw in err.lower() for kw in _STRESS_OOM_KEYWORDS):
            return True
    return False


def _is_server_down(results: list) -> bool:
    """全部请求都是连接错误（非 HTTP 400/500）。"""
    if not results:
        return True
    return all(r.get("status", 0) == 0 for r in results)


async def do_stress_test(
    url: str,
    start_conc: int,
    max_conc: int,
    rounds: int,
    no_sia: bool,
):
    label = "noSIA" if no_sia else "SIA"
    prompt = make_prompt(CONC_SWEEP_INPUT_TOKENS, CONC_SWEEP_OUTPUT_TOKENS)
    rows: list = []

    peak_tps   = 0.0
    drop_count = 0      # 连续下滑计数
    stop_reason = ""
    conc = start_conc

    print(f"\n[{label}] Stress Test — 从 conc={conc} 开始，每档 {rounds} 轮，上限 conc={max_conc}")
    print(f"  输入 ≈{CONC_SWEEP_INPUT_TOKENS} tokens，max_out={CONC_SWEEP_OUTPUT_TOKENS}")
    print(f"  停止条件：tok/s < 峰值×{_STRESS_DROP_THRESHOLD:.0%}（连续{_STRESS_DROP_CONSECUTIVE}档）"
          f" 或 失败率>{_STRESS_ERROR_THRESHOLD:.0%} 或 服务崩溃 或 conc>{max_conc}\n")

    while True:
        total_reqs = conc * rounds
        print(f"  conc={conc:>4}  ({total_reqs} 请求)...", end=" ", flush=True)

        # 每档 timeout 随并发适当放宽（高并发时请求排队时间更长）
        timeout_s = max(300, conc * 20)
        try:
            all_results: list = []
            async with aiohttp.ClientSession() as session:
                t_start = time.perf_counter()
                for _ in range(rounds):
                    batch = await asyncio.gather(
                        *[one_request(session, url, prompt,
                                      CONC_SWEEP_OUTPUT_TOKENS, no_sia)
                          for _ in range(conc)]
                    )
                    all_results.extend(batch)
                wall_s = time.perf_counter() - t_start
        except Exception as exc:
            print(f"CRASH ({exc})")
            stop_reason = f"异常崩溃: {exc}"
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS,
                         "skip": True, "reason": stop_reason})
            break

        # ── 失败分析 ───────────────────────────────────────────────
        ok = [r for r in all_results if r.get("status") == 200]
        fail_rate = 1.0 - len(ok) / max(len(all_results), 1)

        if _is_server_down(all_results):
            print("CRASH (服务不可达)")
            stop_reason = "服务崩溃/不可达"
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS,
                         "skip": True, "reason": stop_reason})
            break

        if _is_oom(all_results):
            print("OOM")
            stop_reason = "CUDA OOM"
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS,
                         "skip": True, "reason": stop_reason})
            break

        if fail_rate > _STRESS_ERROR_THRESHOLD:
            print(f"FAIL (失败率 {fail_rate:.0%}，ok={len(ok)}/{total_reqs})")
            stop_reason = f"失败率 {fail_rate:.0%} > {_STRESS_ERROR_THRESHOLD:.0%}"
            rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS,
                         "skip": True, "reason": stop_reason})
            break

        # ── 正常有数据 ─────────────────────────────────────────────
        m = compute_metrics(ok, wall_s)
        tps = m.get("out_tps", 0.0)
        rows.append({"conc": conc, "target": CONC_SWEEP_INPUT_TOKENS, "m": m,
                     "fail_rate": fail_rate})

        peak_tps = max(peak_tps, tps)
        drop_flag = (peak_tps > 0 and tps < peak_tps * _STRESS_DROP_THRESHOLD)
        drop_count = drop_count + 1 if drop_flag else 0

        status_tag = f"↓{tps/peak_tps:.0%}" if drop_flag else "✓"
        print(f"ok={len(ok)}/{total_reqs}  ITL={m['itl_mean']:.1f}ms  "
              f"tok/s={tps:.1f}  TPM={tps*60:.0f}  {status_tag}  "
              f"(fail={fail_rate:.0%})")

        if drop_count >= _STRESS_DROP_CONSECUTIVE:
            stop_reason = (f"tok/s={tps:.1f} < 峰值{peak_tps:.1f}×{_STRESS_DROP_THRESHOLD:.0%}"
                           f"，连续 {drop_count} 档")
            break

        # ── 下一档 ─────────────────────────────────────────────────
        next_c = _next_conc(conc)
        if next_c <= conc:          # 防止整数舍入卡死
            next_c = conc + max(4, conc // 4)
        if next_c > max_conc:
            stop_reason = f"已达上限 conc={max_conc}（可用 --stress-max-conc 调整）"
            break
        conc = next_c

    # ── 汇总表 ──────────────────────────────────────────────────────────────
    print_table(
        rows,
        f"Stress Test [{label}]  "
        f"(input≈{CONC_SWEEP_INPUT_TOKENS}, max_out={CONC_SWEEP_OUTPUT_TOKENS})"
    )

    # 找出峰值行
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


async def main():
    parser = argparse.ArgumentParser(description="35B SIA 压测 v2")
    parser.add_argument("--url", default=URL_DEFAULT)
    parser.add_argument("--rounds", type=int, default=3, help="每档重复轮数（默认 3）")
    parser.add_argument("--no-sia", action="store_true", help="关闭 SIA（sia_weight=0）")
    parser.add_argument("--compare", action="store_true", help="SIA vs noSIA 对比")
    parser.add_argument("--only-concurrency", action="store_true", help="只跑并发扫描")
    parser.add_argument("--only-input", action="store_true", help="只跑输入长度扫描")
    # ── 极限压测 ──────────────────────────────────────────────────────────────
    parser.add_argument("--stress", action="store_true",
                        help="极限并发压测：从 --stress-start 开始自动递增到崩溃")
    parser.add_argument("--stress-start", type=int, default=1,
                        help="压测起始并发数（默认 1）")
    parser.add_argument("--stress-rounds", type=int, default=2,
                        help="每档重复轮数（默认 2，比常规压测少以加快探索）")
    parser.add_argument("--stress-max-conc", type=int, default=512,
                        help="压测并发上限，防止 KV cache OOM（默认 512）")
    args = parser.parse_args()

    print(f"目标服务: {args.url}")
    if not await check_health(args.url):
        print("❌ /health 不可达，服务未启动？", file=sys.stderr)
        sys.exit(1)
    print("✅ /health OK")

    # ── 极限压测模式 ──────────────────────────────────────────────────────────
    if args.stress:
        modes = [False, True] if args.compare else [args.no_sia]
        for no_sia in modes:
            await do_stress_test(
                args.url,
                start_conc=args.stress_start,
                max_conc=args.stress_max_conc,
                rounds=args.stress_rounds,
                no_sia=no_sia,
            )
        print("\n压测完成。")
        return

    # ── 普通扫描模式 ──────────────────────────────────────────────────────────
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

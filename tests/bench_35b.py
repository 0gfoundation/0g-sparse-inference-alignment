"""
35B SIA 服务压测：TTFT / 并发吞吐 / 限流探测

测量指标：
  TTFT      — 流式请求从发出到收到第一个非空 content token 的时间（ms）
  端到端延迟 — 请求完成总耗时（ms）
  单请求吞吐 — completion_tokens / 总耗时（tok/s）
  聚合吞吐   — N 并发下的总 completion_tokens / wall-clock 时间（tok/s）
  限流探测   — burst 并发，观察 429 / 排队行为

依赖：aiohttp（pip install aiohttp）

用法：
    python tests/bench_35b.py                          # SIA 开启（默认）
    python tests/bench_35b.py --no-sia                 # SIA 关闭（sia_weight=0）
    python tests/bench_35b.py --compare                # SIA vs noSIA 并排对比
    python tests/bench_35b.py --concurrency 1 2 4 --rounds 3 --burst 16
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

PROMPT = (
    "Please list all 8 planets of the solar system in order from the Sun, "
    "with one sentence about each planet."
)

BASE_STREAM_PAYLOAD = {
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 300,
    "temperature": 0.7,
    "stream": True,
    "stream_options": {"include_usage": True},
}


def make_payload(no_sia: bool) -> dict:
    p = dict(BASE_STREAM_PAYLOAD)
    if no_sia:
        p["sia_weight"] = 0
    return p

BURST_PAYLOAD = {
    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    "max_tokens": 10,
    "stream": False,
}


def pct(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


async def one_stream_request(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    """发一次流式请求，返回 TTFT、延迟、token 计数、状态码。"""
    t0 = time.perf_counter()
    ttft_ms = None
    prompt_tokens = 0
    completion_tokens = 0
    status = 0
    error = None

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            status = resp.status
            if status != 200:
                body = await resp.text()
                return {
                    "ttft_ms": None,
                    "total_ms": (time.perf_counter() - t0) * 1000,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "status": status,
                    "error": body[:300],
                }

            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # TTFT：第一个含实际 content 的 chunk
                if ttft_ms is None:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        ttft_ms = (time.perf_counter() - t0) * 1000

                # usage chunk（流式末尾）
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

    except asyncio.TimeoutError:
        error = "timeout"
        status = 0
    except Exception as exc:
        error = str(exc)
        status = 0

    return {
        "ttft_ms": ttft_ms,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "status": status,
        "error": error,
    }


async def run_sweep(url: str, concurrency: int, rounds: int, payload: dict) -> tuple:
    """返回 (results, wall_ms)。每轮并发 concurrency 个请求，共 rounds 轮。"""
    results = []
    async with aiohttp.ClientSession() as session:
        t_start = time.perf_counter()
        for _ in range(rounds):
            batch = await asyncio.gather(
                *[one_stream_request(session, url, payload) for _ in range(concurrency)]
            )
            results.extend(batch)
        wall_ms = (time.perf_counter() - t_start) * 1000
    return results, wall_ms


def print_sweep_summary(results: list, wall_ms: float, concurrency: int, rounds: int):
    ok = [r for r in results if r["status"] == 200]
    errs = [r for r in results if r["status"] != 200]

    total = len(results)
    print(f"  请求总数={total}  成功={len(ok)}  失败={len(errs)}")

    if errs:
        sc = {}
        for r in errs:
            sc[r["status"]] = sc.get(r["status"], 0) + 1
        for r in errs[:2]:
            if r.get("error"):
                print(f"  错误样本: {r['error'][:120]}")
        print(f"  失败状态码分布: {sc}")

    if not ok:
        return

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in ok]
    comp_list = [r["completion_tokens"] for r in ok]

    if ttfts:
        print(
            f"  TTFT (ms)       min={min(ttfts):.0f}  "
            f"p50={statistics.median(ttfts):.0f}  "
            f"p95={pct(ttfts, 0.95):.0f}  "
            f"max={max(ttfts):.0f}"
        )
    print(
        f"  端到端延迟(ms)  min={min(totals):.0f}  "
        f"p50={statistics.median(totals):.0f}  "
        f"p95={pct(totals, 0.95):.0f}  "
        f"max={max(totals):.0f}"
    )

    per_req_tps = [
        r["completion_tokens"] / (r["total_ms"] / 1000)
        for r in ok
        if r["total_ms"] > 0 and r["completion_tokens"] > 0
    ]
    if per_req_tps:
        print(
            f"  单请求吞吐(tok/s)  avg={statistics.mean(per_req_tps):.1f}  "
            f"median={statistics.median(per_req_tps):.1f}"
        )

    total_comp = sum(comp_list)
    agg_tps = total_comp / (wall_ms / 1000) if wall_ms > 0 else 0
    print(f"  聚合吞吐(tok/s)   {agg_tps:.1f}  (wall={wall_ms/1000:.1f}s, total_comp={total_comp})")


async def rate_limit_probe(url: str, burst: int):
    """burst 并发发送短请求，观察排队和 429 行为。"""
    print(f"\n── 限流探测（{burst} 并发 × 1 轮，短 prompt）──")

    async def quick(session: aiohttp.ClientSession):
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=BURST_PAYLOAD,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                await resp.read()
                return resp.status, (time.perf_counter() - t0) * 1000
        except Exception:
            return 0, (time.perf_counter() - t0) * 1000

    t_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[quick(session) for _ in range(burst)])
    wall_ms = (time.perf_counter() - t_start) * 1000

    sc: dict = {}
    latencies = []
    for status, ms in results:
        sc[status] = sc.get(status, 0) + 1
        latencies.append(ms)

    print(f"  wall-clock: {wall_ms/1000:.1f}s")
    print(f"  状态码分布: {sc}")
    if latencies:
        print(
            f"  延迟(ms)   min={min(latencies):.0f}  "
            f"p50={statistics.median(latencies):.0f}  "
            f"max={max(latencies):.0f}"
        )
    if 429 in sc:
        print(f"  ⚠ 检测到限流：{sc[429]} 个请求返回 429")
    else:
        print(f"  ✅ 无 429（推理服务本身无速率限制，请求在 vLLM 队列中排队）")


async def check_health(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
    except Exception:
        return False


async def run_mode(url: str, concurrency: list, rounds: int, payload: dict, label: str):
    """跑一组并发扫描并打印结果，label 用于区分 SIA / noSIA。"""
    for c in concurrency:
        total_req = c * rounds
        print(f"── [{label}] 并发度 {c}（{total_req} 请求 = {c} 并发 × {rounds} 轮）──")
        results, wall_ms = await run_sweep(url, c, rounds, payload)
        print_sweep_summary(results, wall_ms, c, rounds)
        print()


async def main():
    parser = argparse.ArgumentParser(description="35B SIA 压测")
    parser.add_argument("--url", default=URL_DEFAULT, help="服务地址")
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 2, 4],
        help="并发度列表（默认 1 2 4）"
    )
    parser.add_argument("--rounds", type=int, default=3, help="每个并发度重复轮数（默认 3）")
    parser.add_argument("--burst", type=int, default=16, help="限流探测并发数（默认 16）")
    parser.add_argument("--skip-ratelimit", action="store_true", help="跳过限流探测")
    parser.add_argument("--no-sia", action="store_true", help="关闭 SIA 干预（sia_weight=0）")
    parser.add_argument("--compare", action="store_true", help="依次跑 SIA 和 noSIA，方便对比")
    args = parser.parse_args()

    print(f"目标服务: {args.url}")

    # 健康检查
    if not await check_health(args.url):
        print("❌ /health 不可达，服务未启动？", file=sys.stderr)
        sys.exit(1)
    print("✅ /health OK\n")

    if args.compare:
        # SIA 开启
        print("═══ SIA 开启（sia_weight=default） ═══")
        await run_mode(args.url, args.concurrency, args.rounds, make_payload(False), "SIA")
        # SIA 关闭
        print("═══ SIA 关闭（sia_weight=0，纯 vLLM） ═══")
        await run_mode(args.url, args.concurrency, args.rounds, make_payload(True), "noSIA")
    else:
        label = "noSIA" if args.no_sia else "SIA"
        payload = make_payload(args.no_sia)
        await run_mode(args.url, args.concurrency, args.rounds, payload, label)

    # 限流探测
    if not args.skip_ratelimit:
        await rate_limit_probe(args.url, args.burst)

    print("\n压测完成。")


if __name__ == "__main__":
    asyncio.run(main())

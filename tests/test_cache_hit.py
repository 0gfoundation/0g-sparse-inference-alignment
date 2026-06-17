"""
V7  Prefix Cache 命中（cached_tokens）

连发两次相同 prompt，第二次应命中缓存：
  请求 1: cached_tokens == 0
  请求 2: cached_tokens > 0

适用于 35B（block_size≈1056）和 30B（block_size=16）。

用法：
    python tests/test_cache_hit.py [--url http://localhost:8000]
"""
import argparse, json, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# 200 次重复 ≈ 1220 tokens（超过 35B block_size=1056 和 30B block_size=16 两者阈值）
LONG_SYSTEM = "You are a helpful assistant. " * 200


def send_one(url: str) -> dict:
    """发一次流式请求，返回 usage dict 或 None。"""
    payload = {
        "messages": [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "Say hello."},
        ],
        "max_tokens": 10,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, stream=True)
    resp.raise_for_status()

    usage = None
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode()
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            chunk = json.loads(line[6:])
            if "usage" in chunk:
                usage = chunk["usage"]
    return usage


def check(url: str) -> bool:
    print(f"目标: {url}")
    print(f"system prompt 長さ: {len(LONG_SYSTEM)} chars")
    print("発送两次相同请求，验证第二次 cached_tokens > 0 ...")
    print()

    results = []
    for n in (1, 2):
        try:
            usage = send_one(url)
        except Exception as e:
            print(f"请求 {n}: ERROR — {e}", file=sys.stderr)
            return False

        if usage is None:
            print(f"请求 {n}: ERROR — 未收到 usage chunk", file=sys.stderr)
            return False

        pt = usage.get("prompt_tokens", "?")
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"请求 {n}: prompt_tokens={pt}, cached_tokens={cached}")
        results.append({"cached": cached})

    req2_cached = results[1]["cached"] if len(results) == 2 else 0
    ok = req2_cached > 0
    print()
    print(f"请求2 cached_tokens={req2_cached}  {'✅ 缓存命中' if ok else '❌ 期望 >0'}")
    print("说明: 请求1 cached_tokens 可能非零（服务级缓存跨请求持久），只要请求2 >0 即通过")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT, help="SIA 服务地址")
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

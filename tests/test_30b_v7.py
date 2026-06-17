"""
V7  Prefix Cache 命中（30B）

连发两次相同 prompt，第二次应命中缓存：
  请求 2: usage.prompt_tokens_details.cached_tokens > 0

30B 使用纯 attention 模型，block_size=16；prompt 需超过一个 block。

PASS: req2 cached_tokens > 0
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# ~200 tokens（远超 block_size=16，且在 30B max_model_len=9216 内）
SHARED_PREFIX = "You are a helpful assistant. " * 30


def send(url: str, msg: str) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": SHARED_PREFIX},
            {"role": "user", "content": msg},
        ],
        "max_tokens": 10,
        "stream": False,
    }
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def check(url: str) -> bool:
    try:
        d1 = send(url, "Say hi.")
        d2 = send(url, "Say hi.")
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    def cached(d: dict) -> int:
        return (d.get("usage", {})
                  .get("prompt_tokens_details", {})
                  .get("cached_tokens", 0))

    c1 = cached(d1)
    c2 = cached(d2)
    pt1 = d1.get("usage", {}).get("prompt_tokens", 0)
    pt2 = d2.get("usage", {}).get("prompt_tokens", 0)

    print(f"  req1: prompt_tokens={pt1}, cached_tokens={c1}")
    print(f"  req2: prompt_tokens={pt2}, cached_tokens={c2}  {'✅' if c2 > 0 else '❌ 期望 > 0'}")

    if c2 == 0:
        print("  ❌ 前缀缓存未生效（APC 未启用或 vLLM 版本不支持上报）", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

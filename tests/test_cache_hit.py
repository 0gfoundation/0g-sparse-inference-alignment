"""
验证 APC prefix cache 是否正确外报 cached_tokens。

原理：0GM-35B 是 hybrid 模型（full_attention + GatedDeltaNet），
需要 --mamba_cache_mode align 才能启用 APC。block_size ≈ 1056 tokens，
prompt >= 1056 tokens 的请求在第二次命中时 cached_tokens = 1056。

用法：
    python tests/test_cache_hit.py [--url http://localhost:8000]
"""
import argparse, json, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# 200 次重复 ≈ 1220 tokens（超过 1056 的 block_size 阈值）
LONG_SYSTEM = "You are a helpful assistant. " * 200


def send(url: str, n: int) -> None:
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

    if usage is None:
        print(f"请求 {n}: ERROR — 未收到 usage chunk", file=sys.stderr)
        return

    prompt_t = usage.get("prompt_tokens", "?")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    status = "✅" if (n == 1 and cached == 0) or (n == 2 and cached > 0) else "❌"
    print(f"请求 {n}: prompt_tokens={prompt_t}, cached_tokens={cached}  {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT, help="SIA 服务地址")
    args = parser.parse_args()

    print(f"目标: {args.url}")
    print(f"system prompt 长度: {len(LONG_SYSTEM)} chars")
    print("发送两次相同请求，验证第二次 cached_tokens > 0 ...")
    print()

    send(args.url, 1)
    send(args.url, 2)

    print()
    print("预期: 请求1 cached_tokens=0，请求2 cached_tokens=1056")


if __name__ == "__main__":
    main()

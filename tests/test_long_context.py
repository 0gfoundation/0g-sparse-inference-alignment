"""
验证服务是否真正支持长上下文（max_model_len=32768）。

原理：构造约 30000 tokens 的 prompt，发送请求并验证：
1. HTTP 200（未被截断拒绝）
2. prompt_tokens 接近预期值
3. 服务正常返回 completion（finish_reason != null）

用法：
    python tests/test_long_context.py [--url http://localhost:8000]
"""
import argparse, json, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# "The quick brown fox..." 约 10 tokens/句（含 chat template 开销）× 2800 句 ≈ 28000 tokens
# 留 ~4700 tokens 余量，确保 chat template 开销后不超过 32768 上限
LONG_CONTENT = "The quick brown fox jumps over the lazy dog. " * 2800


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT, help="SIA 服务地址")
    args = parser.parse_args()

    print(f"目标: {args.url}")
    print(f"prompt 长度: {len(LONG_CONTENT)} chars（约 30000 tokens）")
    print("发送长上下文请求...")
    print()

    payload = {
        "messages": [
            {"role": "user", "content": LONG_CONTENT + " Summarize in one sentence."},
        ],
        "max_tokens": 50,
        "stream": False,
    }

    try:
        resp = requests.post(f"{args.url}/v1/chat/completions", json=payload, timeout=300)
    except requests.exceptions.Timeout:
        print("❌ 请求超时（>300s）", file=sys.stderr)
        sys.exit(1)

    print(f"HTTP {resp.status_code}")

    if resp.status_code != 200:
        print(f"❌ 非 200 响应: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    usage = data.get("usage", {})
    choice = data.get("choices", [{}])[0]
    prompt_tokens = usage.get("prompt_tokens", 0)
    finish_reason = choice.get("finish_reason")
    content = choice.get("message", {}).get("content", "")

    ok_tokens = prompt_tokens > 20000
    ok_finish = finish_reason is not None

    print(f"prompt_tokens   : {prompt_tokens}  {'✅' if ok_tokens else '❌ 预期 >20000'}")
    print(f"finish_reason   : {finish_reason}  {'✅' if ok_finish else '❌'}")
    print(f"response content: {content[:120]}")
    print()

    if ok_tokens and ok_finish:
        print("✅ 长上下文支持验证通过（max_model_len=32768 生效）")
    else:
        print("❌ 验证失败", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

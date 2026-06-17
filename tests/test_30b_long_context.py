"""
长上下文验证（30B，max_model_len=16384）

构造约 10000 tokens 的 prompt（留余量给 output + template），
验证服务正常处理、prompt_tokens 符合预期。

PASS: HTTP 200, prompt_tokens > 8000, finish_reason 非 null
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# ~10 tokens/句 × 1000 句 ≈ 10000 tokens（含 chat template 约 50 tokens，总计约 10050，在 16384 内）
LONG_CONTENT = "The quick brown fox jumps over the lazy dog. " * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()

    print(f"目标: {args.url}")
    print(f"prompt 长度: {len(LONG_CONTENT)} chars（约 10000 tokens）")

    payload = {
        "messages": [{"role": "user", "content": LONG_CONTENT + " Summarize in one sentence."}],
        "max_tokens": 50,
        "stream": False,
    }
    try:
        resp = requests.post(f"{args.url}/v1/chat/completions", json=payload, timeout=120)
    except Exception as e:
        print(f"❌ 请求失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"❌ 非 200 响应: {resp.text[:200]}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    usage = data.get("usage", {})
    choice = data.get("choices", [{}])[0]
    pt = usage.get("prompt_tokens", 0)
    finish = choice.get("finish_reason")
    content = choice.get("message", {}).get("content", "")

    ok_tokens = pt > 8000
    ok_finish = finish is not None

    print(f"prompt_tokens : {pt}  {'✅' if ok_tokens else '❌ 期望 >8000'}")
    print(f"finish_reason : {finish}  {'✅' if ok_finish else '❌'}")
    print(f"response      : {content[:120]}")

    if ok_tokens and ok_finish:
        print("✅ 长上下文验证通过（max_model_len=16384 内正常响应）")
        sys.exit(0)
    else:
        print("❌ 验证失败", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

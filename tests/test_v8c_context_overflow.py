"""
V8c  context 超长 → 400

发送超过 max_model_len 的 prompt，服务应返回 400 而非 500。
默认构造 ~40000 tokens 的 prompt，适用于 max_model_len ≤ 35000 的模型。

PASS: HTTP 400, error.type == "invalid_request_error", message 说明 context length
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# 与 test_long_context.py 同款句子（已验证 2800 次 ≈ 28000 tokens）
# 4000 次 ≈ 40000 tokens，可溢出 max_model_len=32768
OVERFLOW_CONTENT = "The quick brown fox jumps over the lazy dog. " * 4000


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": OVERFLOW_CONTENT}],
        "max_tokens": 10,
    }
    print(f"  prompt 长度: {len(OVERFLOW_CONTENT)} chars（约 40000 tokens，超过 32768 上限）")
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code == 500:
        print("  ❌ 返回 500（服务内部崩溃，应返回 400）", file=sys.stderr)
        return False
    ok_status = resp.status_code == 400
    if not ok_status:
        print(f"  ❌ 期望 400，得到 {resp.status_code}", file=sys.stderr)
        return False

    data = resp.json()
    ok_format = "error" in data and "detail" not in data
    err_type = data.get("error", {}).get("type", "")
    err_msg = data.get("error", {}).get("message", "")
    ok_type = err_type == "invalid_request_error"
    ok_msg = "context" in err_msg.lower() or "length" in err_msg.lower() or "token" in err_msg.lower()

    print(f"  格式: {'OpenAI ✅' if ok_format else 'FastAPI detail ❌'}")
    print(f"  error.type   : {err_type!r}  {'✅' if ok_type else '❌'}")
    print(f"  error.message: {err_msg[:120]!r}  {'✅' if ok_msg else '❌ 期望含 context/length/token'}")
    return ok_format and ok_type and ok_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

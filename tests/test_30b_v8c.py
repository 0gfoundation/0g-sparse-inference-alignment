"""
V8c  context 超长（30B，max_model_len=2048）

发送超过 2048 tokens 的 prompt，服务应返回 400 而非 500。
"The quick brown fox..." * 300 ≈ 3000 tokens，安全超过 2048 上限。

PASS: HTTP 400, error.type == "invalid_request_error", message 含 context/length/token
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# ~3000 tokens，超过 30B max_model_len=2048
OVERFLOW_CONTENT = "The quick brown fox jumps over the lazy dog. " * 300


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": OVERFLOW_CONTENT}],
        "max_tokens": 10,
    }
    print(f"  prompt 长度: {len(OVERFLOW_CONTENT)} chars（约 3000 tokens，超过 2048 上限）")
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code == 500:
        print("  ❌ 返回 500（服务崩溃，应返回 400）", file=sys.stderr)
        return False
    if resp.status_code != 400:
        print(f"  ❌ 期望 400，得到 {resp.status_code}", file=sys.stderr)
        return False

    data = resp.json()
    ok_format = "error" in data and "detail" not in data
    err_type = data.get("error", {}).get("type", "")
    err_msg = data.get("error", {}).get("message", "")
    ok_type = err_type == "invalid_request_error"
    ok_msg = any(kw in err_msg.lower() for kw in ("context", "length", "token"))

    print(f"  格式: {'OpenAI ✅' if ok_format else 'FastAPI detail ❌'}")
    print(f"  error.type   : {err_type!r}  {'✅' if ok_type else '❌'}")
    print(f"  error.message: {err_msg[:120]!r}  {'✅' if ok_msg else '❌ 期望含 context/length/token'}")
    return ok_format and ok_type and ok_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

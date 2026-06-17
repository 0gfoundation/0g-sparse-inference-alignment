"""
V8b  空 messages 数组

messages=[] 应返回 400，而非 500。
PASS: HTTP 400, error.type == "invalid_request_error"
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "messages": [],
        "max_tokens": 10,
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    ok_status = resp.status_code == 400
    if resp.status_code == 500:
        print("  ❌ 返回 500（服务内部崩溃，应返回 400）", file=sys.stderr)
        return False
    if not ok_status:
        print(f"  ❌ 期望 400，得到 {resp.status_code}", file=sys.stderr)
        return False

    data = resp.json()
    ok_format = "error" in data and "detail" not in data
    err_type = data.get("error", {}).get("type", "")
    ok_type = err_type == "invalid_request_error"

    print(f"  格式: {'OpenAI ✅' if ok_format else 'FastAPI detail ❌'}")
    print(f"  error.type: {err_type!r}  {'✅' if ok_type else '❌'}")
    return ok_format and ok_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

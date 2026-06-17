"""
V5  tool call 拒绝

SIA 的 Value Model 无法评估 tool call 质量，服务主动拒绝。
PASS: HTTP 400, error.type == "invalid_request_error"
FAIL: 200（静默忽略）或 500
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "max_tokens": 256,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    ok_status = resp.status_code == 400
    if not ok_status:
        print(f"  ❌ 期望 400，得到 {resp.status_code}", file=sys.stderr)
        return False

    data = resp.json()
    err = data.get("error", {})
    err_type = err.get("type", "")
    err_msg = err.get("message", "")
    ok_type = err_type == "invalid_request_error"

    print(f"  error.type   : {err_type!r}  {'✅' if ok_type else '❌ 期望 invalid_request_error'}")
    print(f"  error.message: {err_msg[:100]!r}")
    return ok_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

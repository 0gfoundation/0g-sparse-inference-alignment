"""
V5  tool call 拒绝（30B）

含 tools 的请求应返回 400（服务端拒绝，RM 未经 tool call 对齐训练）。
PASS: HTTP 400, error.type == "invalid_request_error"
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 600,
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 400:
        print(f"  ❌ 期望 400，得到 {resp.status_code}", file=sys.stderr)
        if resp.status_code == 200:
            choices = resp.json().get("choices", [{}])
            tc = choices[0].get("message", {}).get("tool_calls")
            print(f"  tool_calls={tc}  （服务未拒绝，静默吐文本）", file=sys.stderr)
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
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

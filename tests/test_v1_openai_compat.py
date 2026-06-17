"""
V1  OpenAI 兼容接口

PASS: HTTP 200, choices[0].message.content 非空, finish_reason 合法
"""
import argparse, json, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "Reply with exactly one word: hello"}],
        "max_tokens": 20,
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  ❌ 期望 200，得到 {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return False

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        print("  ❌ choices 为空", file=sys.stderr)
        return False

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    finish_reason = choices[0].get("finish_reason")

    ok_content = bool(content)
    ok_finish = finish_reason in ("stop", "length")

    print(f"  content      : {content[:80]!r}  {'✅' if ok_content else '❌'}")
    print(f"  finish_reason: {finish_reason}  {'✅' if ok_finish else '❌'}")
    return ok_content and ok_finish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

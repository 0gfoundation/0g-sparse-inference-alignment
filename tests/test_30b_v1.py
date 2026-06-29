"""
V1  OpenAI 兼容接口（30B）

PASS: HTTP 200, choices[0].message.content 非空, finish_reason 合法（stop/length）
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
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
    ok_choices = len(choices) > 0
    content = choices[0].get("message", {}).get("content", "") if ok_choices else ""
    finish = choices[0].get("finish_reason") if ok_choices else None
    ok_content = bool(content)
    ok_finish = finish in ("stop", "length")

    print(f"  choices        : {len(choices)}  {'✅' if ok_choices else '❌'}")
    print(f"  content        : {content[:60]!r}  {'✅' if ok_content else '❌'}")
    print(f"  finish_reason  : {finish!r}  {'✅' if ok_finish else '❌ 期望 stop/length'}")
    return ok_choices and ok_content and ok_finish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

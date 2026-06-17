"""
V2  非流式 usage（30B）

PASS: usage.prompt_tokens > 0, completion_tokens > 0, total_tokens == sum
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 64,
        "stream": False,
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  ❌ {resp.text[:200]}", file=sys.stderr)
        return False

    usage = resp.json().get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0)
    ok_pt = pt > 0
    ok_ct = ct > 0
    ok_tt = tt == pt + ct

    print(f"  prompt_tokens    : {pt}  {'✅' if ok_pt else '❌'}")
    print(f"  completion_tokens: {ct}  {'✅' if ok_ct else '❌'}")
    print(f"  total_tokens     : {tt}  {'✅' if ok_tt else f'❌ 期望 {pt+ct}'}")
    return ok_pt and ok_ct and ok_tt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

"""
V2  非流式 usage（计费命脉）

PASS: usage.prompt_tokens > 0, usage.completion_tokens > 0, usage.total_tokens > 0
FAIL: usage 缺失或全为 0 → router 跳过计费，无法收费
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "Say: hi"}],
        "max_tokens": 32,
        "stream": False,
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
    usage = data.get("usage")
    if not usage:
        print("  ❌ usage 字段缺失", file=sys.stderr)
        return False

    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0)

    ok_pt = pt > 0
    ok_ct = ct > 0
    ok_tt = tt == pt + ct

    print(f"  prompt_tokens    : {pt}  {'✅' if ok_pt else '❌'}")
    print(f"  completion_tokens: {ct}  {'✅' if ok_ct else '❌'}")
    print(f"  total_tokens     : {tt}  {'✅' if ok_tt else '❌ 期望 prompt+completion'}")
    return ok_pt and ok_ct and ok_tt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

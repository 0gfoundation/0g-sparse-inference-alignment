"""
V3  流式结尾 usage（30B）

PASS: SSE 流末尾（[DONE] 之前）存在 usage chunk，
      且 usage.prompt_tokens > 0, usage.completion_tokens > 0
"""
import argparse, json, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload,
                             stream=True, timeout=120)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  ❌ 期望 200，得到 {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return False

    usage = None
    chunk_count = 0
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        chunk_count += 1
        u = chunk.get("usage")
        if u and u.get("prompt_tokens", 0) > 0:
            usage = u

    if chunk_count == 0:
        print("  ❌ 未收到任何 SSE chunk", file=sys.stderr)
        return False

    if not usage:
        print(f"  ❌ 收到 {chunk_count} 个 chunk，但无 usage chunk", file=sys.stderr)
        return False

    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    ok_pt = pt > 0
    ok_ct = ct > 0

    print(f"  SSE chunks       : {chunk_count}")
    print(f"  prompt_tokens    : {pt}  {'✅' if ok_pt else '❌'}")
    print(f"  completion_tokens: {ct}  {'✅' if ok_ct else '❌'}")
    return ok_pt and ok_ct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

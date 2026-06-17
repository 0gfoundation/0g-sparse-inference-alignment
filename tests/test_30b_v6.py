"""
V6  vision 多模态（30B）

发送一张橙色图（base64 PNG），模型应识别出 "orange"。
PASS: HTTP 200, content 含 "orange"（大小写不敏感）, usage 正常
"""
import argparse, base64, sys
import requests

URL_DEFAULT = "http://localhost:8000"

# 8×8 纯橙色 PNG（#FF8000），base64 编码
ORANGE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAAXNSR0IArs4c6QAAAB"
    "ZJREFUKFNj/M9w9D8DBTAxUMGuBAAp9QQJXvCE1QAAAABJRU5ErkJggg=="
)


def check(url: str) -> bool:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{ORANGE_PNG_B64}"
                        },
                    },
                    {"type": "text", "text": "What color is this? Reply in one word."},
                ],
            }
        ],
        "max_tokens": 20,
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  ❌ 期望 200，得到 {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return False

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    ok_color = "orange" in content.lower()
    ok_usage = usage.get("prompt_tokens", 0) > 0

    print(f"  content : {content!r}  {'✅' if ok_color else '❌ 期望含 orange'}")
    print(f"  prompt_tokens: {usage.get('prompt_tokens')}  {'✅' if ok_usage else '❌'}")
    return ok_color and ok_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    sys.exit(0 if check(args.url) else 1)


if __name__ == "__main__":
    main()

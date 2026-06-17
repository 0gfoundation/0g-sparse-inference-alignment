"""
V6  vision 多模态

发送一张纯橙色 PNG，问模型颜色。
PASS: HTTP 200, 响应内容包含 orange / red / warm 相关词
"""
import argparse, base64, struct, sys, zlib
import requests

URL_DEFAULT = "http://localhost:8000"


def make_orange_png(w: int = 32, h: int = 32) -> str:
    """生成纯橙色 PNG，返回 base64 字符串。"""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([255, 165, 0]) * w for _ in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


KEYWORDS = {"orange", "red", "warm", "amber", "color"}


def check(url: str) -> bool:
    b64 = make_orange_png()
    data_uri = f"data:image/png;base64,{b64}"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "What color is this image? Reply in one word."},
                ],
            }
        ],
        "max_tokens": 32,
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
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  response: {content!r}")

    ok = any(kw in content.lower() for kw in KEYWORDS)
    if not ok:
        print(f"  ❌ 响应未包含颜色关键词 {KEYWORDS}", file=sys.stderr)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

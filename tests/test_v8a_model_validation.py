"""
V8a  model 名称校验

传入不存在的 model 名，服务应返回 OpenAI 兼容的 404 错误。
PASS: HTTP 404, error.type == "invalid_request_error"
FAIL: 200（静默接受）或 {"detail": ...}（FastAPI 默认格式）
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    payload = {
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    ok_status = resp.status_code == 404
    if not ok_status:
        print(f"  ❌ 期望 404，得到 {resp.status_code}", file=sys.stderr)
        if resp.status_code == 200:
            print("  ❌ 服务未校验 model 字段，静默使用已加载模型", file=sys.stderr)
        return False

    data = resp.json()
    # 检查是 OpenAI 格式 {"error": {...}} 而非 FastAPI 的 {"detail": ...}
    ok_format = "error" in data and "detail" not in data
    err_type = data.get("error", {}).get("type", "")
    ok_type = err_type == "invalid_request_error"

    print(f"  格式: {'OpenAI ✅' if ok_format else 'FastAPI detail ❌'}")
    print(f"  error.type: {err_type!r}  {'✅' if ok_type else '❌ 期望 invalid_request_error'}")
    return ok_format and ok_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
/v1/models 字段验证

PASS:
  - owned_by == "0G Foundation"（router 注册硬要求）
  - id 为 basename（无路径分隔符 "/"）
"""
import argparse, sys
import requests

URL_DEFAULT = "http://localhost:8000"


def check(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/v1/models", timeout=10)
    except Exception as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  ❌ 期望 200，得到 {resp.status_code}", file=sys.stderr)
        return False

    data = resp.json()
    models = data.get("data", [])
    if not models:
        print("  ❌ data 列表为空", file=sys.stderr)
        return False

    model = models[0]
    model_id = model.get("id", "")
    owned_by = model.get("owned_by", "")

    ok_owned = owned_by == "0G Foundation"
    ok_basename = "/" not in model_id

    print(f"  id       : {model_id!r}  {'✅ basename' if ok_basename else '❌ 含路径分隔符，应为 basename'}")
    print(f"  owned_by : {owned_by!r}  {'✅' if ok_owned else '❌ 期望 0G Foundation'}")
    return ok_owned and ok_basename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()
    ok = check(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

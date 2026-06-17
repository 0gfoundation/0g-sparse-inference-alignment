#!/bin/bash
# VL-30B SIA 服务集成测试一键脚本
#
# 用法：
#   bash tests/run_all_30b.sh                          # 默认 http://localhost:8000
#   bash tests/run_all_30b.sh http://localhost:8000
#   bash tests/run_all_30b.sh http://host:8000 --skip-vision

URL="${1:-http://localhost:8000}"
SKIP_VISION=0
for arg in "$@"; do
    [ "$arg" = "--skip-vision" ] && SKIP_VISION=1
done

PASS=0
FAIL=0
SKIP=0
TMPFILE="/tmp/_sia_30b_test_out.txt"

run() {
    local label="$1"; shift
    printf "  %-38s" "$label"
    if python "$@" --url "$URL" >"$TMPFILE" 2>&1; then
        echo "✅ PASS"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL"
        grep -E "❌|Error|error|期望|HTTP" "$TMPFILE" | head -3 | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    local label="$1"
    printf "  %-38s" "$label"
    echo "⏭  SKIP"
    SKIP=$((SKIP + 1))
}

echo "════════════════════════════════════════════"
echo "  VL-30B SIA 服务集成测试"
echo "  目标: $URL"
echo "════════════════════════════════════════════"

run "V1  OpenAI兼容接口"            tests/test_30b_v1.py
run "V2  非流式 usage（计费命脉）"  tests/test_30b_v2.py
run "V3  流式结尾 usage（计费命脉）" tests/test_30b_v3.py
if [ "$SKIP_VISION" -eq 1 ]; then
    skip "V6  vision 多模态"
else
    run "V6  vision 多模态"          tests/test_30b_v6.py
fi
run "V7  cache 命中字段"            tests/test_30b_v7.py
run "V5  tool call 拒绝 → 400"      tests/test_30b_v5.py
run "V8a model 名称校验 → 404"      tests/test_30b_v8a.py
run "V8b 空 messages → 400"         tests/test_30b_v8b.py
run "V8c context 超长 → 400"        tests/test_30b_v8c.py
run "    /v1/models 字段"           tests/test_30b_model_id.py
run "    长上下文（max_model_len）"  tests/test_30b_long_context.py

TOTAL=$((PASS + FAIL + SKIP))
echo "════════════════════════════════════════════"
printf "  PASS=%-3d FAIL=%-3d SKIP=%-3d TOTAL=%d\n" $PASS $FAIL $SKIP $TOTAL
if [ $FAIL -eq 0 ]; then
    echo "  ✅ 全部通过"
else
    echo "  ❌ 有 $FAIL 项失败"
fi
echo "════════════════════════════════════════════"

exit $FAIL

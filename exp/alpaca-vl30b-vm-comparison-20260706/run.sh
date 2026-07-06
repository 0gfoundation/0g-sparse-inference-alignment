#!/usr/bin/env bash
# Controlled VM comparison: scalar head vs vocab_lowrank BT
#
# Both VMs generate 200 AlpacaEval responses (identical LLM + SIA params).
# Skywork scores all outputs. A shared nosia baseline is also generated.
#
# Usage:
#   cd /workspace/sia-repo/0g-sparse-inference-alignment
#   bash exp/alpaca-vl30b-vm-comparison-20260706/run.sh VENV_PATH
#
#   VENV_PATH: path to vl30b-fast venv (create with scripts/setup_venv_vl30b_fast.sh if absent)
#
# Steps run (idempotent — skips steps if output file already exists):
#   1. nosia baseline  (--sia_weight 0)
#   2. SIA with scalar head VM
#   3. SIA with vocab_lowrank BT VM
#   4. Skywork scoring of all three
#   5. Summary comparison table

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
EXPDIR="$REPO/exp/alpaca-vl30b-vm-comparison-20260706"
VENV="${1:?Usage: $0 VENV_PATH}"
PYTHON="$VENV/bin/python"

# ── Model paths ──────────────────────────────────────────────────────────────
LLM=/workspace/models/Qwen3-VL-30B-A3B-Instruct
VM_SCALAR=/workspace/models/VM-Qwen3-4B-merged-for-vllm
VM_VOCAB=/workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-bt-20260704-merged
SKYWORK=/workspace/models/Skywork-Reward-V2-Llama-3.1-8B

# ── Generation params (identical for all three runs) ─────────────────────────
N_QUESTIONS=200
MAX_TOKENS=2048
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=20
REP_PENALTY=1.0

# ── SIA params ────────────────────────────────────────────────────────────────
SIA_TOPK=10
SIA_WEIGHT=1.0
SIA_ENTROPY_THR=1.0
SIA_B2_GPU_MEM=0.08   # matches docker-compose.yml

PORT=8000

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Start SIA server, wait for /health ───────────────────────────────────────
start_server() {
    local vm_model="$1"
    local label="$2"
    log "Starting server: $label  (rm_model=$(basename $vm_model))"

    SIA_RM_CUDAGRAPH=piecewise \
    SIA_RM_MULTIPROCESS=0 \
    SIA_DEBUG_HIST=1 \
    "$PYTHON" "$REPO/src/sia_vllm_server.py" \
        --llm        "$LLM" \
        --model_id   "VL30B-SIA-$label" \
        --rm_backend b2 \
        --rm_model   "$vm_model" \
        --rm_b2_gpu_mem  "$SIA_B2_GPU_MEM" \
        --topk "$SIA_TOPK" --weight "$SIA_WEIGHT" \
        --entropy_threshold "$SIA_ENTROPY_THR" \
        --max_model_len "$MAX_TOKENS" \
        --enable_prefix_caching \
        --host 0.0.0.0 \
        --port "$PORT" \
        > "$EXPDIR/server_${label}.log" 2>&1 &
    SERVER_PID=$!
    log "  Server PID=$SERVER_PID — waiting for /health ..."

    local waited=0
    while [ $waited -lt 900 ]; do
        sleep 5; waited=$((waited+5))
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            log "  Server ready after ${waited}s"
            return 0
        fi
    done
    log "ERROR: server did not come up in 900s. Check $EXPDIR/server_${label}.log"
    kill "$SERVER_PID" 2>/dev/null || true
    return 1
}

stop_server() {
    log "Stopping server PID=$SERVER_PID ..."
    kill "$SERVER_PID" 2>/dev/null || true
    local waited=0
    while [ $waited -lt 60 ]; do
        sleep 2; waited=$((waited+2))
        kill -0 "$SERVER_PID" 2>/dev/null || { log "  Server stopped"; return 0; }
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    sleep 3
    log "  Server force-killed"
}

# ── Generate 200 responses ────────────────────────────────────────────────────
run_gen() {
    local output="$1"
    local label="$2"
    local sia_weight_override="${3:-}"   # empty = use server default (SIA on); "0.0" = nosia

    local extra_args=""
    [ -n "$sia_weight_override" ] && extra_args="--sia_weight $sia_weight_override"

    log "Generating: $label → $(basename $output)"
    (
        cd "$REPO"
        "$PYTHON" eval/alpaca_eval.py \
            --base_url "http://localhost:$PORT/v1" \
            --model    "VL30B-SIA-$label" \
            --dataset  data/alpaca_eval/alpaca_eval.json \
            --output   "$output" \
            --limit    "$N_QUESTIONS" \
            --max_tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --top_k "$TOP_K" \
            --repetition_penalty "$REP_PENALTY" \
            $extra_args \
            2>&1 | tee "$EXPDIR/gen_${label}.log"
    )
}

# ── Score with Skywork ────────────────────────────────────────────────────────
run_score() {
    local input="$1"
    local output="$2"
    local label="$3"
    log "Scoring: $label with Skywork"
    (
        cd "$REPO"
        "$PYTHON" scripts/measure_alpaca_reward.py \
            --input_file  "$input" \
            --output_file "$output" \
            --rm "$SKYWORK" \
            --device cuda:0 \
            2>&1 | tee "$EXPDIR/score_${label}.log"
    )
}

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — nosia baseline (server with scalar head, SIA weight=0)
# ═══════════════════════════════════════════════════════════════════════
if [ ! -f "$EXPDIR/nosia.json" ]; then
    start_server "$VM_SCALAR" "nosia"
    run_gen "$EXPDIR/nosia.json" "nosia" "0.0"
    stop_server
else
    log "nosia.json exists — skipping"
fi

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — SIA with scalar head VM
# ═══════════════════════════════════════════════════════════════════════
if [ ! -f "$EXPDIR/scalar_sia.json" ]; then
    start_server "$VM_SCALAR" "scalar"
    run_gen "$EXPDIR/scalar_sia.json" "scalar"
    stop_server
else
    log "scalar_sia.json exists — skipping"
fi

# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — SIA with vocab_lowrank BT VM
# ═══════════════════════════════════════════════════════════════════════
if [ ! -f "$EXPDIR/vocab_sia.json" ]; then
    start_server "$VM_VOCAB" "vocab"
    run_gen "$EXPDIR/vocab_sia.json" "vocab"
    stop_server
else
    log "vocab_sia.json exists — skipping"
fi

# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Skywork scoring (runs on single GPU without SIA server)
# ═══════════════════════════════════════════════════════════════════════
[ -f "$EXPDIR/nosia_scored.json"      ] || run_score "$EXPDIR/nosia.json"      "$EXPDIR/nosia_scored.json"      "nosia"
[ -f "$EXPDIR/scalar_sia_scored.json" ] || run_score "$EXPDIR/scalar_sia.json" "$EXPDIR/scalar_sia_scored.json" "scalar"
[ -f "$EXPDIR/vocab_sia_scored.json"  ] || run_score "$EXPDIR/vocab_sia.json"  "$EXPDIR/vocab_sia_scored.json"  "vocab"

# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — Summary
# ═══════════════════════════════════════════════════════════════════════
log "=== FINAL RESULTS ==="
"$PYTHON" - <<'PYEOF'
import json
from pathlib import Path

expdir = Path("exp/alpaca-vl30b-vm-comparison-20260706")

def stats(name):
    p = expdir / f"{name}_scored.json"
    if not p.exists():
        return None, 0
    d = json.load(open(p))
    r = [x["reward"] for x in d if x.get("reward") is not None]
    return sum(r)/len(r), len(r)

nosia_mean,  nosia_n  = stats("nosia")
scalar_mean, scalar_n = stats("scalar_sia")
vocab_mean,  vocab_n  = stats("vocab_sia")

def pct(a, b):
    return f"{(a-b)/b*100:+.2f}%" if b else "n/a"

print(f"\n{'='*60}")
print(f"  {'Run':<28} {'mean reward':>12}  {'n':>5}  {'vs nosia':>10}")
print(f"  {'-'*56}")
print(f"  {'nosia baseline':<28} {nosia_mean:>12.4f}  {nosia_n:>5}")
print(f"  {'scalar head SIA':<28} {scalar_mean:>12.4f}  {scalar_n:>5}  {pct(scalar_mean, nosia_mean):>10}")
print(f"  {'vocab_lowrank BT SIA':<28} {vocab_mean:>12.4f}  {vocab_n:>5}  {pct(vocab_mean, nosia_mean):>10}")
print(f"  {'delta (vocab - scalar)':<28} {vocab_mean-scalar_mean:>+12.4f}")
print(f"{'='*60}\n")
PYEOF

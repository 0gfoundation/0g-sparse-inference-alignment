#!/usr/bin/env bash
# Backbone-only generation test — no vLLM, no SIA, no head intervention.
#
# Uses transformers directly to load each VM as Qwen3ForCausalLM and generates
# 200 AlpacaEval responses. Avoids all vLLM --hf-overrides / EngineCore issues.
#
# Runs both VMs sequentially (not concurrently) so each gets the full GPU.
# Output:
#   /workspace/exp/alpaca-vm-backbone-YYYYMMDD/scalar_backbone.json
#   /workspace/exp/alpaca-vm-backbone-YYYYMMDD/vocab_backbone.json

set -euo pipefail

REPO=/workspace/sia-repo/0g-sparse-inference-alignment
VENV_PYTHON=/opt/venv-vl30b/bin/python

VM_SCALAR=/workspace/models/VM-Qwen3-4B-merged-for-vllm
VM_VOCAB=/workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-frozen-20260706-merged
DATA=/workspace/sia-repo/0g-sparse-inference-alignment/data/alpaca_eval/alpaca_eval.json

TAG=$(date +%Y%m%d)
OUT=/workspace/exp/alpaca-vm-backbone-$TAG

echo "[entrypoint] backbone generation: $OUT"
echo "[entrypoint] scalar VM: $VM_SCALAR"
echo "[entrypoint] vocab  VM: $VM_VOCAB"

# ── Scalar VM ────────────────────────────────────────────────────────────────
echo ""
echo "[entrypoint] === Step 1/2: scalar VM backbone ==="
"$VENV_PYTHON" "$REPO/scripts/vm_backbone_gen.py" \
  --vm    "$VM_SCALAR" \
  --label scalar \
  --data  "$DATA" \
  --output "$OUT/scalar_backbone.json" \
  --limit 200 \
  --max_tokens 512 \
  --temperature 1.0 \
  --device cuda:0

# ── Vocab VM ─────────────────────────────────────────────────────────────────
echo ""
echo "[entrypoint] === Step 2/2: vocab_lowrank VM backbone ==="
"$VENV_PYTHON" "$REPO/scripts/vm_backbone_gen.py" \
  --vm    "$VM_VOCAB" \
  --label vocab \
  --data  "$DATA" \
  --output "$OUT/vocab_backbone.json" \
  --limit 200 \
  --max_tokens 512 \
  --temperature 1.0 \
  --device cuda:0

echo ""
echo "[entrypoint] 两个 backbone 生成完成。结果保存到 $OUT"
echo "[entrypoint] 下一步：用 Skywork 打分"
echo ""
echo "  $VENV_PYTHON $REPO/scripts/measure_alpaca_reward.py \\"
echo "    --input_file $OUT/scalar_backbone.json \\"
echo "    --output_file $OUT/scalar_backbone_scored.json \\"
echo "    --rm /workspace/models/Skywork-Reward-V2-Llama-3.1-8B --device cuda:0"
echo ""
echo "  $VENV_PYTHON $REPO/scripts/measure_alpaca_reward.py \\"
echo "    --input_file $OUT/vocab_backbone.json \\"
echo "    --output_file $OUT/vocab_backbone_scored.json \\"
echo "    --rm /workspace/models/Skywork-Reward-V2-Llama-3.1-8B --device cuda:0"

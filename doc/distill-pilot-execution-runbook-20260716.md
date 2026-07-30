# Distillation Pilot — Execution Runbook (VL-30B, Fix 1)

**Date:** 2026-07-16
**Scope:** Step-by-step commands to run the Fix 1 (scalar-head → vocab_lowrank distillation) pilot end to end, from data collection through the go/no-go decision gate. See `doc/vocab-lowrank-root-cause-and-fix-plan-20260714.md` for the root-cause analysis and design rationale this pilot implements, and the approved plan for the code changes (`src/sia_rm/client.py`'s `SIA_DISTILL_LOG` hook, `SIA/src/value_model/model.py`'s `vocab_rewards_last_only` flag, `SIA/src/value_model/train.py`'s `--loss_type distill` mode).

`docker-compose.scalar-head-20260710.yml` already has the `SIA_DISTILL_LOG` env var wired in — no manual YAML edit needed.

---

## Step 1 — Restart the scalar-head VL-30B server (with distill logging enabled)

```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
mkdir -p exp
docker compose -f docker-compose.scalar-head-20260710.yml down
docker compose -f docker-compose.scalar-head-20260710.yml up -d
until curl -sf http://localhost:8000/health >/dev/null; do sleep 15; done
echo "server up"
```

## Step 2 — Generate 150 prompts (distill data is logged as a side effect of normal generation)

```bash
docker compose -f docker-compose.scalar-head-20260710.yml exec sia-vl30b bash -c '
cd /workspace/sia-repo/0g-sparse-inference-alignment
python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --limit 150 --max_tokens 2048 --temperature 1.0 \
  --output exp/distill_pilot_vl30b_gen.json \
  > exp/distill_pilot_vl30b_gen.log 2>&1
'
```

~10-20 minutes for 150 prompts.

## Step 3 — Spot-check the logged JSONL before trusting it

```bash
wc -l exp/distill_pilot_vl30b.jsonl
head -3 exp/distill_pilot_vl30b.jsonl
python3 -c "
import json
n=0; sids=set()
for line in open('exp/distill_pilot_vl30b.jsonl'):
    r = json.loads(line)
    n += 1
    sids.add(r['sid'])
    assert len(r['cand_ids']) == len(r['scores']), r
print(f'{n} records, {len(sids)} distinct sessions, cand_ids/scores length all match')
"
```

Expect a few hundred to a few thousand records (depends on the intervention rate) and distinct sessions close to 150 (roughly one session per generation). If the file is empty, check the env var actually reached the container:
```bash
docker compose -f docker-compose.scalar-head-20260710.yml exec sia-vl30b env | grep SIA_DISTILL_LOG
```

## Step 4 — Stop the server (free GPU), train

```bash
docker compose -f docker-compose.scalar-head-20260710.yml down

docker compose -f docker-compose.scalar-head-20260710.yml run --rm sia-vl30b bash -c '
cd /workspace/sia-repo/SIA
python src/value_model/train.py \
  --loss_type distill \
  --data_file /workspace/sia-repo/0g-sparse-inference-alignment/exp/distill_pilot_vl30b.jsonl \
  --base_model_path /workspace/models/VM-Qwen3-4B-merged-for-vllm \
  --output_dir models/VM-Qwen3-4B-vocab-lowrank-distill-pilot-20260716 \
  --head_type vocab_lowrank --head_rank 64 \
  --freeze_backbone \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
  --batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 --num_epochs 10 --max_length 2048 \
  --save_model
' 2>&1 | tee exp/distill_pilot_train.log
```

Runs inside the existing `sia-vl30b` image (already has `peft`/`sklearn`/`matplotlib` etc.) rather than assuming a bare-host Python environment.

## Step 5 — Read off the validation metric (no separate script needed)

Printed automatically every epoch during training:
```
验证 - Loss(KL): X.XXXX, Argmax_match: 0.XXX
```

```bash
grep "Argmax_match" exp/distill_pilot_train.log
cat models/VM-Qwen3-4B-vocab-lowrank-distill-pilot-20260716/training_history.json | python3 -m json.tool | grep -A5 '"val"'
```

`argmax_match` = how often the student's top-1 pick among the same K candidates matches the teacher's (scalar head's) top-1. Baselines to compare against: current vocab_lowrank is stuck at 42-44% (measured `top1_flip`), scalar head is ~66%.

## Step 6 — Decision gate

- **`argmax_match` climbs clearly above the 42-44% baseline** → proceed to deployment validation:
  ```bash
  python scripts/convert_rm_for_vllm.py \
    --rm /workspace/models/Qwen3-4B-Base \
    --rm_lora /workspace/sia-repo/SIA/models/VM-Qwen3-4B-vocab-lowrank-distill-pilot-20260716 \
    --output /workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-distill-pilot-merged
  ```
  Then deploy on a new docker-compose config (adapted from `docker-compose.sia-backbone-frozen-20260706.yml`, pointing `--rm_model` at the new merged checkpoint), smoke-test live `top1_flip` on a short run (should move toward the 50-80% healthy range), then a full 805Q AlpacaEval confirmation run before considering scaling up data volume or extending to 0GM-35B.
- **No meaningful improvement** → do not scale up data collection yet; revisit data volume, temperature, or the training recipe first.

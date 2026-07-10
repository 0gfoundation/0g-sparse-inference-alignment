# VL-30B AlpacaEval 805Q — SIA new head vs old head vs noSIA

**Date:** 2026-07-10
**LLM:** Qwen3-VL-30B-A3B-Instruct (b2 inproc, vLLM 0.17.1)
**Eval:** AlpacaEval **full 805Q** (previous experiments in this series only ran 200Q)
**Scoring RM:** Skywork-Reward-V2-Llama-3.1-8B (third-party, independent of the VM/SIA reward models)
**Sampling:** `temperature=1.0`, `max_tokens=2048`, no `top_p`/`top_k`/`repetition_penalty` override (unified across all 3 arms so the VM is the only variable)

---

## Background

Prior 200Q AlpacaEval runs on VL-30B showed two SIA Value Model heads roughly tied:

| Experiment | VM | n | Δ vs noSIA |
|---|---|---|---|
| Scalar head (2026-06-10) | `VM-Qwen3-4B-merged-for-vllm` (official) | 200 | +4.69% |
| vocab_lowrank sia-backbone frozen (2026-07-07) | `VM-Qwen3-4B-vocab-lowrank-sia-backbone-frozen-20260706-merged` | 200 | +4.82% |

This experiment reruns both heads on the **full 805-question set**, plus a noSIA baseline generated independently in *each* of the two docker deployments (as a cross-check that the noSIA baseline doesn't depend on which VM happens to be loaded, since `sia_weight=0` disables RM scoring entirely).

Earlier the same day, an unrelated 200Q run had accidentally used a freshly re-exported copy of the new-head VM (`VM-Qwen3-4B-vocab-lowrank-sia-backbone-frozen-vllm`) instead of the one that produced the +4.82% result. Before this experiment, `docker-compose.sia-backbone-frozen-20260706.yml` was reverted to point back at the original `...-sia-backbone-frozen-20260706-merged` checkpoint. A SHA256 check of all files in both directories (`chat_template.jinja`, `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`) confirmed the two paths are byte-identical anyway, so this was a naming/provenance issue, not a content difference.

---

## Server configs

### New head — `docker-compose.sia-backbone-frozen-20260706.yml`

```
--llm        /workspace/models/Qwen3-VL-30B-A3B-Instruct
--model_id   Qwen3-VL-30B-A3B-Instruct-SIA
--rm_backend b2
--rm_model   /workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-sia-backbone-frozen-20260706-merged
--vm_head_type vocab_lowrank --vm_head_rank 64
--llm_gpu_mem    0.50
--rm_b2_gpu_mem  0.08
--topk 10 --weight 1.0 --entropy_threshold 1.0
--max_model_len 2048
--enable_prefix_caching
```

### Old head — `docker-compose.scalar-head-20260710.yml` (new file, added this experiment)

Identical to the above except the RM and no head-type flags (scalar head is the default):

```
--llm        /workspace/models/Qwen3-VL-30B-A3B-Instruct
--model_id   Qwen3-VL-30B-A3B-Instruct-SIA
--rm_backend b2
--rm_model   /workspace/models/VM-Qwen3-4B-merged-for-vllm
--llm_gpu_mem    0.50
--rm_b2_gpu_mem  0.08
--topk 10 --weight 1.0 --entropy_threshold 1.0
--max_model_len 2048
--enable_prefix_caching
```

---

## Commands run

### 1. New head — docker up + generation

```bash
docker compose -f docker-compose.sia-backbone-frozen-20260706.yml down
docker compose -f docker-compose.sia-backbone-frozen-20260706.yml up -d --build
until curl -sf http://localhost:8000/health >/dev/null; do sleep 15; done

docker compose -f docker-compose.sia-backbone-frozen-20260706.yml exec sia-vl30b bash -c '
cd /workspace/sia-repo/0g-sparse-inference-alignment
python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --max_tokens 2048 --temperature 1.0 \
  --output exp/alpaca-vl30b-3way-805q-20260710/sia_newhead.json \
  > exp/alpaca-vl30b-3way-805q-20260710/sia_newhead_gen.log 2>&1
'
```

**noSIA (same new-head container, `--sia_weight 0`):**

```bash
docker compose -f docker-compose.sia-backbone-frozen-20260706.yml exec sia-vl30b bash -c '
cd /workspace/sia-repo/0g-sparse-inference-alignment
python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --max_tokens 2048 --temperature 1.0 --sia_weight 0 \
  --output exp/alpaca-vl30b-3way-805q-20260710/nosia_newheaddocker.json \
  > exp/alpaca-vl30b-3way-805q-20260710/nosia_gen_newheaddocker.log 2>&1
'
```

### 2. Old head — docker restart + generation

```bash
docker compose -f docker-compose.sia-backbone-frozen-20260706.yml down
docker compose -f docker-compose.scalar-head-20260710.yml up -d --build
until curl -sf http://localhost:8000/health >/dev/null; do sleep 15; done

docker compose -f docker-compose.scalar-head-20260710.yml exec sia-vl30b bash -c '
cd /workspace/sia-repo/0g-sparse-inference-alignment
python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --max_tokens 2048 --temperature 1.0 \
  --output exp/alpaca-vl30b-3way-805q-20260710/sia_oldhead.json \
  > exp/alpaca-vl30b-3way-805q-20260710/sia_oldhead_gen.log 2>&1
'
```

**noSIA (same old-head container, `--sia_weight 0`, no restart):**

```bash
docker compose -f docker-compose.scalar-head-20260710.yml exec sia-vl30b bash -c '
cd /workspace/sia-repo/0g-sparse-inference-alignment
python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --max_tokens 2048 --temperature 1.0 --sia_weight 0 \
  --output exp/alpaca-vl30b-3way-805q-20260710/nosia_oldhead.json \
  > exp/alpaca-vl30b-3way-805q-20260710/nosia_gen_oldhead.log 2>&1
'
```

### 3. Scoring (all 4 generation files, run strictly serially on one GPU)

Server stopped first to free GPU memory for the Skywork RM. All 4 scoring calls were chained in a single shell script with no backgrounding between them (each only starts once the previous process has exited) — this was an explicit requirement, not a convenience choice, to avoid two 8B-model forward passes contending for the same GPU.

```bash
RM=/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B
IN=exp/alpaca-vl30b-3way-805q-20260710
OUT=exp/alpaca-vl30b-3way-805q-20260710-scored

python scripts/measure_alpaca_reward.py \
  --input_file  $IN/sia_newhead.json \
  --output_file $OUT/sia_newhead_scored.json \
  --rm $RM --device cuda:0 > $OUT/sia_newhead_score.log 2>&1

python scripts/measure_alpaca_reward.py \
  --input_file  $IN/nosia_newheaddocker.json \
  --output_file $OUT/nosia_newheaddocker_scored.json \
  --rm $RM --device cuda:0 > $OUT/nosia_newheaddocker_score.log 2>&1

python scripts/measure_alpaca_reward.py \
  --input_file  $IN/sia_oldhead.json \
  --output_file $OUT/sia_oldhead_scored.json \
  --rm $RM --device cuda:0 > $OUT/sia_oldhead_score.log 2>&1

python scripts/measure_alpaca_reward.py \
  --input_file  $IN/nosia_oldhead.json \
  --output_file $OUT/nosia_oldhead_scored.json \
  --rm $RM --device cuda:0 > $OUT/nosia_oldhead_score.log 2>&1
```

---

## Results

| Arm | n (scored) | mean reward | p50 | min | max | Δ vs its own noSIA | Δ% |
|---|---|---|---|---|---|---|---|
| SIA — new head (vocab_lowrank) | 791/805 | **29.0972** | 28.75 | −2.1875 | 69.00 | +1.2669 | **+4.55%** |
| noSIA (generated in new-head container) | 799/805 | 27.8303 | 27.375 | −5.75 | 63.75 | — | — |
| SIA — old head (scalar) | 803/805 | **30.2142** | 29.75 | −1.6953 | 65.00 | +2.4258 | **+8.73%** |
| noSIA (generated in old-head container) | 797/805 | 27.7884 | 26.75 | −7.3438 | 66.00 | — | — |

**noSIA cross-check:** 27.8303 vs 27.7884 — diff 0.0419 (0.15% relative). The two independently-generated noSIA baselines agree almost exactly, confirming the noSIA arm is invariant to which VM is loaded in the container (expected: `sia_weight=0` disables RM scoring entirely) and that the two docker deployments don't introduce a systematic bias. This makes the 3-way comparison trustworthy.

---

## Analysis

**At full 805Q, the old (scalar) head clearly outperforms the new (vocab_lowrank) head** — +8.73% vs +4.55%, roughly 2× the uplift. This reverses the earlier 200Q impression that the two heads were roughly tied (+4.69% vs +4.82%).

Two candidate explanations, not mutually exclusive:

1. **200Q was underpowered.** With only 200 samples the earlier "tied" result may simply have been noise; 805Q gives a more reliable estimate of the true effect size for both heads.
2. **Sampling params changed for the old head.** The 2026-06-10 old-head reference run used `top_p=0.95 top_k=20 repetition_penalty=1.0`; this experiment intentionally unified sampling across all 3 arms (temperature=1.0 only, no top_p/top_k/rep_penalty) so the VM would be the sole variable. It's possible the old head's advantage is sensitive to this sampling change in a way the new head isn't.

**Efficiency vs. quality tradeoff confirmed.** The vocab_lowrank new head does 1 VM forward per decoding step regardless of `topk`, vs K=10 for the scalar head — a real throughput win — but at 805Q scale it now shows a visible quality cost (roughly half the alignment uplift of the scalar head) that wasn't apparent at 200Q.

## Conclusions

1. The scalar head remains the stronger choice for alignment quality on VL-30B AlpacaEval; the vocab_lowrank head's efficiency advantage comes with a real quality cost that only became visible at full 805Q scale.
2. Before drawing a final conclusion on head architecture, worth isolating the sampling-param change as a variable — e.g. rerun the old head with its original `top_p=0.95/top_k=20/rep_penalty=1.0` sampling to see whether its +8.73% shrinks back toward +4.69%, which would point to sampling sensitivity rather than a true head-architecture gap.
3. It would also help to look at per-question win/loss patterns for the new head's biggest losses vs the scalar head, to see if there's a common failure mode (similar to the "early termination on verbose-preferring RM" mechanism found in the earlier 0GM-35B / VL-30B cross-family analysis).

---

## Data

- Generation output + logs: [`exp/alpaca-vl30b-3way-805q-20260710/`](../exp/alpaca-vl30b-3way-805q-20260710/)
  - [`sia_newhead.json`](../exp/alpaca-vl30b-3way-805q-20260710/sia_newhead.json) / [`sia_newhead_gen.log`](../exp/alpaca-vl30b-3way-805q-20260710/sia_newhead_gen.log)
  - [`nosia_newheaddocker.json`](../exp/alpaca-vl30b-3way-805q-20260710/nosia_newheaddocker.json) / [`nosia_gen_newheaddocker.log`](../exp/alpaca-vl30b-3way-805q-20260710/nosia_gen_newheaddocker.log)
  - [`sia_oldhead.json`](../exp/alpaca-vl30b-3way-805q-20260710/sia_oldhead.json) / [`sia_oldhead_gen.log`](../exp/alpaca-vl30b-3way-805q-20260710/sia_oldhead_gen.log)
  - [`nosia_oldhead.json`](../exp/alpaca-vl30b-3way-805q-20260710/nosia_oldhead.json) / [`nosia_gen_oldhead.log`](../exp/alpaca-vl30b-3way-805q-20260710/nosia_gen_oldhead.log)
- Scored output + logs: [`exp/alpaca-vl30b-3way-805q-20260710-scored/`](../exp/alpaca-vl30b-3way-805q-20260710-scored/)
  - [`sia_newhead_scored.json`](../exp/alpaca-vl30b-3way-805q-20260710-scored/sia_newhead_scored.json) / [`sia_newhead_score.log`](../exp/alpaca-vl30b-3way-805q-20260710-scored/sia_newhead_score.log)
  - [`nosia_newheaddocker_scored.json`](../exp/alpaca-vl30b-3way-805q-20260710-scored/nosia_newheaddocker_scored.json) / [`nosia_newheaddocker_score.log`](../exp/alpaca-vl30b-3way-805q-20260710-scored/nosia_newheaddocker_score.log)
  - [`sia_oldhead_scored.json`](../exp/alpaca-vl30b-3way-805q-20260710-scored/sia_oldhead_scored.json) / [`sia_oldhead_score.log`](../exp/alpaca-vl30b-3way-805q-20260710-scored/sia_oldhead_score.log)
  - [`nosia_oldhead_scored.json`](../exp/alpaca-vl30b-3way-805q-20260710-scored/nosia_oldhead_scored.json) / [`nosia_oldhead_score.log`](../exp/alpaca-vl30b-3way-805q-20260710-scored/nosia_oldhead_score.log)
- Server config: [`docker-compose.sia-backbone-frozen-20260706.yml`](../docker-compose.sia-backbone-frozen-20260706.yml) (new head), [`docker-compose.scalar-head-20260710.yml`](../docker-compose.scalar-head-20260710.yml) (old head)

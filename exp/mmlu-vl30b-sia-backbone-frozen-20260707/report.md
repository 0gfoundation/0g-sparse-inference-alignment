# MMLU 300Q — VL-30B + vocab_lowrank VM (sia-backbone-frozen-20260706)

**Date**: 2026-07-07
**Config**: `docker-compose.sia-backbone-frozen-20260706.yml`
**VM**: VM-Qwen3-4B-vocab-lowrank-sia-backbone-frozen-20260706-merged (vocab_lowrank head, frozen backbone, BT+TD loss)
**LLM**: Qwen3-VL-30B-A3B-Instruct

## Parameters

| Parameter | Value |
|-----------|-------|
| dataset | edinburgh-dawg/mmlu-redux |
| subjects | 30 (see list below) |
| limit/subject | 10 |
| total questions | 300 |
| temperature | 1.0 |
| repetition_penalty | 1.0 |
| max_tokens (driver) | 2048 (mmlu_eval.py built-in) |
| topk (server) | 10 |
| weight (server) | 1.0 |
| entropy_threshold | 1.0 |
| max_model_len | 2048 |
| noSIA method | `--sia_weight 0.0` per-request override |

## Overall Results

| arm | accuracy | correct/300 | avg tokens/Q | throughput |
|-----|----------|-------------|--------------|------------|
| **SIA** | **0.8467** | **254/300** | 279.3 | 90.4 tok/s |
| noSIA | 0.8267 | 248/300 | 274.2 | 92.5 tok/s |
| **Δ** | **+2.0 pp** | **+6** | | |

Total wall: SIA 927s (15.5 min), noSIA 889s (14.8 min).

## Per-Subject Breakdown

| subject | SIA | noSIA | Δ |
|---------|-----|-------|---|
| anatomy | 6/10 (60.0%) | 6/10 (60.0%) | =+0/10 |
| astronomy | 9/10 (90.0%) | 8/10 (80.0%) | ++1/10 |
| business_ethics | 7/10 (70.0%) | 7/10 (70.0%) | =+0/10 |
| clinical_knowledge | 10/10 (100.0%) | 9/10 (90.0%) | ++1/10 |
| college_chemistry | 8/10 (80.0%) | 8/10 (80.0%) | =+0/10 |
| college_computer_science | 8/10 (80.0%) | 10/10 (100.0%) | --2/10 |
| college_mathematics | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| college_medicine | 9/10 (90.0%) | 9/10 (90.0%) | =+0/10 |
| college_physics | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| conceptual_physics | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| econometrics | 7/10 (70.0%) | 8/10 (80.0%) | --1/10 |
| electrical_engineering | 7/10 (70.0%) | 6/10 (60.0%) | ++1/10 |
| formal_logic | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| global_facts | 7/10 (70.0%) | 5/10 (50.0%) | ++2/10 |
| high_school_chemistry | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| high_school_geography | 8/10 (80.0%) | 8/10 (80.0%) | =+0/10 |
| high_school_macroeconomics | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| high_school_mathematics | 9/10 (90.0%) | 9/10 (90.0%) | =+0/10 |
| high_school_physics | 8/10 (80.0%) | 8/10 (80.0%) | =+0/10 |
| high_school_statistics | 9/10 (90.0%) | 8/10 (80.0%) | ++1/10 |
| high_school_us_history | 10/10 (100.0%) | 9/10 (90.0%) | ++1/10 |
| human_aging | 7/10 (70.0%) | 6/10 (60.0%) | ++1/10 |
| logical_fallacies | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| machine_learning | 8/10 (80.0%) | 7/10 (70.0%) | ++1/10 |
| miscellaneous | 10/10 (100.0%) | 10/10 (100.0%) | =+0/10 |
| philosophy | 8/10 (80.0%) | 8/10 (80.0%) | =+0/10 |
| professional_accounting | 7/10 (70.0%) | 9/10 (90.0%) | --2/10 |
| professional_law | 7/10 (70.0%) | 6/10 (60.0%) | ++1/10 |
| public_relations | 7/10 (70.0%) | 6/10 (60.0%) | ++1/10 |
| virology | 8/10 (80.0%) | 8/10 (80.0%) | =+0/10 |

SIA wins: 10/30 subjects, ties: 17/30, loses: 3/30.

## Comparison with Historical MMLU Experiments

| experiment | VM | topk | Q | SIA acc | noSIA acc | Δ |
|------------|----|------|---|---------|-----------|---|
| C.5 HTTP (vllm 0.19) | scalar head | — | 150 | 0.8000 | 0.7867 | +1.33 pp |
| C.6 b2 inproc (vllm 0.17.1) | scalar head | 5 | 150 | 0.8133 | ~0.7867 | ~+2.67 pp |
| **This run b2 inproc** | **vocab_lowrank** | **10** | **300** | **0.8467** | **0.8267** | **+2.0 pp** |

Note: noSIA baseline improved by ~4 pp (0.7867 → 0.8267) relative to historical runs. Likely cause: max_model_len=2048 (vs 4096 previously) shortens thinking chains, which helps MMLU multiple-choice answers. The SIA delta (+2.0 pp) is within the historical range (+1.33~+2.67 pp).

## Files

| file | description |
|------|-------------|
| `sia_300q.json` | SIA arm results (accuracy 0.8467, per-subject, full Q details) |
| `nosia_300q.json` | noSIA arm results (accuracy 0.8267) |
| `sia_driver.log` | per-Q CoT + latency log, SIA |
| `nosia_driver.log` | per-Q CoT + latency log, noSIA |

## Commands Used

```bash
# SIA arm
python3 eval/mmlu_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --subjects anatomy astronomy business_ethics clinical_knowledge \
             college_chemistry college_computer_science college_mathematics \
             college_medicine college_physics conceptual_physics \
             econometrics electrical_engineering formal_logic global_facts \
             high_school_chemistry high_school_geography high_school_macroeconomics \
             high_school_mathematics high_school_physics high_school_statistics \
             high_school_us_history human_aging logical_fallacies machine_learning \
             miscellaneous philosophy professional_accounting professional_law \
             public_relations virology \
  --limit 10 --temperature 1.0 --repetition_penalty 1.0 \
  --output exp/mmlu-vl30b-sia-backbone-frozen-20260707/sia_300q.json

# noSIA arm (identical + --sia_weight 0.0)
python3 eval/mmlu_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct-SIA \
  --subjects anatomy astronomy business_ethics clinical_knowledge \
             college_chemistry college_computer_science college_mathematics \
             college_medicine college_physics conceptual_physics \
             econometrics electrical_engineering formal_logic global_facts \
             high_school_chemistry high_school_geography high_school_macroeconomics \
             high_school_mathematics high_school_physics high_school_statistics \
             high_school_us_history human_aging logical_fallacies machine_learning \
             miscellaneous philosophy professional_accounting professional_law \
             public_relations virology \
  --limit 10 --temperature 1.0 --repetition_penalty 1.0 \
  --sia_weight 0.0 \
  --output exp/mmlu-vl30b-sia-backbone-frozen-20260707/nosia_300q.json
```

# B2 Decode-Mode Feasibility Analysis + M1a/M1b PoC Plan

**Date**: 2026-05-27
**Purpose**: Before investing 1100+ LOC to implement a stateful B2 RM, spend 1-2 days on a PoC to resolve the most critical technical unknown — **whether score_candidates can use the vLLM decode + CUDA graph path** — this is the determining factor for B2's performance ceiling.
**Premise**: User decided to prioritize single-GPU environment (no dependency on 2 GPUs); target RM call latency from 78ms → ≤ 15ms, overall throughput 30 tok/s → ~45-50 tok/s.

---

## 1. Background: Where B2 Currently Stands

`doc/parallel-decoding-design.md` §2.8 measured verification:
- In vLLM's `LLM.generate()` continuous decode path, Qwen3-4B single token decode = **7.02ms**
- Far less than LLM forward time L=11.3ms
- Proves B2 is feasible at the compute level

But the measurement used `LLM.generate()` — this is the **generation path**, using `lm_head` to output token logits. SIA's RM needs the **`score_head` to output a reward scalar** — **these are two different paths**.

---

## 2. fix_a_token vs score_candidates: Huge Difference in Difficulty

| Operation | Confidence | Reason |
|---|---|---|
| **fix_a_token** (push 1 token to RM on each SKIP step) | **High (80%+)** | 1:1 correspondence with vLLM `LLM.generate()` decode loop form; only needs to force-sample a specified token |
| **score_candidates** (score 5 candidates on each INTERVENE step) | **Medium (40-60%)** | `score_head` is not in the vLLM decode CUDA graph; need to solve "how to get reward on the decode path" |

Root cause of the difference — the path captured by vLLM's decode CUDA graph:

```
input_ids → embed → 36×transformer_block → lm_head → vocabulary logits → sample → output token
                                              ↑
                                  vLLM decode CUDA graph endpoint
```

What we need instead:

```
input_ids → embed → 36×transformer_block → score_head(Linear(2560,1)) → reward scalar
                                              ↑
                                  A parallel head alongside lm_head
```

**The two heads are mathematically incompatible**: lm_head outputs 151936-dim token logits; score_head outputs a 1-dim scalar. Even with the decode step's logits, the reward cannot be computed.

---

## 3. Three Potential Solutions

### Option A: ForCausalLM + external score head (recommended for first verification)

```
vLLM loads Qwen3-4B-CausalLM (not SequenceClassification) → uses decode + CUDA graph
But vLLM by default only returns logits (post-lm_head); we need the hidden state (pre-lm_head)
→ Need to extract hidden state from inside the decode loop
```

- **Advantage**: fully reuses vLLM's existing decode CUDA graph
- **Core unknown**: Can vLLM 0.10.1.1 expose last_hidden_state inside the decode loop? Needs PoC verification
- **PoC cost**: ~1 day

### Option B: Custom vLLM model adapter

```
Register a hybrid model class (inheriting Qwen3ForCausalLM):
- Keep lm_head (so vLLM decode loop works)
- Add a score_head output attribute
- Each forward step returns both lm_head logits and score
vLLM re-captures this hybrid model's decode CUDA graph
```

- **Advantage**: native support, no dependency on undocumented APIs
- **Risk**: unfamiliar with vLLM model registry mechanism and CUDA graph re-capture conditions; engineering effort ~300-500 LOC
- **PoC cost**: ~3 days

### Option C: Fallback — score_candidates still uses /classify path

```
fix_a_token: uses Option A's decode + CUDA graph, ~7ms ✓
score_candidates: still uses existing /classify, ~30ms (no change)
```

- **Advantage**: 100% technically feasible, just with reduced throughput
- **Trade-off**: score doesn't get the decode speedup; **overall benefit drops from +60% to +25%**

---

## 4. Throughput Estimates for Each Option on Single GPU

Parameters: L=11.3ms, F (fix_a_token)=7ms, S (score) varies by option, I=0.28

| Option | F | S | Per-token (single GPU) | tok/s | vs current 30 tok/s |
|---|---:|---:|---:|---:|---:|
| Current (C1+C2 / A2) | — | 78 | 33.5ms | 30 | — |
| **Options A/B both succeed** (S via decode) | 7 | 7-10 | ~20ms | **~49** | **+63%** 🎯 |
| **Partial success** (only fix via decode) | 7 | 30 | ~26.7ms | **~37** | **+25%** |
| **Complete failure** (even fix is blocked) | 30 | 30 | 49ms | 20 | -33% ❌ |

**Key insight**: `score_candidates` is decisive — if it can use decode → near ceiling; if not → benefit halved. This is why M1a/M1b must be verified first.

---

## 5. M1a/M1b PoC Implementation Plan

Split according to the principle of "verify the most critical unknowns first, then do full implementation." Each milestone has a clear go/no-go.

### 5.1 M1a — Verify whether hidden_state can be extracted from the decode loop (~1 day)

**Goal**: Find at least one way to obtain `last_hidden_state` per step (pre-lm_head, shape `[batch, hidden_dim=2560]`) in the vLLM continuous decode + CUDA graph path.

**New file**: `scripts/poc_b2_extract_hidden_state.py`

**Attempt order** (simple first, complex later):

#### Attempt (i): vLLM built-in `output_hidden_states` parameter

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/workspace/SIA/models/Qwen3-4B", dtype="bfloat16",
          gpu_memory_utilization=0.3)

# Check if SamplingParams supports this
sp = SamplingParams(temperature=0, max_tokens=10, output_hidden_states=True)
out = llm.generate(["test"], sp)
# Check if out[0].outputs[0] has a .hidden_states field
```

- ✅ If supported → **Option A lands, M1a complete in 1 day**
- ❌ If SamplingParams doesn't have this field → try (ii)

#### Attempt (ii): Write a LogitsProcessor to capture hidden state before logit computation

vLLM's `LogitsProcessor` interface only receives logits, not hidden state directly. **But the hidden state exists before calling `lm_head`** — we can hook `gpu_model_runner` or override the model forward.

```python
# Approach: subclass Qwen3ForCausalLM, override forward
class Qwen3WithHiddenStateLeak(Qwen3ForCausalLM):
    last_hidden_state = None  # class attribute side channel

    def forward(self, *args, **kwargs):
        # Run through final layer norm, before lm_head
        hidden = super().model(*args, **kwargs).last_hidden_state
        self.__class__.last_hidden_state = hidden  # store in side channel
        return self.lm_head(hidden)  # give to vLLM decode loop
```

- Register with vLLM ModelRegistry, run generate, check if `Qwen3WithHiddenStateLeak.last_hidden_state` can be retrieved
- ⚠️ Risk: vLLM runs the PagedAttention path; model forward may have been rewritten by vLLM; simple subclassing may not work

#### Attempt (iii): Directly patch vLLM `gpu_model_runner.execute_model`

```python
# In vllm/v1/worker/gpu_model_runner.py execute_model
# Add a callback: after execute completes, pass hidden_states out via zmq or SharedTensor
```

- Large engineering effort, but definitely doable
- ⚠️ vLLM upgrades require re-patching

**M1a Go/No-go**:
- ✅ Any attempt retrieves hidden_state (and CUDA graph still works) → proceed to M1b
- ❌ All three fail → try Option B (custom model adapter) or fall back to Option C

---

### 5.2 M1b — Measure score_candidates end-to-end latency + reward correctness (~1 day)

**Goal**: Building on M1a's hidden state extraction, add an external score head, measure 5-candidate real latency, and verify reward values are consistent with BF16 baseline.

**New file**: `scripts/poc_b2_score_candidates.py`

**Steps**:

```python
# 1. Load VM checkpoint's score_head weights as standalone nn.Linear
import safetensors.torch as st
score_weight = st.load_file(
    "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/model.safetensors"
)["score.weight"]  # shape (1, 2560)
score_head = torch.nn.Linear(2560, 1, bias=False).cuda().bfloat16()
score_head.weight.data = score_weight.to("cuda", dtype=torch.bfloat16)

# 2. Use M1a's hidden state extraction method + vLLM to run 5 candidates
candidates = ["...", "...", "...", "...", "..."]  # 5 chat-formatted prompts
sp = SamplingParams(temperature=0, max_tokens=1, ...)

# Measure steady-state latency (warmup + 100 iter for p50)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    out = llm.generate(candidates, sp)
    hidden = get_last_hidden_state_via_M1a_method()  # shape (5, 2560)
    rewards = score_head(hidden).squeeze(-1)  # shape (5,)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

# 3. Compare reward values with BF16 baseline
# Run the same candidates on /classify once, compare both sets of rewards' Pearson
```

**Key verification**:
- **Latency S = p50 of times**
- **Correctness**: run PoC RM + BF16 baseline on 50 different prompts × 5 candidates, compare rewards
  - Pearson correlation
  - Top-1 candidate agreement rate

**M1b Go/No-go matrix**:

| S (measured latency) | Pearson | Recommended action | Expected single-GPU throughput |
|---|---|---|---|
| < 10ms | > 0.995 | ✅ Invest in M2/M3, use Option A | ~49 tok/s |
| 10-15ms | > 0.99 | ✅ Invest in M2/M3, use Option A, slightly discounted | ~45 tok/s |
| 15-25ms | > 0.99 | ⚠️ Still viable but reduced benefit | ~40 tok/s |
| 25-30ms | > 0.99 | ⚠️ Borderline, consider Option B (custom model adapter) | ~35 tok/s |
| > 30ms or Pearson < 0.95 | any | ❌ Use Option C fallback / redesign | ~37 tok/s |

---

## 6. Decision Matrix (choose next path after M1)

```
M1a result
  ├─ ✅ Success (hidden state retrieved, CUDA graph works)
  │     ├─ M1b: S < 15ms + Pearson > 0.99
  │     │     └─→ ✅ Full investment in M2/M3 (Option A), expected +60% throughput
  │     ├─ M1b: S in 15-25ms range
  │     │     └─→ ⚠️ Still invest in M2/M3, but target reduced to +40%
  │     └─ M1b: S > 25ms or reward inaccurate
  │           └─→ Re-evaluate, consider Option B or C
  │
  └─ ❌ Failure (hidden state extraction infeasible)
        ├─→ Try Option B (custom model adapter, ~3 day PoC)
        │     └─→ If successful, redo M1b
        └─→ If B also fails, use Option C (fallback, ~+25% throughput)
```

---

## 6.5 M1a / M1b Measured Results (2026-05-27)

### M1a — hidden_state extraction ✅ PASS

- Hooked into `Qwen3ForCausalLM.compute_logits` entry; hidden_state is directly accessible
- Per-token decode = **6.84ms** (vs no-patch baseline 7.02ms; ~0.2ms extra is hook writing to /tmp)
- CUDA graph still in effect (no recompilation of any kind)
- Key insight: `monkey-patch + import` does not cross subprocess boundaries; EngineCore is a forked subprocess, so it must **directly modify vLLM source code** for the subprocess's re-import to pick up the patch
- Script: [`scripts/poc_b2_extract_hidden_state.py`](../scripts/poc_b2_extract_hidden_state.py)

### M1b — score numerical correctness ✅ PASS (after one re-run)

#### First run (mismatch): vLLM rewards vs BF16 baseline showed ~7× systematic scaling

| candidate | vLLM | BF16 | \|Δ\| |
|-----------|------|------|-------|
| '4' | +1.13 | +7.06 | 5.94 |
| '5' | -1.23 | -9.00 | 7.77 |

- Pearson = 0.9731 (< 0.99 ❌)
- Top-1 consistent ('4' highest for both), but value scale differs by ~6–8×

#### Root cause analysis (diagnose script: [`scripts/poc_b2_diagnose_scaling.py`](../scripts/poc_b2_diagnose_scaling.py))

Investigation path:
1. **vLLM final norm missing?** ❌ No. `Qwen2Model.forward` line 369 `self.norm(hidden_states, residual)` confirms final RMSNorm is applied
2. **Different vLLM/HF tokenization?** ❌ No. Dumped input_ids are identical (including trailing `[..., 151645, 198]`)
3. **Sample position off?** ❌ No. HF full-sequence vs vLLM highest cos at pos 23 (last token, cos=0.9998); position aligned
4. **vLLM / HF used different models!** ✅ **This is the root cause**:
   - Script's `MODEL_BASE = /workspace/SIA/models/Qwen3-4B` (base) → used for vLLM
   - Script's `MODEL_VM  = /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` (LoRA merged) → used for HF SequenceClassification
   - VM is a LoRA fine-tuned backbone; weights differ from base → hidden_states are naturally in different representation spaces; cos=0.617, score scale differs by 6×

#### Fix (3 code changes + 1 temporary config change)

**1. config.json** (let vLLM use generative path instead of pooling):
```diff
- "architectures": ["Qwen3ForSequenceClassification"],
+ "architectures": ["Qwen3ForCausalLM"],
```
(`config.json.bak` already backed up; when M2 goes to production, this will use a custom architecture name instead of modifying the base model config)

**2. qwen3.py `load_weights`** (vLLM strict check; it will fail on unrecognized `score.weight`):
```diff
- loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if tie else None))
+ skip = ["score."]  # VM score head does not belong to Qwen3ForCausalLM
+ if self.config.tie_word_embeddings:
+     skip.append("lm_head.")
+ loader = AutoWeightsLoader(self, skip_prefixes=skip)
```

**3. qwen3.py `compute_logits` patch** (lazy-load score head + compute reward):
```python
if type(self)._b2_score_weight is None:
    _data = safetensors.torch.load_file(
        "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/"
        "model-00002-of-00002.safetensors"  # score.weight is in shard 2
    )
    type(self)._b2_score_weight = _data["score.weight"].to(
        hidden_states.device, hidden_states.dtype
    )
rewards = (hidden_states @ type(self)._b2_score_weight.T).squeeze(-1)
```

**4. PoC script**: vLLM model path also points to VM-Qwen3-4B-merged-for-vllm.

#### Re-run results (M1b genuinely PASS)

| candidate | vLLM | BF16 | \|Δ\| |
|-----------|------|------|-------|
| '4'   | **+7.094** | +7.063 | 0.031 |
| '5'   | -8.875 | -9.000 | 0.125 |
| '3'   | -7.313 | -7.313 | 0.000 |
| '6'   | -8.938 | -8.875 | 0.063 |
| '100' | -9.000 | -8.938 | 0.063 |

- **Pearson = 0.9999** ✅ (> 0.99)
- **max \|Δ\| = 0.125** ✅ (< 0.5)
- **Top-1 = '4' consistent** ✅
- batch=5 generate p50 = 14.8ms (includes prefill; decode-only ≈ 7ms)
- Diagnostic hidden_state cos similarity = **0.9998**, norm 99.9% consistent → vLLM and HF are numerically equivalent on the same backbone

### Decision: ✅ Proceed with Option A, invest in M2/M3

Per the §6 matrix, M1b S=14.8ms (includes prefill; expected ~7ms after M2 stateful + decode-only) + Pearson > 0.99 → falls in the "**full investment in M2/M3, expected +60% throughput**" cell.

### Cleanup Required Before Starting M2

During the PoC, for rapid verification, **vLLM source code + VM config were modified directly**:
1. `/workspace/SIA/venv2/lib/python3.12/site-packages/vllm/model_executor/models/qwen3.py` — added PoC hook + modified `load_weights`. Restore before M2: `cp qwen3.py.bak qwen3.py`
2. `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/config.json` — architectures changed to Qwen3ForCausalLM. Restore: `cp config.json.bak config.json`

M2 should use vLLM `ModelRegistry.register_model` to register a new `Qwen3WithScoreForCausalLM` adapter (in our own code), to avoid modifying vLLM source or base config again — this way the patch is sticky and can be shipped inside the SIA package.

---

## 7. M2 / M3 Outline (to be refined after M1 passes)

### M2 (~400 LOC, 1 week) — Implement stateful server for fix_a_token + score_candidates

- Use `AsyncLLMEngine` to maintain long-running requests representing "RM state for each SIA session"
- `fix_a_token(req_id, token_id)`: force-sample a specified token via LogitsProcessor
- `score_candidates(req_id, candidate_token_ids)`: fork 5 branch requests, take scores after 1-token forward

### M3 (~500 LOC, 2-3 weeks) — Integrate into SIA processor + benchmark

- Modify `sia_vllm_RM.py`: replace HTTP `/classify` calls with in-process calls to M2's stateful server
- Run 600-question MMLU benchmark, compare with G3 baseline
- Write follow-up doc to archive results

---

## 8. Implementation Order Recommendation

```
M1a (~1 day)  ← Start now, verify most critical unknowns
   ↓
M1b (~1 day)  ← Follow immediately after M1a succeeds
   ↓
[Go/No-go decision point]
   ↓
M2 (~1 week)
   ↓
M3 (~2-3 weeks)
```

**As long as M1a + M1b can yield clear data within 2-3 days, the entire B2 path can be assessed within the first week as "invest or not in the next 4 weeks"** — much better than writing 1100 lines and then discovering S=30ms.

---

## 9. References

- B2 overall design: [`doc/parallel-decoding-design.md`](parallel-decoding-design.md) §2
- F=7ms measured verification: [`doc/parallel-decoding-design.md`](parallel-decoding-design.md) §2.8
- Why we can't wait for vLLM to optimize /classify: [`doc/parallel-decoding-design.md`](parallel-decoding-design.md) §2.9
- F test script: [`scripts/measure_rm_continuous_decode.py`](../scripts/measure_rm_continuous_decode.py)
- A2 failure lessons ("quantized RM acceleration doesn't work" source): [`doc/rm-fp8-dynamic-quantization-plan.md`](rm-fp8-dynamic-quantization-plan.md) §9
- Bottleneck identification and LLM vs RM compute time difference: [`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md) §4.1-§4.2

"""
Minimal reproducer for the "5 shared-prefix prompts get split into 1+4
forwards by vLLM" puzzle.

Strips away SIA / RMClient / score head. Just vanilla `LLM.generate(5 prompts)`,
with the prefix pre-warmed, and asks vLLM how many scheduler steps it
took to process them.

We instrument by monkey-patching `Scheduler.schedule` to log
`scheduler_output.total_num_scheduled_tokens` and the per-request token
counts every step.

Run:
    /workspace/SIA/venv2/bin/python scripts/repro_vllm_5_prompts_split.py [variant]

variants:
    default              — vLLM default config
    no_chunked           — enable_chunked_prefill=False (pooling-style)
    n_equals_5           — send 1 prompt with SamplingParams(n=5)
    max_partial_5        — max_num_partial_prefills=5 (probably v0 only)
    big_batched_tokens   — explicit max_num_batched_tokens=64
    inproc               — VLLM_ENABLE_V1_MULTIPROCESSING=0 (InprocClient,
                           no subprocess, no ZMQ race — expected to fix 1+4)
"""
import os
import sys
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
os.environ.setdefault("PYTHONPATH", _SRC)

# we still register the model so we don't have to download a stock Qwen,
# but the score head is irrelevant for this experiment
os.environ["SIA_RM_PROFILE"] = "1"
os.environ["SIA_REWARD_FILE_ID"] = f"repro_{uuid.uuid4().hex[:8]}"

# This must be set BEFORE importing vllm — InprocClient picks it up at
# LLMEngine.from_engine_args time.
if len(sys.argv) > 1 and sys.argv[1] == "inproc":
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print(">> VLLM_ENABLE_V1_MULTIPROCESSING=0  (InprocClient mode)")

import sia_rm  # noqa: F401 — triggers ModelRegistry.register

variant = sys.argv[1] if len(sys.argv) > 1 else "default"

MODEL = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"
PREFIX_LEN = 600
N_CAND = 5

llm_kwargs = dict(
    model=MODEL,
    hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
    dtype="bfloat16",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.3,
    max_model_len=4096,
    enforce_eager=False,
    disable_log_stats=True,
)
if variant == "no_chunked":
    llm_kwargs["enable_chunked_prefill"] = False
elif variant == "max_partial_5":
    llm_kwargs["max_num_partial_prefills"] = 5
elif variant == "big_batched_tokens":
    llm_kwargs["max_num_batched_tokens"] = 64
    llm_kwargs["max_num_seqs"] = 32

# ---- monkeypatch the scheduler to log per-step batching decisions ----
import vllm.v1.core.sched.scheduler as v1sched

_original_schedule = v1sched.Scheduler.schedule

_step_counter = {"n": 0}
_log_active = {"v": False}


def _instrumented_schedule(self):
    out = _original_schedule(self)
    if _log_active["v"]:
        _step_counter["n"] += 1
        tok = out.total_num_scheduled_tokens
        per_req = {}
        try:
            per_req = dict(out.num_scheduled_tokens)
        except Exception:
            pass
        n_new = len(out.scheduled_new_reqs)
        n_run = len(out.scheduled_cached_reqs.req_ids) if hasattr(
            out.scheduled_cached_reqs, "req_ids") else "?"
        running = len(self.running)
        waiting = len(self.waiting)
        print(f"  [scheduler] step #{_step_counter['n']}: "
              f"total_tok={tok}  new={n_new}  cached={n_run}  "
              f"running={running}  waiting={waiting}  "
              f"per_req_tok={per_req}")
    return out


v1sched.Scheduler.schedule = _instrumented_schedule


from vllm import LLM, SamplingParams, TokensPrompt

print(f"\n=== variant = {variant} ===")
print(f"LLM kwargs:")
for k, v in llm_kwargs.items():
    if k != "model":
        print(f"  {k} = {v}")

print("\nLoading LLM ...")
t0 = time.perf_counter()
llm = LLM(**llm_kwargs)
print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

import random
random.seed(42)
prefix = [random.randint(1000, 100000) for _ in range(PREFIX_LEN)]
candidates = [random.randint(1000, 100000) for _ in range(N_CAND)]

sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True)

print(">>> Warming up: send the prefix alone first (Stage A) ...")
_log_active["v"] = True
_step_counter["n"] = 0
llm.generate([TokensPrompt(prompt_token_ids=prefix)], sp, use_tqdm=False)
print(f"    Stage A used {_step_counter['n']} scheduler step(s).\n")

if variant == "n_equals_5":
    # send 1 prompt with n=5 — should generate 5 outputs from the same prefix.
    # All 5 are spawned from the same parent, so they may share a single
    # decode forward of batch=5.
    sp_n5 = SamplingParams(temperature=1.0, max_tokens=1, min_tokens=1,
                           ignore_eos=True, n=5, seed=42)
    print(">>> Test: 1 prompt with n=5 (one prompt, 5 sample branches) ...")
    _step_counter["n"] = 0
    llm.generate([TokensPrompt(prompt_token_ids=prefix)], sp_n5,
                 use_tqdm=False)
else:
    print(">>> Test: 5 shared-prefix prompts (prefix + [c_i]) ...")
    prompts = [TokensPrompt(prompt_token_ids=prefix + [c]) for c in candidates]
    _step_counter["n"] = 0
    llm.generate(prompts, sp, use_tqdm=False)

print(f"\n    Stage B used {_step_counter['n']} scheduler step(s).")
print(f"=== END variant = {variant} ===\n")

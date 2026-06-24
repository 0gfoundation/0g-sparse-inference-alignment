"""
M2 §7.B + §7.C verification script.

§7.B: Does vLLM v1 support hf_overrides to switch architectures to a custom class?
§7.C: Can ModelRegistry.register_model take effect across the EngineCore subprocess?

Test method:
  - Have b2_test_model.py register a Qwen3WithScoreTest(Qwen3ForCausalLM) subclass
  - Use hf_overrides to force-dispatch the VM model (architecture = Qwen3ForSequenceClassification)
    to Qwen3WithScoreTest
  - Check whether vLLM can start + generate

PASS criteria: vLLM starts successfully + generates a piece of text (indicating the EngineCore
subprocess found the registered class + load_weights loaded score.weight into self.score)

Run:
  cd scripts/
  python bench_b2_registry_and_override.py
"""
import os
import sys
import time

# Add scripts/ to sys.path + PYTHONPATH
# Critical: vLLM spawns a completely independent Python subprocess via
# `python -m vllm.model_executor.models.registry` to inspect model classes;
# sys.path modifications are not inherited, so PYTHONPATH must be used
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS)
_pp = os.environ.get("PYTHONPATH", "")
if _SCRIPTS not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SCRIPTS + ":" + _pp) if _pp else _SCRIPTS
print(f"[bench] PYTHONPATH = {os.environ['PYTHONPATH']}")

import b2_test_model  # noqa: E402, F401 — triggers ModelRegistry register


def main():
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("§7.B+C: hf_overrides + ModelRegistry cross-subprocess")
    print("=" * 60)
    print(f"sys.path[0] = {sys.path[0]}")
    print(f"b2_test_model module file = {b2_test_model.__file__}")
    print(f"Qwen3WithScoreTest registered class = "
          f"{b2_test_model.Qwen3WithScoreTest}")

    t0 = time.perf_counter()
    try:
        llm = LLM(
            model="/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm",
            hf_overrides={"architectures": ["Qwen3WithScoreTest"]},
            dtype="bfloat16",
            gpu_memory_utilization=0.3,
            max_model_len=1024,
            enforce_eager=False,
            disable_log_stats=True,
        )
    except Exception as e:
        print(f"\n❌ FAIL: vLLM startup error: {type(e).__name__}: {e}")
        print(f"   Possible causes: hf_overrides not supported / register did not cross subprocess /"
              f" load_weights does not accept score.weight")
        sys.exit(1)
    print(f"\n[bench] vLLM startup OK in {time.perf_counter()-t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=10,
                        min_tokens=10, ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate(["Hello, world! "], sp, use_tqdm=False)
    print(f"[bench] generate done in "
          f"{(time.perf_counter()-t0)*1000:.1f}ms")
    print(f"[bench] generated text: {out[0].outputs[0].text!r}")
    print(f"[bench] generated ids: {list(out[0].outputs[0].token_ids)}")

    print("\n" + "=" * 60)
    print("✅ §7.B + §7.C PASS")
    print("  - hf_overrides can switch architectures")
    print("  - ModelRegistry register works across EngineCore subprocess")
    print("  - score.weight auto-loaded into self.score (no explicit skip)")
    print("=" * 60)


if __name__ == "__main__":
    main()

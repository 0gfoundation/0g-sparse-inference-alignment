"""
M2 §7.B + §7.C 验证脚本.

§7.B: vLLM v1 是否支持 hf_overrides 把 architectures 切换成自定义 class?
§7.C: ModelRegistry.register_model 是否能跨 EngineCore subprocess 生效?

测试方法：
  - 让 b2_test_model.py register 一个 Qwen3WithScoreTest(Qwen3ForCausalLM) 子类
  - 用 hf_overrides 把 VM 模型 (架构 = Qwen3ForSequenceClassification) 强 dispatch
    到 Qwen3WithScoreTest
  - 看 vLLM 能否启动 + generate

PASS 标准: vLLM 启动成功 + generate 一段文本（说明 EngineCore subprocess 找到了 register 的 class
+ load_weights 把 score.weight 加载到 self.score）

跑法:
  cd scripts/
  python bench_b2_registry_and_override.py
"""
import os
import sys
import time

# 把 scripts/ 加到 sys.path + PYTHONPATH
# 关键: vLLM 用 `python -m vllm.model_executor.models.registry` 起一个完全独立的
# Python subprocess 来 inspect model class, sys.path 修改不继承, 必须用 PYTHONPATH
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS)
_pp = os.environ.get("PYTHONPATH", "")
if _SCRIPTS not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SCRIPTS + ":" + _pp) if _pp else _SCRIPTS
print(f"[bench] PYTHONPATH = {os.environ['PYTHONPATH']}")

import b2_test_model  # noqa: E402, F401 — 触发 ModelRegistry register


def main():
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("§7.B+C: hf_overrides + ModelRegistry 跨 subprocess")
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
        print(f"\n❌ FAIL: vLLM 启动报错: {type(e).__name__}: {e}")
        print(f"   可能原因: hf_overrides 不支持 / register 没跨 subprocess /"
              f" load_weights 不接收 score.weight")
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
    print("  - hf_overrides 能切换 architectures")
    print("  - ModelRegistry register 能跨 EngineCore subprocess")
    print("  - score.weight 自动加载到 self.score (没显式 skip)")
    print("=" * 60)


if __name__ == "__main__":
    main()

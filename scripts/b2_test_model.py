"""
M2 §7.B + §7.C 用的测试 model class.

放在独立 module 里，让 vLLM EngineCore subprocess 也能 import 到。
ModelRegistry 用 lazy "<module>:<class>" 字符串，subprocess fork 后会去
sys.modules / sys.path 找这个 module 并实例化 class.
"""
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM


class Qwen3WithScoreTest(Qwen3ForCausalLM):
    """Qwen3 + 外接 score head, 用来验证 hf_overrides + ModelRegistry."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = vllm_config.model_config.hf_config.hidden_size
        self.score = nn.Linear(hidden, 1, bias=False)


# 模块 import 时副作用：注册到 vLLM
# 注意 lazy "<module>:<class>" 字符串格式
ModelRegistry.register_model(
    "Qwen3WithScoreTest",
    "b2_test_model:Qwen3WithScoreTest",
)

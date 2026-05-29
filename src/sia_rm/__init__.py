"""
SIA RM: in-process stateful Reward Model client for SIA.

Public API:
  - Qwen3WithScoreForCausalLM: vLLM model class registered via ModelRegistry,
    extracts hidden_state in compute_logits hook and exposes reward via
    thread-local channel.
  - get_last_rewards(): read the rewards from the most recent compute_logits
    call (returned as a torch.Tensor on CPU, float32).

Importing this package has side effects (it registers the model class with
vLLM's ModelRegistry). When using with vLLM in subprocess mode, make sure
this package is importable from PYTHONPATH inside the EngineCore subprocess.
"""
from .qwen3_with_score import (
    Qwen3WithScoreForCausalLM,
    get_last_rewards,
    read_rewards,
    read_all_rewards,
    truncate_rewards,
    read_all_timings,
    truncate_timings,
)
from .client import RMClient

__all__ = [
    "Qwen3WithScoreForCausalLM",
    "RMClient",
    "get_last_rewards",
    "read_rewards",
    "read_all_rewards",
    "truncate_rewards",
    "read_all_timings",
    "truncate_timings",
]

"""
Qwen3 + score head, packaged as a vLLM model class.

Replaces the PoC source patch of vllm/.../qwen3.py with a subclass that vLLM
dispatches to when the model config has
`architectures=["Qwen3WithScoreForCausalLM"]` (forced via hf_overrides).

Score head loading: VM checkpoint has `score.weight` (shape (1, hidden)),
vLLM's AutoWeightsLoader auto-routes it to `self.score.weight` because the
module name matches.

Reward channel: vLLM v1's EngineCore runs in a child subprocess (NOT a fork
— spawned via `python -m`), so threading.local() / module globals cannot
bridge the boundary. The reward tensor is written to a binary file under
$SIA_REWARD_DIR (default /dev/shm, tmpfs) and read by the parent. Cost:
~50us for 5 float32 values, dwarfed by the 7-15ms forward.

Filename: `${SIA_REWARD_DIR}/sia_reward_${SIA_REWARD_FILE_ID}.bin`
- SIA_REWARD_FILE_ID lets multiple RMClient instances coexist (one per
  vLLM EngineCore subprocess). RMClient sets a unique ID per instance.

Binary format (little-endian):
  [int32 n_samples] [n_samples × float32 rewards]
"""
from __future__ import annotations

import os
import struct
from typing import Optional

import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata


def _reward_path() -> str:
    """Resolve the reward file path from env vars. Re-resolved on every call
    so RMClient can set the file id after import."""
    d = os.environ.get("SIA_REWARD_DIR", "/dev/shm")
    fid = os.environ.get("SIA_REWARD_FILE_ID", "default")
    return os.path.join(d, f"sia_reward_{fid}.bin")


def _write_rewards(rewards_cpu_f32: torch.Tensor) -> None:
    """Append rewards to file (one record per compute_logits call).

    vLLM may split a batched generate into multiple forward passes
    (prefill + decode, or batch shuffling), so the call to compute_logits
    is NOT guaranteed to see all samples at once. RMClient aggregates
    records until it has the expected count.

    Binary record: [int32 n] [n × float32]. File is opened in append mode.
    """
    path = _reward_path()
    n = rewards_cpu_f32.numel()
    rec = struct.pack("<i", n) + rewards_cpu_f32.contiguous().numpy().tobytes()
    with open(path, "ab") as f:
        f.write(rec)


def read_all_rewards() -> list:
    """Read all reward records (one per compute_logits call) written since
    the file was last truncated. Returns a list of torch.Tensor."""
    path = _reward_path()
    if not os.path.exists(path):
        return []
    out = []
    import numpy as np
    try:
        with open(path, "rb") as f:
            data = f.read()
        i = 0
        while i + 4 <= len(data):
            (n,) = struct.unpack("<i", data[i:i+4])
            i += 4
            if i + n * 4 > len(data):
                break
            arr = np.frombuffer(data[i:i+n*4], dtype=np.float32, count=n)
            out.append(torch.from_numpy(arr.copy()))
            i += n * 4
    except Exception:
        pass
    return out


def read_rewards() -> Optional[torch.Tensor]:
    """Convenience: concatenate all records into one tensor."""
    records = read_all_rewards()
    if not records:
        return None
    return torch.cat(records)


def truncate_rewards() -> None:
    """Clear the reward file. RMClient should call before each batch."""
    path = _reward_path()
    if os.path.exists(path):
        os.remove(path)


# ---------- Optional split-stage profiling (SIA_RM_PROFILE=1) ----------
#
# When SIA_RM_PROFILE=1, each compute_logits call appends a record of
# (forward_gpu_ms, score_head_gpu_ms, n_input_tokens, n_samples) to a
# /dev/shm file. RMClient reads this to attribute GPU time across stages.
#
# Wall-clock for each Stage is measured client-side by RMClient by
# splitting score_candidates into separate llm.generate() calls.

_PROFILE_ENABLED = os.environ.get("SIA_RM_PROFILE", "0") == "1"


def _timing_path() -> str:
    d = os.environ.get("SIA_REWARD_DIR", "/dev/shm")
    fid = os.environ.get("SIA_REWARD_FILE_ID", "default")
    return os.path.join(d, f"sia_timing_{fid}.bin")


def _write_timing(forward_ms: float, score_ms: float,
                  n_input: int, n_samples: int) -> None:
    """Append one timing record. Format: [f32 fwd_ms][f32 score_ms]
    [i32 n_input][i32 n_samples]."""
    path = _timing_path()
    rec = struct.pack("<ffii", forward_ms, score_ms, n_input, n_samples)
    with open(path, "ab") as f:
        f.write(rec)


def read_all_timings() -> list:
    """Read all timing records since last truncate. Returns list of
    {'forward_ms', 'score_ms', 'n_input', 'n_samples'}."""
    path = _timing_path()
    if not os.path.exists(path):
        return []
    out = []
    rec_size = 4 + 4 + 4 + 4  # ffii = 16 bytes
    try:
        with open(path, "rb") as f:
            data = f.read()
        for i in range(0, len(data), rec_size):
            chunk = data[i:i + rec_size]
            if len(chunk) < rec_size:
                break
            fwd, score, n_in, n_smp = struct.unpack("<ffii", chunk)
            out.append({
                'forward_ms': fwd, 'score_ms': score,
                'n_input': n_in, 'n_samples': n_smp,
            })
    except Exception:
        pass
    return out


def truncate_timings() -> None:
    path = _timing_path()
    if os.path.exists(path):
        os.remove(path)


class Qwen3WithScoreForCausalLM(Qwen3ForCausalLM):
    """Qwen3ForCausalLM + external score head, used as B2 RM backbone."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = vllm_config.model_config.hf_config.hidden_size
        # plain nn.Linear — single-card SIA workload, no TP needed
        self.score = nn.Linear(hidden, 1, bias=False)
        # Profile-mode state. Allocated even if disabled (cheap) so we
        # don't branch on every call.
        self._sia_prof_fwd_start: Optional[torch.cuda.Event] = None
        self._sia_prof_fwd_end: Optional[torch.cuda.Event] = None
        self._sia_prof_n_input: int = 0

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        # SIA_RM_PROFILE: time the transformer-stack forward (= KV
        # computation for the new tokens dispatched in this call). The
        # cuda.Event is recorded on the default stream; compute_logits
        # reads it after super().forward() completes.
        if _PROFILE_ENABLED:
            if input_ids is not None:
                self._sia_prof_n_input = int(input_ids.shape[0])
            elif inputs_embeds is not None:
                self._sia_prof_n_input = int(inputs_embeds.shape[0])
            self._sia_prof_fwd_start = torch.cuda.Event(enable_timing=True)
            self._sia_prof_fwd_end = torch.cuda.Event(enable_timing=True)
            self._sia_prof_fwd_start.record()

        out = super().forward(input_ids, positions, intermediate_tensors,
                              inputs_embeds)

        if _PROFILE_ENABLED:
            self._sia_prof_fwd_end.record()
        return out

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # hidden_states shape: (n_samples, hidden), already gathered to
        # sample positions by vLLM (one row per request's last token).
        if _PROFILE_ENABLED:
            score_start = torch.cuda.Event(enable_timing=True)
            score_end = torch.cuda.Event(enable_timing=True)
            score_start.record()

        try:
            sw = self.score.weight  # shape (1, hidden), bf16
            rewards = (hidden_states @ sw.T).squeeze(-1)
            if _PROFILE_ENABLED:
                score_end.record()
                # Block until both forward and score events recorded; this
                # synchronization is the cost of accurate measurement and
                # is only paid in profile mode.
                torch.cuda.synchronize()
                fwd_ms = 0.0
                if self._sia_prof_fwd_start is not None:
                    fwd_ms = self._sia_prof_fwd_start.elapsed_time(
                        self._sia_prof_fwd_end)
                score_ms = score_start.elapsed_time(score_end)
                _write_timing(fwd_ms, score_ms,
                              self._sia_prof_n_input,
                              int(hidden_states.shape[0]))
            _write_rewards(rewards.detach().to("cpu", torch.float32))
        except Exception:
            # Don't crash decode if reward/timing write fails — RMClient
            # will see stale or None and can decide what to do
            pass

        # Still run lm_head logits — vLLM scheduler needs them to sample
        # (a dummy) token to keep the request alive.
        return super().compute_logits(hidden_states, sampling_metadata)


# Back-compat alias for callers that imported the old name
def get_last_rewards() -> Optional[torch.Tensor]:
    return read_rewards()


# Register on module import.
ModelRegistry.register_model(
    "Qwen3WithScoreForCausalLM",
    "sia_rm.qwen3_with_score:Qwen3WithScoreForCausalLM",
)

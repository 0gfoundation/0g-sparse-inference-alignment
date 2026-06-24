"""Dummy LogitsProcessor variants for SIA overhead calibration.

Goal: measure how much of SIA's per-token overhead is structural (the
sync wait for GPU forward to complete, or the cost of being a registered
LogitsProcessor at all) vs. truly added by SIA's intervention logic
(entropy decision + RM call + index_add_).

Two variants:

  noop:  apply(logits) → return logits unchanged. Tests the cost of vLLM
         simply having a logits_processor hook at all.

  sync:  Replicates SIA's apply() up to (and including) the cpu().tolist()
         sync, but does NOT make an intervention decision, does NOT call
         the RM, and does NOT modify logits. Tests how much of the
         3.36 ms apply_cpu_sync seen in SIA is unavoidable LLM-forward
         sync (which sampler would have done anyway) vs. truly added.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate, LogitsProcessor,
)


def make_dummy_processor(variant: str, topk: int = 5):
    """Returns a LogitsProcessor class (not instance) compatible with
    vllm.AsyncEngineArgs(logits_processors=[...]).

    variant: 'noop' | 'sync'
    """
    if variant not in ("noop", "sync"):
        raise ValueError(f"variant must be 'noop' or 'sync', got {variant}")

    _VARIANT = variant
    _TOPK = topk

    class DummyLogitsProcessor(LogitsProcessor):
        # per-token profiling counters (same shape as SIA processor's)
        _PROFILE_DETAIL: bool = os.environ.get("SIA_PROFILE", "1") == "1"
        _PF_INTERVAL: int = int(os.environ.get("SIA_PF_INTERVAL", "200"))

        def __init__(self, vllm_config, device: torch.device,
                     is_pin_memory: bool) -> None:
            self._device = device
            self._pf_stats: dict[str, list] = {
                "apply_total":     [],
                "apply_topk_ent":  [],
                "apply_cpu_sync":  [],
            }
            self._n_apply = 0
            print(f"[dummy-{_VARIANT}] LogitsProcessor ready", flush=True)

        def is_argmax_invariant(self) -> bool:
            # noop / sync do not modify logits, argmax is unchanged
            return True

        def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
            # No per-request state to maintain
            pass

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if _VARIANT == "noop":
                # Do not touch logits, no sync — measure pure hook overhead
                if self._PROFILE_DETAIL:
                    self._n_apply += 1
                    if self._n_apply % self._PF_INTERVAL == 0:
                        # noop variant has nothing to record, just print count
                        print(f"[dummy-noop @{self._n_apply}] no work",
                              flush=True)
                return logits

            # variant == 'sync': replicate SIA processor's entropy + sync flow,
            # no decision, no RM call, no logits modification. Measures "how much
            # does SIA's sync step alone cost"
            pf_on = self._PROFILE_DETAIL
            t0 = time.perf_counter() if pf_on else 0.0

            topk_result = torch.topk(logits, _TOPK, dim=-1)
            log_probs = F.log_softmax(topk_result.values.float(), dim=-1)
            probs = log_probs.exp()
            entropies = -(probs * log_probs).sum(dim=-1)
            t_after_dispatch = time.perf_counter() if pf_on else 0.0

            _ = entropies.cpu().tolist()  # GPU pipeline sync
            t_after_sync = time.perf_counter() if pf_on else 0.0

            # No decision made, return directly
            if pf_on:
                self._pf_stats["apply_total"].append(
                    (t_after_sync - t0) * 1000)
                self._pf_stats["apply_topk_ent"].append(
                    (t_after_dispatch - t0) * 1000)
                self._pf_stats["apply_cpu_sync"].append(
                    (t_after_sync - t_after_dispatch) * 1000)
                self._n_apply += 1
                if self._n_apply % self._PF_INTERVAL == 0:
                    parts = []
                    for k, arr in self._pf_stats.items():
                        if not arr:
                            continue
                        s = sorted(arr)
                        n = len(s)
                        parts.append(
                            f"{k}: p50={s[n//2]:.2f} "
                            f"p95={s[min(n-1,int(n*0.95))]:.2f} "
                            f"max={s[-1]:.2f}"
                        )
                    print(f"[dummy-sync @{self._n_apply}] " +
                          " | ".join(parts), flush=True)

            return logits

    return DummyLogitsProcessor

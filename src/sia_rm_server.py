"""
SIA RM Server — standalone Reward Model scoring service

Deploys the RM (Value Model + optional LoRA) as an independent FastAPI service.
The main LLM server (sia_vllm_server.py) calls this service via HTTP for scoring.
Hot-swapping the RM only requires calling /reload — no need to restart vLLM.

Endpoints:
  POST /score    — batch score topk candidate tokens (per-token call)
  POST /reload   — hot-swap the RM (synchronous, returns after completion)
  GET  /status   — query current load status
  GET  /health   — health check

Usage:
  # Without LoRA
  python sia_rm_server.py \\
    --rm /path/to/rm --rm_device cuda:0 --port 8001

  # With LoRA
  python sia_rm_server.py \\
    --rm /path/to/rm --rm_lora /path/to/lora --rm_device cuda:0 --port 8001
"""

import argparse
import os
import threading
import time
from typing import Optional

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------------------------
# ValueModel wrapper (consistent with the original sia_vllm_RM.py logic)
# ---------------------------------------------------------------------------

class _ValueModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _ValueModelWrapper(nn.Module):
    def __init__(self, base_model, token_reward_head: nn.Linear):
        super().__init__()
        self.base_model = base_model
        self.token_reward_head = token_reward_head
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None,
                past_key_values=None, use_cache=False,
                position_ids=None, **kwargs):
        if hasattr(self.base_model, 'model'):
            backbone = self.base_model.model
        elif hasattr(self.base_model, 'transformer'):
            backbone = self.base_model.transformer
        else:
            backbone = self.base_model

        # Newer HF backbone requires a Cache object, does not accept legacy tuple
        kv_in = _to_dynamic_cache(past_key_values)
        prefix_kv_len = _dc_seq_len(kv_in)

        backbone_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_in,
            use_cache=use_cache,
            output_hidden_states=False,
        )
        if position_ids is not None:
            backbone_kwargs["position_ids"] = position_ids
        out = backbone(**backbone_kwargs)

        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out.last_hidden_state

        token_rewards = self.token_reward_head(hidden.float()).squeeze(-1)  # (batch, seq)

        # CUDA graph requires arange to be created on device, to avoid CPU→GPU copy
        # triggering "operation not permitted when stream is capturing"
        _batch_idx = torch.arange(
            token_rewards.size(0), device=token_rewards.device,
        )
        if past_key_values is not None:
            # KV cache mode: find the last valid token in the diff
            if attention_mask is not None:
                diff_mask = attention_mask[:, prefix_kv_len:]
                diff_lens = diff_mask.sum(dim=1) - 1
                logits = token_rewards[_batch_idx, diff_lens].unsqueeze(-1)
            else:
                logits = token_rewards[:, -1].unsqueeze(-1)
        elif attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            logits = token_rewards[_batch_idx, seq_lens].unsqueeze(-1)
        else:
            logits = token_rewards[:, -1].unsqueeze(-1)

        if use_cache:
            return _ValueModelOutputWithPast(
                logits=logits,
                past_key_values=getattr(out, 'past_key_values', None),
            )
        return _ValueModelOutput(logits=logits)


class _ValueModelOutputWithPast:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_rm_model = None
_rm_tok = None
_rm_device: str = "cuda:0"
_rm_lock = threading.Lock()
_status = {"status": "initializing", "rm": "", "rm_lora": None}

# Cross-step KV state: request_id → {"prefix_ids": list[int], "kv": DynamicCache}
_req_kv: dict = {}
_MAX_REQ_KV = 64

# Number of trailing tokens to trim, to eliminate BPE boundary effects
_BPE_TRIM = 5

# hit/miss stats (for logging only)
_kv_stats = {"hit": 0, "miss": 0}

# Whether to enable torch.compile (set to True when CLI --compile is passed)
_compile_mode: bool = False

# ---------------------------------------------------------------------------
# Profiling switches and stats
# ---------------------------------------------------------------------------
# Set env var RM_PROFILE=0 to disable detailed timing; default is 1 (enabled)
_PROFILE_DETAIL: bool = os.environ.get("RM_PROFILE", "1") == "1"

# Per-phase elapsed time (ms) for rolling stats
_pf_phases = [
    "tokenize", "prefix_calc", "state_lookup",
    "prep_tensor", "kv_expand", "forward", "score_extract", "kv_save",
    "miss_prefix_fwd",
]
_pf_stats: dict = {p: [] for p in _pf_phases}
_pf_call_count: int = 0
_PF_STATS_INTERVAL = 50  # print p50/p95/max summary every N calls


def _pf_now():
    """Synchronize GPU then return high-precision CPU time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _pf_record(phase: str, t_start: float, t_end: float):
    if not _PROFILE_DETAIL:
        return
    dt = (t_end - t_start) * 1000.0
    arr = _pf_stats.get(phase)
    if arr is not None:
        arr.append(dt)
        if len(arr) > 200:
            del arr[:100]


def _pf_summary_if_due():
    """Print a p50/p95/max summary every _PF_STATS_INTERVAL calls."""
    global _pf_call_count
    if not _PROFILE_DETAIL:
        return
    _pf_call_count += 1
    if _pf_call_count % _PF_STATS_INTERVAL != 0:
        return
    parts = []
    for phase in _pf_phases:
        arr = _pf_stats.get(phase) or []
        if not arr:
            continue
        s = sorted(arr[-100:])
        n = len(s)
        p50 = s[n // 2]
        p95 = s[min(int(n * 0.95), n - 1)]
        mx = s[-1]
        parts.append(f"{phase}: p50={p50:.1f} p95={p95:.1f} max={mx:.1f}")
    print(f"[RM-pf-summary @{_pf_call_count}] " + " | ".join(parts), flush=True)
    # Also print a per-layer summary every N calls (if enabled)
    _pf_layer_summary()


# === Per-layer profiling (optional, enabled via env var RM_PROFILE_LAYERS=1) ===
_PROFILE_LAYERS: bool = os.environ.get("RM_PROFILE_LAYERS", "0") == "1"
_layer_times: dict = {}      # layer_idx → [ms, ms, ...]
_layer_count: int = 0
_layer_active: bool = False  # temporarily set by hook during forward
_layer_t_prev: float = 0.0   # timestamp from the previous layer's hook


def _layer_pre_hook(idx):
    def _hook(_module, _inputs):
        global _layer_t_prev
        if not _layer_active:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _layer_t_prev = time.perf_counter()
    return _hook


def _layer_post_hook(idx):
    def _hook(_module, _inputs, _output):
        global _layer_t_prev
        if not _layer_active:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        dt = (t - _layer_t_prev) * 1000.0
        _layer_times.setdefault(idx, []).append(dt)
        if len(_layer_times[idx]) > 100:
            _layer_times[idx] = _layer_times[idx][-50:]
    return _hook


def _install_layer_hooks(model):
    """Install pre/post hooks on each transformer decoder layer. Requires RM_PROFILE_LAYERS=1."""
    if not _PROFILE_LAYERS:
        return
    # Find the base transformer (Qwen3ForSequenceClassification's .model.layers)
    base = getattr(model, "base_model", model)
    backbone = getattr(base, "model", None) or getattr(base, "transformer", None) or base
    layers = getattr(backbone, "layers", None)
    if layers is None:
        print("[RM] _install_layer_hooks: layers not found, skip", flush=True)
        return
    n = 0
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(_layer_pre_hook(i))
        layer.register_forward_hook(_layer_post_hook(i))
        n += 1
    print(f"[RM] installed per-layer hooks on {n} layers", flush=True)


def _pf_layer_summary():
    if not _PROFILE_LAYERS or not _layer_times:
        return
    # Take the average time per layer from the last 50 calls and print
    arr_per_layer = []
    for idx in sorted(_layer_times.keys()):
        v = _layer_times[idx]
        if not v:
            continue
        arr_per_layer.append(sum(v) / len(v))
    if not arr_per_layer:
        return
    total = sum(arr_per_layer)
    avg = total / len(arr_per_layer)
    mx = max(arr_per_layer)
    mn = min(arr_per_layer)
    print(
        f"[RM-pf-layers] n_layers={len(arr_per_layer)} "
        f"avg_per_layer={avg:.2f}ms min={mn:.2f}ms max={mx:.2f}ms "
        f"total_layers_time={total:.1f}ms",
        flush=True,
    )


# ---------------------------------------------------------------------------
# DynamicCache helpers
# ---------------------------------------------------------------------------

def _to_dynamic_cache(past_kv):
    """Convert any past_key_values format to DynamicCache."""
    if past_kv is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return past_kv
    if isinstance(past_kv, DynamicCache):
        return past_kv
    if isinstance(past_kv, tuple):
        return DynamicCache.from_legacy_cache(past_kv)
    return past_kv


def _dc_seq_len(dc) -> int:
    """Get the cached sequence length from a DynamicCache."""
    if dc is None:
        return 0
    if hasattr(dc, 'get_seq_length'):
        return dc.get_seq_length()
    if hasattr(dc, 'key_cache') and dc.key_cache:
        return dc.key_cache[0].shape[-2]
    return 0


def _dc_expand_batch(dc, k: int):
    """Expand a batch=1 DynamicCache to batch=k (materializing via .contiguous()).
    Compatible with transformers 4.57+ new API (layers[i].keys/values + update()).
    """
    from transformers.cache_utils import DynamicCache
    new_dc = DynamicCache()
    n_layers = 0
    for i, layer in enumerate(dc.layers):
        K = layer.keys.expand(k, -1, -1, -1).contiguous()
        V = layer.values.expand(k, -1, -1, -1).contiguous()
        new_dc.update(K, V, i)
        n_layers += 1
    # Record shape for single-line output reference when _PROFILE_DETAIL is enabled
    if _PROFILE_DETAIL:
        new_dc._pf_n_layers = n_layers  # type: ignore
    return new_dc


def _dc_extract_batch0_trim(dc, keep_len: int):
    """
    Extract batch[0] from a batch>=1 DynamicCache, keeping only the first keep_len positions.
    The KV for prefix tokens is identical across all batches (in causal attention, pos<keep_len
    does not depend on subsequent tokens).
    Compatible with transformers 4.57+ new API.
    """
    try:
        from transformers.cache_utils import DynamicCache
        new_dc = DynamicCache()
        for i, layer in enumerate(dc.layers):
            K = layer.keys[0:1, :, :keep_len, :].contiguous()
            V = layer.values[0:1, :, :keep_len, :].contiguous()
            new_dc.update(K, V, i)
        return new_dc
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CUDA graph static shape bucketing
# ---------------------------------------------------------------------------

# Global: single HIT bucket (covers the most common batch=topk, diff<=64, kv<=2048 cases)
_CUDA_GRAPH_ENABLED: bool = False
_hit_bucket = None  # _CudaGraphHitBucket instance

# Adaptive disable (per-request): if a request hits too many NaNs in the bucket,
# only that request's bucket is disabled; other requests still benefit from acceleration.
# Global disable is triggered in the worst case (to prevent catastrophic failure).
_cg_request_nan: dict = {}        # request_id → consecutive NaN count
_CG_PER_REQ_NAN_THRESHOLD: int = 5  # downgrade if same request gets 5 consecutive NaNs
_cg_per_req_disabled: set = set()  # set of downgraded request_ids

# Global safety net
_cg_consecutive_nan: int = 0
_CG_NAN_DISABLE_THRESHOLD: int = 50  # disable globally after 50 cumulative consecutive NaNs across requests
_cg_runtime_disabled: bool = False

# Model architecture info, used during bucket initialization
_mc_num_layers: int = 0
_mc_num_kv_heads: int = 0
_mc_head_dim: int = 0
_mc_dtype = None


class _CudaGraphHitBucket:
    """
    Single static-shape CUDA graph bucket for HIT path forward.

    Covered inputs:
      - input_ids:      (batch, diff_max)
      - attention_mask: (batch, kv_max + diff_max)
      - past_key_values: DynamicCache with batch=batch, kv_max positions per layer
      - position_ids:   (batch, diff_max)   — explicitly passed, because the diff start
                                              position in the buffer is kv_max (not the real kv_actual)

    Buffer layout (actual kv_actual <= kv_max, diff_actual <= diff_max):
      KV buffer [0..kv_max-1]:
        [0..kv_actual-1]  <- real prefix KV (input)
        [kv_actual..kv_max-1] <- padding (garbage, masked by attention_mask)
      input_ids [0..diff_max-1]:
        [0..diff_actual-1] <- real diff tokens
        [diff_actual..diff_max-1] <- padding (masked by attention_mask)
      attention_mask [0..kv_max+diff_max-1]:
        [0..kv_actual-1]                   = real prefix mask
        [kv_actual..kv_max-1]              = 0 (mask out padded prefix)
        [kv_max..kv_max+diff_actual-1]     = real diff mask
        [kv_max+diff_actual..]             = 0
      position_ids [0..diff_max-1]:
        [i] = kv_actual + i  for i in 0..diff_actual-1 (RoPE uses real absolute positions)
    """

    def __init__(self, model_wrapper, batch, diff_max, kv_max,
                 num_layers, num_kv_heads, head_dim, dtype, device):
        self.batch = batch
        self.diff_max = diff_max
        self.kv_max = kv_max
        self.full = kv_max + diff_max
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.model_wrapper = model_wrapper

        # Input buffers (CPU→GPU channel, written by run() before each replay)
        self.input_ids = torch.zeros(batch, diff_max, dtype=torch.long, device=device)
        self.attention_mask = torch.zeros(batch, self.full, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(batch, diff_max, dtype=torch.long, device=device)

        # Prefix KV buffers: per layer (batch, num_kv_heads, kv_max, head_dim)
        # Referenced by DynamicCache; attention reads from here during model forward
        self.prefix_K = [
            torch.zeros(batch, num_kv_heads, kv_max, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.prefix_V = [
            torch.zeros(batch, num_kv_heads, kv_max, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # Set by capture: graph + output tensor references
        self.graph = None
        self.output_logits = None   # (batch, 1)
        self.full_K = None          # list of (batch, h, full, d) — post-forward K
        self.full_V = None

    def _new_dc(self):
        """Construct a DynamicCache whose layers point to self.prefix_K / prefix_V."""
        from transformers.cache_utils import DynamicCache
        dc = DynamicCache()
        for i in range(self.num_layers):
            dc.update(self.prefix_K[i], self.prefix_V[i], i)
        return dc

    def capture(self, n_warmup: int = 3):
        """Run warmup forwards then capture the CUDA graph."""
        print(
            f"[RM] CudaGraph warmup+capture: batch={self.batch} "
            f"diff_max={self.diff_max} kv_max={self.kv_max} "
            f"layers={self.num_layers} kv_heads={self.num_kv_heads} head_dim={self.head_dim}",
            flush=True,
        )
        # Initialize buffers to valid non-zero values
        self.input_ids.fill_(1)
        # attention_mask: use the sparse layout from real inference (some zeros), to ensure
        # SDPA dispatch selects a backend that supports masks (mem_efficient/math) rather
        # than being misled by an all-ones mask into choosing Flash
        # (Flash doesn't support custom mask values; it treats all keys as valid).
        self.attention_mask.zero_()
        cap_kv_actual = self.kv_max // 2          # pretend half is real prefix
        cap_diff_actual = self.diff_max // 2      # pretend half is real diff
        self.attention_mask[:, :cap_kv_actual] = 1
        self.attention_mask[:, self.kv_max:self.kv_max + cap_diff_actual] = 1
        pos_diff = torch.arange(
            self.kv_max, self.kv_max + self.diff_max,
            dtype=torch.long, device=self.device,
        )
        self.position_ids.copy_(pos_diff.unsqueeze(0).expand(self.batch, -1))

        # Warmup runs (no capture, let cuDNN / kernel settle on a stable choice)
        for w in range(n_warmup):
            dc = self._new_dc()
            with torch.inference_mode():
                _ = self.model_wrapper(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    past_key_values=dc,
                    use_cache=True,
                    position_ids=self.position_ids,
                )
            del dc
        torch.cuda.synchronize()

        # Capture
        capture_dc = self._new_dc()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.inference_mode():
                out = self.model_wrapper(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    past_key_values=capture_dc,
                    use_cache=True,
                    position_ids=self.position_ids,
                )
        # Save references to output tensors allocated during capture (fixed addresses)
        self.output_logits = out.logits  # (batch, 1)
        self.full_K = [layer.keys for layer in capture_dc.layers]
        self.full_V = [layer.values for layer in capture_dc.layers]
        # Retain capture_dc reference to prevent tensors from being freed
        self._capture_dc = capture_dc

        print(
            f"[RM] CudaGraph captured. "
            f"output_logits shape={tuple(self.output_logits.shape)} "
            f"full_K[0] shape={tuple(self.full_K[0].shape)}",
            flush=True,
        )

    def can_fit(self, batch: int, diff_actual: int, kv_actual: int) -> bool:
        return (batch == self.batch
                and diff_actual <= self.diff_max
                and kv_actual <= self.kv_max)

    def run(self, prefix_dc, diff_ids, attn_mask_compact, position_ids_actual,
            extension_len: int):
        """
        Run one HIT forward using CUDA graph.

        Args:
          prefix_dc: DynamicCache, batch=1, kv_actual positions
          diff_ids: (batch, diff_actual)
          attn_mask_compact: (batch, kv_actual + diff_actual) — original compact layout,
                              this method rearranges it to the bucket's sparse layout
          position_ids_actual: (batch, diff_actual) — real absolute positions [kv_actual..]
          extension_len: the first N tokens of the diff sequence belong to "extension",
                          used to build the new prefix (reused on next call)

        Returns:
          logits: (batch, 1)
          new_prefix_dc: DynamicCache batch=1, kv_actual + extension_len positions

        Returns None on failure (including NaN, CUDA errors, etc.); caller falls back to eager.
        """
        try:
            return self._run_unsafe(prefix_dc, diff_ids, attn_mask_compact,
                                     position_ids_actual, extension_len)
        except Exception as e:
            # No exception (including delayed CUDA async error reports) should bring down the server
            import traceback
            print(
                f"[RM-cg-error] Exception in bucket.run(): {type(e).__name__}: {e}\n"
                f"Stack:\n{traceback.format_exc()}\n"
                f"  batch={diff_ids.size(0) if diff_ids is not None else '?'} "
                f"diff_actual={diff_ids.size(1) if diff_ids is not None else '?'} "
                f"extension_len={extension_len}",
                flush=True,
            )
            return None

    def _run_unsafe(self, prefix_dc, diff_ids, attn_mask_compact,
                     position_ids_actual, extension_len: int):
        batch = diff_ids.size(0)
        diff_actual = diff_ids.size(1)
        kv_actual = prefix_dc.layers[0].keys.size(2)

        assert batch == self.batch
        assert diff_actual <= self.diff_max
        assert kv_actual <= self.kv_max

        # 0) Input prefix_dc integrity check (debug: locate NaN source) — scan all layers
        in_K_stack = torch.stack([layer.keys for layer in prefix_dc.layers])
        in_V_stack = torch.stack([layer.values for layer in prefix_dc.layers])
        in_K_finite = torch.isfinite(in_K_stack).all().item()
        in_V_finite = torch.isfinite(in_V_stack).all().item()
        if not (in_K_finite and in_V_finite):
            print(
                f"[RM-cg-warn] INPUT prefix_dc has NaN/Inf (K_finite={in_K_finite} "
                f"V_finite={in_V_finite})! kv_actual={kv_actual} → fallback to eager",
                flush=True,
            )
            return None

        # 1) Write input_ids (zero out padding)
        self.input_ids.zero_()
        self.input_ids[:, :diff_actual].copy_(diff_ids)

        # 2) Write position_ids (fill padding with valid consecutive positions)
        self.position_ids[:, :diff_actual].copy_(position_ids_actual)
        if diff_actual < self.diff_max:
            # Padding values don't matter since attention_mask=0; use consecutive positions to avoid RoPE errors
            pad_pos = torch.arange(
                kv_actual + diff_actual, kv_actual + self.diff_max,
                dtype=torch.long, device=self.device,
            )
            self.position_ids[:, diff_actual:].copy_(pad_pos.unsqueeze(0).expand(self.batch, -1))

        # 3) Rearrange attention_mask to bucket sparse layout:
        #    [0:kv_actual] = real prefix mask (compact input [0:kv_actual])
        #    [kv_actual:kv_max] = 0 (padded prefix, masked out)
        #    [kv_max:kv_max+diff_actual] = real diff mask
        #    [kv_max+diff_actual:] = 0
        self.attention_mask.zero_()
        self.attention_mask[:, :kv_actual].copy_(attn_mask_compact[:, :kv_actual])
        self.attention_mask[:, self.kv_max:self.kv_max + diff_actual].copy_(
            attn_mask_compact[:, kv_actual:kv_actual + diff_actual]
        )

        # 4) Copy prefix KV to the first kv_actual positions of the buffer (expand batch=1 → batch=k)
        #    Critical: must explicitly zero out the padded portion [kv_actual:kv_max], because:
        #    - those positions may retain real K/V values from the previous call (not all-zeros)
        #    - mem_efficient attention's max-shifted softmax doing Q@K on stale K values
        #      can produce numerical instability (even though mask=0 should mask them out,
        #      empirically stale data can poison the entire bucket output → NaN)
        for i in range(self.num_layers):
            # Zero the entire buffer first, then write the real prefix
            self.prefix_K[i].zero_()
            self.prefix_V[i].zero_()
            self.prefix_K[i][:, :, :kv_actual, :].copy_(
                prefix_dc.layers[i].keys.expand(self.batch, -1, -1, -1)
            )
            self.prefix_V[i][:, :, :kv_actual, :].copy_(
                prefix_dc.layers[i].values.expand(self.batch, -1, -1, -1)
            )

        # 5) Replay
        self.graph.replay()

        # 6) Extract logits (clone to avoid overwrite on next replay)
        logits = self.output_logits.clone()

        # 6.5) NaN/Inf safety check — return None on occasional NaN to trigger eager fallback
        #      (root cause TBD; usually occurs after a long-prefix MISS, which poisons all subsequent HITs)
        if not torch.isfinite(logits).all():
            nan_cnt = int((~torch.isfinite(logits)).sum().item())
            print(
                f"[RM-cg-warn] NaN/Inf in bucket output: nan_cnt={nan_cnt} "
                f"batch={batch} diff_actual={diff_actual} kv_actual={kv_actual} "
                f"extension_len={extension_len} → fallback to eager",
                flush=True,
            )
            return None

        # 7) Build new_prefix_dc (batch=1, kv_actual + extension_len positions)
        #    Data sources:
        #      [0:kv_actual]: full_K[0:1, :, 0:kv_actual, :] (original prefix, unchanged)
        #      [kv_actual:kv_actual+ext]: full_K[0:1, :, kv_max:kv_max+ext, :] (newly computed extension)
        new_prefix_len = kv_actual + extension_len
        from transformers.cache_utils import DynamicCache
        new_dc = DynamicCache()
        # 7a) Also check the extension portion (new K/V) for NaN:
        #     logits might come back clean due to masking even if the K/V has NaN,
        #     but saving that into _req_kv and using it on the next HIT → guaranteed NaN.
        #     So stack and scan here.
        nan_in_extension = False
        if extension_len > 0:
            ext_K_stack = torch.stack(
                [self.full_K[i][0:1, :, self.kv_max:self.kv_max + extension_len, :]
                 for i in range(self.num_layers)]
            )
            ext_V_stack = torch.stack(
                [self.full_V[i][0:1, :, self.kv_max:self.kv_max + extension_len, :]
                 for i in range(self.num_layers)]
            )
            nan_in_extension = (
                not torch.isfinite(ext_K_stack).all().item()
                or not torch.isfinite(ext_V_stack).all().item()
            )
            if nan_in_extension:
                print(
                    f"[RM-cg-warn] NaN/Inf in bucket extension K/V (logits were clean "
                    f"but new prefix would be poisoned)! "
                    f"batch={batch} diff_actual={diff_actual} kv_actual={kv_actual} "
                    f"extension_len={extension_len} → fallback to eager",
                    flush=True,
                )
                return None

        for i in range(self.num_layers):
            new_K_i = torch.empty(
                1, self.num_kv_heads, new_prefix_len, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            new_V_i = torch.empty_like(new_K_i)
            new_K_i[:, :, :kv_actual, :].copy_(self.full_K[i][0:1, :, :kv_actual, :])
            new_V_i[:, :, :kv_actual, :].copy_(self.full_V[i][0:1, :, :kv_actual, :])
            if extension_len > 0:
                new_K_i[:, :, kv_actual:new_prefix_len, :].copy_(
                    self.full_K[i][0:1, :, self.kv_max:self.kv_max + extension_len, :]
                )
                new_V_i[:, :, kv_actual:new_prefix_len, :].copy_(
                    self.full_V[i][0:1, :, self.kv_max:self.kv_max + extension_len, :]
                )
            new_dc.update(new_K_i, new_V_i, i)
        return logits, new_dc


def _init_cuda_graph_buckets(model_wrapper, device, batch, diff_max, kv_max):
    """Initialize the HIT bucket. Returns False on failure (caller should fall back to eager).

    Key technique: SDPA backend selection gets baked into the CUDA graph (no re-dispatch on replay).
    So we temporarily switch to the mem_efficient backend for capture, then restore to Flash
    after capture completes, allowing the eager path to keep using Flash (faster) while
    bucket replay uses mem_efficient (correctly handles masks).
    """
    global _hit_bucket, _mc_num_layers, _mc_num_kv_heads, _mc_head_dim, _mc_dtype

    # Save original SDPA flags
    orig_flash = orig_mem_eff = orig_math = orig_cudnn = None
    try:
        orig_flash = torch.backends.cuda.flash_sdp_enabled()
        orig_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
        orig_math = torch.backends.cuda.math_sdp_enabled()
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
            orig_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
        print(
            f"[RM] Original SDPA flags: flash={orig_flash} mem_eff={orig_mem_eff} "
            f"math={orig_math} cudnn={orig_cudnn}",
            flush=True,
        )
    except Exception as e:
        print(f"[RM] Failed to read SDPA flags ({e}); continuing", flush=True)

    # Temporarily switch to mem_efficient backend (for capture)
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        print(
            "[RM] Capture-time SDPA: flash=OFF mem_eff=ON math=ON cudnn=OFF",
            flush=True,
        )
    except Exception as e:
        print(f"[RM] Failed to set SDPA backend ({e}); continuing anyway", flush=True)

    init_ok = False
    try:
        config = model_wrapper.config
        _mc_num_layers = config.num_hidden_layers
        _mc_num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        _mc_head_dim = (
            getattr(config, "head_dim", None)
            or (config.hidden_size // config.num_attention_heads)
        )
        _mc_dtype = next(model_wrapper.parameters()).dtype
        print(
            f"[RM] CudaGraph init: layers={_mc_num_layers} "
            f"kv_heads={_mc_num_kv_heads} head_dim={_mc_head_dim} dtype={_mc_dtype}",
            flush=True,
        )
        bucket = _CudaGraphHitBucket(
            model_wrapper=model_wrapper,
            batch=batch,
            diff_max=diff_max,
            kv_max=kv_max,
            num_layers=_mc_num_layers,
            num_kv_heads=_mc_num_kv_heads,
            head_dim=_mc_head_dim,
            dtype=_mc_dtype,
            device=device,
        )
        bucket.capture()
        _hit_bucket = bucket
        init_ok = True
        return True
    except Exception as e:
        print(f"[RM] CudaGraph init failed ({type(e).__name__}: {e}); fall back to eager.", flush=True)
        import traceback
        traceback.print_exc()
        _hit_bucket = None
        return False
    finally:
        # Restore original SDPA flags: allow the eager path to keep using Flash (faster).
        # The CUDA graph has already baked in the mem_eff backend from capture time,
        # so replay won't re-dispatch — this restore does not affect the bucket.
        try:
            if orig_flash is not None:
                torch.backends.cuda.enable_flash_sdp(orig_flash)
            if orig_mem_eff is not None:
                torch.backends.cuda.enable_mem_efficient_sdp(orig_mem_eff)
            if orig_math is not None:
                torch.backends.cuda.enable_math_sdp(orig_math)
            if orig_cudnn is not None and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(orig_cudnn)
            print(
                f"[RM] Restored SDPA flags (post-capture): flash={orig_flash} "
                f"mem_eff={orig_mem_eff} math={orig_math} cudnn={orig_cudnn} "
                f"— eager path will use these.",
                flush=True,
            )
        except Exception as e:
            print(f"[RM] Failed to restore SDPA flags ({e})", flush=True)


def _find_common_prefix(seqs: list) -> list:
    if not seqs:
        return []
    min_len = min(len(s) for s in seqs)
    prefix = []
    for i in range(min_len):
        if all(s[i] == seqs[0][i] for s in seqs):
            prefix.append(seqs[0][i])
        else:
            break
    return prefix


# ---------------------------------------------------------------------------
# Core: single-forward KV cache scoring
# ---------------------------------------------------------------------------

def _score_with_prefix_kv(
    prefix_dc,          # DynamicCache, batch=1, seq=stable_prefix_len
    stable_prefix_len: int,
    diff_seqs: list,    # list[list[int]], diff token IDs per candidate (extension + cand)
    update_len: int,    # length of the "extension" prefix within diff_seqs (used to update cache)
    request_id: Optional[str] = None,  # used for per-request bucket disable decision
) -> tuple:
    """
    Single forward: batch=k, seq=max(diff_len), using prefix_dc as KV cache.
    Returns (scores: list[float], new_prefix_dc: DynamicCache | None)
      new_prefix_dc contains KV for stable_prefix + extension (batch=1).
    """
    k = len(diff_seqs)
    max_diff = max(len(d) for d in diff_seqs)
    pad_id = _rm_tok.pad_token_id or 0

    _pf_on = _PROFILE_DETAIL
    _t0 = _pf_now() if _pf_on else 0.0

    # --- Phase: prep_tensor (build diff_ids + attn_mask) ---
    diff_ids = torch.full((k, max_diff), pad_id, dtype=torch.long, device=_rm_device)
    for i, d in enumerate(diff_seqs):
        diff_ids[i, :len(d)] = torch.tensor(d, dtype=torch.long)

    full_len = stable_prefix_len + max_diff
    attn_mask = torch.zeros((k, full_len), dtype=torch.long, device=_rm_device)
    attn_mask[:, :stable_prefix_len] = 1
    for i, d in enumerate(diff_seqs):
        attn_mask[i, stable_prefix_len:stable_prefix_len + len(d)] = 1

    _t1 = _pf_now() if _pf_on else 0.0

    # --- Phase: kv_expand (expand KV cache from batch=1 to batch=k) ---
    batch_dc = _dc_expand_batch(prefix_dc, k)

    if _compile_mode:
        import torch._dynamo as _dynamo
        _dynamo.mark_dynamic(diff_ids, 1)
        _dynamo.mark_dynamic(attn_mask, 1)
        for _layer in batch_dc.layers:
            _dynamo.mark_dynamic(_layer.keys, 2)
            _dynamo.mark_dynamic(_layer.values, 2)

    _t2 = _pf_now() if _pf_on else 0.0

    # --- Prefer CUDA graph bucket (HIT path fast track) ---
    global _cg_consecutive_nan, _cg_runtime_disabled
    per_req_skip = (request_id is not None and request_id in _cg_per_req_disabled)
    if (_CUDA_GRAPH_ENABLED and _hit_bucket is not None
            and not _cg_runtime_disabled
            and not per_req_skip
            and _hit_bucket.can_fit(k, max_diff, stable_prefix_len)):
        # Build real absolute position_ids: [stable_prefix_len .. stable_prefix_len+max_diff-1]
        pos_ids = torch.arange(
            stable_prefix_len, stable_prefix_len + max_diff,
            dtype=torch.long, device=_rm_device,
        ).unsqueeze(0).expand(k, -1).contiguous()

        cg_result = _hit_bucket.run(
            prefix_dc=prefix_dc,
            diff_ids=diff_ids,
            attn_mask_compact=attn_mask,
            position_ids_actual=pos_ids,
            extension_len=update_len,
        )
        # bucket returns None when NaN/Inf detected; fall through to eager path here
        if cg_result is not None:
            _cg_consecutive_nan = 0  # reset global consecutive failure count
            if request_id is not None:
                _cg_request_nan[request_id] = 0  # also reset per-request count
            logits_tensor, new_prefix_dc = cg_result
            _t3 = _pf_now() if _pf_on else 0.0
            scores = logits_tensor.flatten().tolist()
            _t4 = _pf_now() if _pf_on else 0.0

            if _pf_on:
                _pf_record("prep_tensor",   _t0, _t1)
                _pf_record("kv_expand",     _t1, _t2)
                _pf_record("forward",       _t2, _t3)
                _pf_record("score_extract", _t3, _t4)
                ms = lambda a, b: (b - a) * 1000.0
                print(
                    f"[RM-pf-fwd-cg] k={k} diff={max_diff} kv={stable_prefix_len} "
                    f"update_len={update_len} | "
                    f"prep={ms(_t0,_t1):.1f} expand={ms(_t1,_t2):.1f} "
                    f"cg_run={ms(_t2,_t3):.1f} score={ms(_t3,_t4):.1f} "
                    f"total={ms(_t0,_t4):.1f}ms",
                    flush=True,
                )
            return scores, new_prefix_dc
        # cg_result is None: bucket output has NaN
        # 1) per-request count + disable only that request if too many accumulated
        if request_id is not None:
            _cg_request_nan[request_id] = _cg_request_nan.get(request_id, 0) + 1
            if _cg_request_nan[request_id] >= _CG_PER_REQ_NAN_THRESHOLD:
                if request_id not in _cg_per_req_disabled:
                    _cg_per_req_disabled.add(request_id)
                    print(
                        f"[RM] Per-request bucket disabled for {request_id[:8]} "
                        f"({_cg_request_nan[request_id]} NaN failures); other requests "
                        f"can still use bucket.",
                        flush=True,
                    )
        # 2) Global consecutive failure count (across requests)
        _cg_consecutive_nan += 1
        if _cg_consecutive_nan >= _CG_NAN_DISABLE_THRESHOLD and not _cg_runtime_disabled:
            _cg_runtime_disabled = True
            print(
                f"[RM] CUDA graph runtime-disabled after {_cg_consecutive_nan} "
                f"consecutive NaN failures (bucket appears broken; falling back to "
                f"eager permanently)",
                flush=True,
            )

    # --- Otherwise fall through to the original eager path ---
    # Phase: forward (main forward, all 36 layers)
    global _layer_active
    _layer_active = _PROFILE_LAYERS
    try:
        with torch.inference_mode():
            out = _rm_model(
                input_ids=diff_ids,
                attention_mask=attn_mask,
                past_key_values=batch_dc,
                use_cache=True,
            )
    finally:
        _layer_active = False

    _t3 = _pf_now() if _pf_on else 0.0

    # --- Phase: score_extract (extract logits) ---
    scores = out.logits.flatten().tolist()

    _t4 = _pf_now() if _pf_on else 0.0

    # --- Phase: kv_save (extract and trim batch[0] for next step) ---
    new_prefix_dc = None
    raw_kv = getattr(out, 'past_key_values', None)
    if raw_kv is not None:
        new_len = stable_prefix_len + update_len
        kv_dc = _to_dynamic_cache(raw_kv)
        if kv_dc is not None and _dc_seq_len(kv_dc) >= new_len:
            new_prefix_dc = _dc_extract_batch0_trim(kv_dc, new_len)

    _t5 = _pf_now() if _pf_on else 0.0

    if _pf_on:
        _pf_record("prep_tensor",   _t0, _t1)
        _pf_record("kv_expand",     _t1, _t2)
        _pf_record("forward",       _t2, _t3)
        _pf_record("score_extract", _t3, _t4)
        _pf_record("kv_save",       _t4, _t5)
        ms = lambda a, b: (b - a) * 1000.0
        print(
            f"[RM-pf-fwd] k={k} diff={max_diff} kv={stable_prefix_len} "
            f"full={full_len} update_len={update_len} | "
            f"prep={ms(_t0,_t1):.1f} expand={ms(_t1,_t2):.1f} "
            f"fwd={ms(_t2,_t3):.1f} score={ms(_t3,_t4):.1f} "
            f"kvsave={ms(_t4,_t5):.1f} total={ms(_t0,_t5):.1f}ms",
            flush=True,
        )

    return scores, new_prefix_dc


def _try_score_with_kv_cache(texts: list, request_id: str) -> Optional[list]:
    """
    Try scoring with cross-step KV cache. Priority:
      HIT : 1 pass (batch=k, seq~=8)                 fastest, O(8*N), independent of N
      MISS: 2 pass (batch=1, seq=N) + (batch=k, seq~=8)  O(N/k + 8), ~5x faster than batch=k full
    Returns None → score() falls back to baseline (batch=k, seq=N full forward).
    """
    global _req_kv

    _pf_on = _PROFILE_DETAIL
    _t_enter = _pf_now() if _pf_on else 0.0

    # -- Tokenize (no padding, keep raw IDs for prefix matching) -----------------
    try:
        encoded_list = [
            _rm_tok(
                t, return_tensors="pt", truncation=True,
                max_length=2048, padding=False,
            ).input_ids[0].tolist()
            for t in texts
        ]
        _t_tok = _pf_now() if _pf_on else 0.0

        raw_prefix = _find_common_prefix(encoded_list)
        stable_len = len(raw_prefix) - _BPE_TRIM
        if stable_len < 16:
            return None
        stable_prefix = raw_prefix[:stable_len]
        bpe_tail = raw_prefix[stable_len:]
        _t_prefix = _pf_now() if _pf_on else 0.0
    except Exception as e:
        print(f"[RM] tokenize failed ({e})", flush=True)
        return None

    if _pf_on:
        _pf_record("tokenize",    _t_enter, _t_tok)
        _pf_record("prefix_calc", _t_tok,   _t_prefix)

    # -- HIT path: reuse the cached prefix KV from the previous step --------------
    try:
        _t_lookup_start = _pf_now() if _pf_on else 0.0
        state = _req_kv.get(request_id)
        if state is not None:
            cached_ids = state["prefix_ids"]
            if (len(cached_ids) <= len(stable_prefix)
                    and stable_prefix[:len(cached_ids)] == cached_ids):

                prefix_dc = state["kv"]
                cached_len = _dc_seq_len(prefix_dc)
                extension = stable_prefix[len(cached_ids):]
                diff_seqs = [
                    extension + bpe_tail + e[len(raw_prefix):]
                    for e in encoded_list
                ]
                _t_lookup_end = _pf_now() if _pf_on else 0.0
                if _pf_on:
                    _pf_record("state_lookup", _t_lookup_start, _t_lookup_end)
                    print(
                        f"[RM-pf-hit] req={str(request_id)[:8]} "
                        f"cached_ids_len={len(cached_ids)} stable_len={stable_len} "
                        f"extension_len={len(extension)} bpe_tail={len(bpe_tail)} "
                        f"per_seq_diff={[len(d) for d in diff_seqs]} | "
                        f"tokenize={(_t_tok-_t_enter)*1000:.1f} "
                        f"prefix_calc={(_t_prefix-_t_tok)*1000:.1f} "
                        f"state_lookup={(_t_lookup_end-_t_lookup_start)*1000:.1f}ms",
                        flush=True,
                    )

                if all(len(d) > 0 for d in diff_seqs):
                    scores, new_dc = _score_with_prefix_kv(
                        prefix_dc, cached_len, diff_seqs,
                        update_len=len(extension),
                        request_id=request_id,
                    )
                    if new_dc is not None:
                        _req_kv[request_id] = {
                            "prefix_ids": stable_prefix,
                            "kv": new_dc,
                        }
                    _kv_stats["hit"] += 1
                    _pf_summary_if_due()
                    return scores
    except Exception as e:
        print(f"[RM] KV HIT failed ({e}), trying MISS", flush=True)

    # -- Optimized MISS: batch=1 processes only stable_prefix, then batch=k processes only diff --
    try:
        _t_miss_start = _pf_now() if _pf_on else 0.0
        prefix_ids_t = torch.tensor(
            [stable_prefix], dtype=torch.long, device=_rm_device,
        )
        if _compile_mode:
            import torch._dynamo as _dynamo
            _dynamo.mark_dynamic(prefix_ids_t, 1)
        global _layer_active
        _layer_active = _PROFILE_LAYERS
        try:
            with torch.inference_mode():
                prefix_out = _rm_model(input_ids=prefix_ids_t, use_cache=True)
        finally:
            _layer_active = False

        _t_miss_fwd = _pf_now() if _pf_on else 0.0

        raw_kv = getattr(prefix_out, 'past_key_values', None)
        if raw_kv is None:
            raise RuntimeError("no past_key_values from batch=1 prefix forward")

        prefix_dc = _to_dynamic_cache(raw_kv)
        if prefix_dc is None or _dc_seq_len(prefix_dc) < stable_len:
            raise RuntimeError(
                f"KV length {_dc_seq_len(prefix_dc)} < stable_len {stable_len}"
            )

        diff_seqs = [bpe_tail + e[len(raw_prefix):] for e in encoded_list]
        if not all(len(d) > 0 for d in diff_seqs):
            raise RuntimeError("empty diff sequence in MISS path")

        if _pf_on:
            _pf_record("miss_prefix_fwd", _t_miss_start, _t_miss_fwd)
            print(
                f"[RM-pf-miss] req={str(request_id)[:8]} "
                f"stable_len={stable_len} bpe_tail={len(bpe_tail)} "
                f"per_seq_diff={[len(d) for d in diff_seqs]} | "
                f"tokenize={(_t_tok-_t_enter)*1000:.1f} "
                f"prefix_calc={(_t_prefix-_t_tok)*1000:.1f} "
                f"miss_prefix_fwd={(_t_miss_fwd-_t_miss_start)*1000:.1f}ms",
                flush=True,
            )

        scores, _ = _score_with_prefix_kv(
            prefix_dc, stable_len, diff_seqs, update_len=0,
            request_id=request_id,
        )

        if len(_req_kv) >= _MAX_REQ_KV:
            del _req_kv[next(iter(_req_kv))]
        _req_kv[request_id] = {"prefix_ids": stable_prefix, "kv": prefix_dc}

        _kv_stats["miss"] += 1
        _pf_summary_if_due()
        return scores
    except Exception as e:
        print(f"[RM] KV MISS failed ({e})", flush=True)
        return None


# ---------------------------------------------------------------------------
# Model loading functions
# ---------------------------------------------------------------------------

def _load_rm_base(rm_path: str, device: str):
    print(f"[RM] Loading base RM: {rm_path}  device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        rm_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def _load_rm_with_lora(rm_path: str, rm_lora_path: str, device: str):
    from peft import PeftModel

    print(f"[RM] Loading RM base for LoRA: {rm_path}", flush=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    lora_dir = os.path.join(rm_lora_path, "lora_weights")
    print(f"[RM] Loading LoRA from: {lora_dir}", flush=True)
    base_model = PeftModel.from_pretrained(base_model, lora_dir)
    base_model = base_model.merge_and_unload()

    hidden_size = base_model.config.hidden_size
    token_reward_head = nn.Linear(hidden_size, 1)
    head_path = os.path.join(rm_lora_path, "token_reward_head.pt")
    print(f"[RM] Loading token_reward_head from: {head_path}", flush=True)
    head_state = torch.load(head_path, map_location='cpu', weights_only=True)
    token_reward_head.load_state_dict(head_state['token_reward_head'])

    tok = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = _ValueModelWrapper(base_model, token_reward_head).to(device)
    return model, tok


# ---------------------------------------------------------------------------
# torch.compile helpers
# ---------------------------------------------------------------------------

def _apply_compile_and_warmup(model, tok, device: str):
    """
    Apply torch.compile to the model and run warmup forwards to trigger JIT compilation.

    torch.compile eliminates PyTorch kernel launch overhead (~36ms → ~5ms),
    making short-sequence KV cache forwards genuinely faster.
    First warmup takes approximately 20-60s; subsequent calls have no extra overhead.
    """
    import torch._dynamo
    # Use graph break instead of error on unsupported ops
    torch._dynamo.config.suppress_errors = True

    print("[RM] Applying torch.compile(dynamic=True, fullgraph=False)...", flush=True)
    compiled = torch.compile(model, dynamic=True, fullgraph=False)

    print("[RM] Running warmup pass 1/3: standard forward (may take 30-90s)...", flush=True)
    dummy_text = "Warmup " * 50   # ~50 tokens, padded to 256
    dummy_texts = [dummy_text] * 5
    enc = tok(
        dummy_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)
    with torch.inference_mode():
        out1 = compiled(**enc, use_cache=True)
        _ = compiled(**enc, use_cache=True)

    # Pass 2: KV cache HIT path — diff_len must be > _BPE_TRIM, marked as dynamic
    print("[RM] Running warmup pass 2/3: KV cache HIT forward (mark dynamic dims)...", flush=True)
    raw_kv = getattr(out1, 'past_key_values', None)
    if raw_kv is not None:
        try:
            from transformers.cache_utils import DynamicCache
            kv_dc = _to_dynamic_cache(raw_kv)
            if kv_dc is not None:
                diff_len = _BPE_TRIM + 3  # 8, greater than _BPE_TRIM=5, covers minimum actual diff
                prefix_len = _dc_seq_len(kv_dc) - diff_len
                if prefix_len > 0:
                    prefix_dc = _dc_extract_batch0_trim(kv_dc, prefix_len)
                    if prefix_dc is not None:
                        batch_dc = _dc_expand_batch(prefix_dc, 5)
                        diff_ids = torch.zeros((5, diff_len), dtype=torch.long, device=device)
                        full_len = prefix_len + diff_len
                        attn_mask = torch.ones((5, full_len), dtype=torch.long, device=device)
                        # Mark diff dimension as fully symbolic to eliminate shape guard
                        torch._dynamo.mark_dynamic(diff_ids, 1)
                        torch._dynamo.mark_dynamic(attn_mask, 1)
                        for layer in batch_dc.layers:
                            torch._dynamo.mark_dynamic(layer.keys, 2)
                            torch._dynamo.mark_dynamic(layer.values, 2)
                        with torch.inference_mode():
                            _ = compiled(
                                input_ids=diff_ids,
                                attention_mask=attn_mask,
                                past_key_values=batch_dc,
                                use_cache=True,
                            )
                            _ = compiled(
                                input_ids=diff_ids,
                                attention_mask=attn_mask,
                                past_key_values=batch_dc,
                                use_cache=True,
                            )
        except Exception as e:
            print(f"[RM] KV cache HIT warmup skipped ({e})", flush=True)

    # Pass 3: MISS path — batch=1 full forward, seq dimension marked as dynamic
    print("[RM] Running warmup pass 3/3: MISS path batch=1 forward (mark dynamic seq)...", flush=True)
    try:
        dummy_ids_1 = torch.zeros((1, 256), dtype=torch.long, device=device)
        torch._dynamo.mark_dynamic(dummy_ids_1, 1)
        with torch.inference_mode():
            _ = compiled(input_ids=dummy_ids_1, use_cache=True)
            _ = compiled(input_ids=dummy_ids_1, use_cache=True)
    except Exception as e:
        print(f"[RM] MISS path warmup skipped ({e})", flush=True)

    print("[RM] torch.compile warmup complete.", flush=True)
    return compiled


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SIA RM Server", version="0.3.0")


class ScoreRequest(BaseModel):
    user_content: str
    response_so_far: str
    candidate_texts: list[str]
    request_id: Optional[str] = None   # enables cross-step KV cache when provided


class ReloadRequest(BaseModel):
    rm: str
    rm_lora: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def get_status():
    return dict(_status)


@app.post("/score")
def score(req: ScoreRequest):
    """Externally exposed endpoint with top-level try/except to prevent the server process from being killed by exceptions."""
    try:
        return _score_impl(req)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(
            f"[RM-error] Exception in /score handler: {type(e).__name__}: {e}\n"
            f"Stack:\n{tb}",
            flush=True,
        )
        # Return NaN-safe zero scores so the LLM server at least won't hang
        n = len(req.candidate_texts) if req.candidate_texts else 5
        return {"scores": [0.0] * n, "error": f"{type(e).__name__}: {e}"}


def _score_impl(req: ScoreRequest):
    t0 = time.time()
    _t_lock_req = time.perf_counter() if _PROFILE_DETAIL else 0.0
    with _rm_lock:
        _t_lock_acquired = time.perf_counter() if _PROFILE_DETAIL else 0.0
        bos = _rm_tok.bos_token
        texts = []
        for cand_text in req.candidate_texts:
            response_with_cand = req.response_so_far + cand_text
            convs = [
                {"role": "user",      "content": req.user_content},
                {"role": "assistant", "content": response_with_cand},
            ]
            rm_text = _rm_tok.apply_chat_template(convs, tokenize=False)
            if bos and rm_text.startswith(bos):
                rm_text = rm_text[len(bos):]
            texts.append(rm_text)

        t_tok = time.time()
        _t_chat_template = time.perf_counter() if _PROFILE_DETAIL else 0.0

        path = "baseline"
        if req.request_id is not None:
            _hit_before = _kv_stats["hit"]
            scores = _try_score_with_kv_cache(texts, req.request_id)
            if scores is not None:
                path = "kv_hit" if _kv_stats["hit"] > _hit_before else "kv_miss"
            else:
                scores = None
        else:
            scores = None

        if scores is None:
            # True fallback: standard batch forward without use_cache
            _t_fb_start = _pf_now() if _PROFILE_DETAIL else 0.0
            encoded = _rm_tok(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(_rm_device)
            t_enc = time.time()
            _t_fb_tok = _pf_now() if _PROFILE_DETAIL else 0.0
            if _compile_mode:
                import torch._dynamo as _dynamo
                _dynamo.mark_dynamic(encoded["input_ids"], 1)
                if "attention_mask" in encoded:
                    _dynamo.mark_dynamic(encoded["attention_mask"], 1)
            global _layer_active
            _layer_active = _PROFILE_LAYERS
            try:
                with torch.inference_mode():
                    rm_out = _rm_model(**encoded)
            finally:
                _layer_active = False
            _t_fb_fwd = _pf_now() if _PROFILE_DETAIL else 0.0
            t_fwd = time.time()
            scores = rm_out.logits.flatten().tolist()
            seq_len = encoded["input_ids"].shape[1]
            path = "fallback"
            if _PROFILE_DETAIL:
                print(
                    f"[RM-pf-fallback] batch={len(texts)} seq={seq_len} | "
                    f"tokenize={(_t_fb_tok-_t_fb_start)*1000:.1f} "
                    f"fwd={(_t_fb_fwd-_t_fb_tok)*1000:.1f}ms",
                    flush=True,
                )
        else:
            t_enc = t_tok
            t_fwd = time.time()
            seq_len = -1

    total = int((t_fwd - t0) * 1000)
    fwd = int((t_fwd - t_enc) * 1000)
    hit = _kv_stats["hit"]
    miss = _kv_stats["miss"]
    hit_rate = hit / (hit + miss) if (hit + miss) > 0 else 0.0
    if _PROFILE_DETAIL:
        lock_wait = (_t_lock_acquired - _t_lock_req) * 1000.0
        chat_tpl = (_t_chat_template - _t_lock_acquired) * 1000.0
        print(
            f"[RM] score: path={path}  fwd={fwd}ms  total={total}ms"
            f"  seq={seq_len}  hit_rate={hit_rate:.1%}({hit}/{hit+miss})"
            f"  lock_wait={lock_wait:.1f}ms chat_tpl={chat_tpl:.1f}ms",
            flush=True,
        )
    else:
        print(
            f"[RM] score: path={path}  fwd={fwd}ms  total={total}ms"
            f"  seq={seq_len}  hit_rate={hit_rate:.1%}({hit}/{hit+miss})",
            flush=True,
        )
    return {"scores": scores}


@app.post("/reload")
def reload_rm(req: ReloadRequest):
    """Hot-swap the RM (synchronous, returns only after loading completes)."""
    global _rm_model, _rm_tok, _req_kv, _hit_bucket, _CUDA_GRAPH_ENABLED
    with _rm_lock:
        _status["status"] = "reloading"
        print(f"[RM] Reloading: rm={req.rm}  rm_lora={req.rm_lora}", flush=True)

        # CUDA graph is incompatible with the new model (pointers/shapes changed), disable after reload
        if _hit_bucket is not None:
            print("[RM] Disabling CUDA graph bucket due to model reload "
                  "(restart server with --cuda_graph to re-capture).", flush=True)
            _hit_bucket = None
            _CUDA_GRAPH_ENABLED = False

        del _rm_model
        torch.cuda.empty_cache()
        _req_kv.clear()

        if req.rm_lora:
            _rm_model, _rm_tok = _load_rm_with_lora(req.rm, req.rm_lora, _rm_device)
        else:
            _rm_model, _rm_tok = _load_rm_base(req.rm, _rm_device)
        _rm_model.eval()
        _install_layer_hooks(_rm_model)
        if _compile_mode:
            _rm_model = _apply_compile_and_warmup(_rm_model, _rm_tok, _rm_device)

        _status["status"] = "ready"
        _status["rm"] = req.rm
        _status["rm_lora"] = req.rm_lora
        print("[RM] Reload complete.", flush=True)

    return dict(_status)


# ---------------------------------------------------------------------------
# CLI & startup
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA RM Server")
    p.add_argument("--rm",        required=True,  help="RM base model path")
    p.add_argument("--rm_lora",   default=None,   help="RM LoRA checkpoint path")
    p.add_argument("--rm_device", default="cuda:0")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8001)
    p.add_argument("--compile",   action="store_true",
                   help="Enable torch.compile to accelerate inference (eliminates kernel launch overhead, recommended with KV cache)")
    p.add_argument("--cuda_graph", action="store_true",
                   help="Enable CUDA graph + static bucketing to accelerate HIT path (mutually exclusive with --compile)")
    p.add_argument("--cg_batch",    type=int, default=5,   help="CUDA graph bucket batch (default matches --topk)")
    p.add_argument("--cg_diff_max", type=int, default=64,  help="CUDA graph bucket diff_max upper bound")
    p.add_argument("--cg_kv_max",   type=int, default=2048, help="CUDA graph bucket kv_max upper bound")
    args = p.parse_args()
    if args.compile and args.cuda_graph:
        p.error("--compile and --cuda_graph are mutually exclusive, choose one")
    return args


def main():
    global _rm_model, _rm_tok, _rm_device, _compile_mode, _CUDA_GRAPH_ENABLED
    args = parse_args()
    _rm_device = args.rm_device
    _compile_mode = args.compile
    _CUDA_GRAPH_ENABLED = args.cuda_graph

    print("=" * 60)
    print(f"RM       : {args.rm}  device={args.rm_device}")
    if args.rm_lora:
        print(f"RM LoRA  : {args.rm_lora}")
    print(f"Compile  : {args.compile}")
    print(f"CUDA graph: {args.cuda_graph}"
          + (f"  bucket=(batch={args.cg_batch}, diff_max={args.cg_diff_max}, kv_max={args.cg_kv_max})"
             if args.cuda_graph else ""))
    print(f"Profile  : RM_PROFILE={_PROFILE_DETAIL}  RM_PROFILE_LAYERS={_PROFILE_LAYERS}")
    print(f"Server   : http://{args.host}:{args.port}")
    print("=" * 60)

    if args.rm_lora:
        _rm_model, _rm_tok = _load_rm_with_lora(args.rm, args.rm_lora, args.rm_device)
    else:
        _rm_model, _rm_tok = _load_rm_base(args.rm, args.rm_device)
    _rm_model.eval()
    _install_layer_hooks(_rm_model)
    if args.compile:
        _rm_model = _apply_compile_and_warmup(_rm_model, _rm_tok, args.rm_device)
    if args.cuda_graph:
        ok = _init_cuda_graph_buckets(
            _rm_model, args.rm_device,
            batch=args.cg_batch, diff_max=args.cg_diff_max, kv_max=args.cg_kv_max,
        )
        if not ok:
            _CUDA_GRAPH_ENABLED = False
            print("[RM] CUDA graph disabled (init failed); using eager.", flush=True)

    _status["status"] = "ready"
    _status["rm"] = args.rm
    _status["rm_lora"] = args.rm_lora
    print("[RM] Ready.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

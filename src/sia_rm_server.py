"""
SIA RM Server — 独立的 Reward Model 打分服务

将 RM（Value Model + 可选 LoRA）部署为独立 FastAPI 服务。
主 LLM server（sia_vllm_server.py）通过 HTTP 调用本服务打分，
热切换 RM 只需调用 /reload，无需重启 vLLM。

Endpoints:
  POST /score    — 批量对 topk 候选 token 打分（per-token 调用）
  POST /reload   — 热切换 RM（同步，完成后返回）
  GET  /status   — 查询当前加载状态
  GET  /health   — 健康检查

Usage:
  # 不带 LoRA
  python sia_rm_server.py \\
    --rm /path/to/rm --rm_device cuda:0 --port 8001

  # 带 LoRA
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
# ValueModel 包装（与原 sia_vllm_RM.py 逻辑一致）
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

        # 新版 HF backbone 要求 Cache 对象，不接受 legacy tuple
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

        # CUDA graph 要求 arange 在 device 上创建，避免 CPU→GPU 拷贝触发
        # "operation not permitted when stream is capturing"
        _batch_idx = torch.arange(
            token_rewards.size(0), device=token_rewards.device,
        )
        if past_key_values is not None:
            # KV cache 模式：找 diff 中最后一个有效 token
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
# 全局状态
# ---------------------------------------------------------------------------
_rm_model = None
_rm_tok = None
_rm_device: str = "cuda:0"
_rm_lock = threading.Lock()
_status = {"status": "initializing", "rm": "", "rm_lora": None}

# 跨步 KV 状态：request_id → {"prefix_ids": list[int], "kv": DynamicCache}
_req_kv: dict = {}
_MAX_REQ_KV = 64

# 修剪末尾 token 数，消除 BPE 边界影响
_BPE_TRIM = 5

# 统计 hit/miss（仅用于日志）
_kv_stats = {"hit": 0, "miss": 0}

# 是否启用 torch.compile（CLI --compile 时置 True）
_compile_mode: bool = False

# ---------------------------------------------------------------------------
# Profiling 开关与统计
# ---------------------------------------------------------------------------
# 环境变量 RM_PROFILE=0 关闭详细 timing；默认 1（开启）
_PROFILE_DETAIL: bool = os.environ.get("RM_PROFILE", "1") == "1"

# 每个阶段的耗时（ms），用于 rolling stats
_pf_phases = [
    "tokenize", "prefix_calc", "state_lookup",
    "prep_tensor", "kv_expand", "forward", "score_extract", "kv_save",
    "miss_prefix_fwd",
]
_pf_stats: dict = {p: [] for p in _pf_phases}
_pf_call_count: int = 0
_PF_STATS_INTERVAL = 50  # 每 N 次调用输出一次 p50/p95/max


def _pf_now():
    """同步 GPU 后取高精度 CPU 时间。"""
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
    """每 _PF_STATS_INTERVAL 次调用打印一次 p50/p95/max 汇总。"""
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
    # 每 N 次也输出一次 per-layer 汇总（若启用）
    _pf_layer_summary()


# === Per-layer profiling（可选，环境变量 RM_PROFILE_LAYERS=1 启用） ===
_PROFILE_LAYERS: bool = os.environ.get("RM_PROFILE_LAYERS", "0") == "1"
_layer_times: dict = {}      # layer_idx → [ms, ms, ...]
_layer_count: int = 0
_layer_active: bool = False  # forward 中由 hook 临时设置
_layer_t_prev: float = 0.0   # 上一层 hook 时间戳


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
    """对 transformer 每层 decoder layer 安装 pre/post hook，需要 RM_PROFILE_LAYERS=1。"""
    if not _PROFILE_LAYERS:
        return
    # 找到 base transformer（Qwen3ForSequenceClassification 的 .model.layers）
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
    # 取近 50 次每层平均时间，打印
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
# DynamicCache 辅助
# ---------------------------------------------------------------------------

def _to_dynamic_cache(past_kv):
    """将任意格式 past_key_values 转为 DynamicCache。"""
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
    """获取 DynamicCache 已缓存的序列长度。"""
    if dc is None:
        return 0
    if hasattr(dc, 'get_seq_length'):
        return dc.get_seq_length()
    if hasattr(dc, 'key_cache') and dc.key_cache:
        return dc.key_cache[0].shape[-2]
    return 0


def _dc_expand_batch(dc, k: int):
    """将 batch=1 的 DynamicCache 扩展到 batch=k（.contiguous() 物化）。
    兼容 transformers 4.57+ 新 API（layers[i].keys/values + update()）。
    """
    from transformers.cache_utils import DynamicCache
    new_dc = DynamicCache()
    n_layers = 0
    for i, layer in enumerate(dc.layers):
        K = layer.keys.expand(k, -1, -1, -1).contiguous()
        V = layer.values.expand(k, -1, -1, -1).contiguous()
        new_dc.update(K, V, i)
        n_layers += 1
    # 在 _PROFILE_DETAIL 时记录形状供单行输出引用
    if _PROFILE_DETAIL:
        new_dc._pf_n_layers = n_layers  # type: ignore
    return new_dc


def _dc_extract_batch0_trim(dc, keep_len: int):
    """
    从 batch>=1 的 DynamicCache 中取 batch[0]，并只保留前 keep_len 个位置。
    前缀 token 的 KV 对所有 batch 相同（causal attention 中 pos<keep_len 不依赖后续 token）。
    兼容 transformers 4.57+ 新 API。
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
# CUDA graph 静态形状 bucketing
# ---------------------------------------------------------------------------

# 全局：单个 HIT bucket（覆盖最常见的 batch=topk, diff<=64, kv<=2048 场景）
_CUDA_GRAPH_ENABLED: bool = False
_hit_bucket = None  # _CudaGraphHitBucket 实例

# Adaptive disable（per-request）：某个 request 一旦在 bucket 里 NaN 太多，
# 就只关掉那个 request 的 bucket，其他 request 仍然享受加速。
# 全局 disable 在最坏情况下也会触发（防止 catastrophic 失败）。
_cg_request_nan: dict = {}        # request_id → 连续 NaN 次数
_CG_PER_REQ_NAN_THRESHOLD: int = 5  # 同一个 request 连续 5 次 NaN 就降级
_cg_per_req_disabled: set = set()  # 被降级的 request_id 集合

# 全局 safety net
_cg_consecutive_nan: int = 0
_CG_NAN_DISABLE_THRESHOLD: int = 50  # 整个 server 累计 50 次（跨 request）连续 NaN 才全局关
_cg_runtime_disabled: bool = False

# 模型架构信息，bucket 初始化用
_mc_num_layers: int = 0
_mc_num_kv_heads: int = 0
_mc_head_dim: int = 0
_mc_dtype = None


class _CudaGraphHitBucket:
    """
    单个静态形状的 CUDA graph bucket，用于 HIT 路径 forward。

    覆盖输入：
      - input_ids:      (batch, diff_max)
      - attention_mask: (batch, kv_max + diff_max)
      - past_key_values: DynamicCache with batch=batch, kv_max positions per layer
      - position_ids:   (batch, diff_max)   — 显式传入，因为 buffer 中 diff 起始位置
                                              是 kv_max（不是真实 kv_actual）

    Buffer 布局（实际 kv_actual <= kv_max, diff_actual <= diff_max）：
      KV buffer [0..kv_max-1]:
        [0..kv_actual-1]  ← 真实 prefix KV（输入）
        [kv_actual..kv_max-1] ← padding（垃圾，被 attention_mask 屏蔽）
      input_ids [0..diff_max-1]:
        [0..diff_actual-1] ← 真实 diff tokens
        [diff_actual..diff_max-1] ← padding（被 attention_mask 屏蔽）
      attention_mask [0..kv_max+diff_max-1]:
        [0..kv_actual-1]                   = real prefix mask
        [kv_actual..kv_max-1]              = 0 (mask out padded prefix)
        [kv_max..kv_max+diff_actual-1]     = real diff mask
        [kv_max+diff_actual..]             = 0
      position_ids [0..diff_max-1]:
        [i] = kv_actual + i  for i in 0..diff_actual-1 (RoPE 用真实绝对位置)
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

        # 输入 buffers（CPU→GPU 通道，每次 replay 前由 run() 写入）
        self.input_ids = torch.zeros(batch, diff_max, dtype=torch.long, device=device)
        self.attention_mask = torch.zeros(batch, self.full, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(batch, diff_max, dtype=torch.long, device=device)

        # Prefix KV buffers：每层 (batch, num_kv_heads, kv_max, head_dim)
        # 由 DynamicCache 持有引用；model forward 时 attention 从这里读
        self.prefix_K = [
            torch.zeros(batch, num_kv_heads, kv_max, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.prefix_V = [
            torch.zeros(batch, num_kv_heads, kv_max, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # 由 capture 设置：graph + 输出张量引用
        self.graph = None
        self.output_logits = None   # (batch, 1)
        self.full_K = None          # list of (batch, h, full, d) — post-forward K
        self.full_V = None

    def _new_dc(self):
        """构造一个 DynamicCache，layers 指向 self.prefix_K / prefix_V。"""
        from transformers.cache_utils import DynamicCache
        dc = DynamicCache()
        for i in range(self.num_layers):
            dc.update(self.prefix_K[i], self.prefix_V[i], i)
        return dc

    def capture(self, n_warmup: int = 3):
        """运行 warmup forward 后 capture CUDA graph。"""
        print(
            f"[RM] CudaGraph warmup+capture: batch={self.batch} "
            f"diff_max={self.diff_max} kv_max={self.kv_max} "
            f"layers={self.num_layers} kv_heads={self.num_kv_heads} head_dim={self.head_dim}",
            flush=True,
        )
        # 初始化 buffer 为合法的非全零值
        self.input_ids.fill_(1)
        # attention_mask: 用真实推理时的稀疏布局（部分 0），确保 SDPA dispatch
        # 选支持 mask 的 backend（mem_efficient/math），而不是被全 1 mask 误导成 Flash
        # （Flash 不支持自定义 mask 值，会把所有 key 当成有效）。
        self.attention_mask.zero_()
        cap_kv_actual = self.kv_max // 2          # 假装一半是真 prefix
        cap_diff_actual = self.diff_max // 2      # 假装一半是真 diff
        self.attention_mask[:, :cap_kv_actual] = 1
        self.attention_mask[:, self.kv_max:self.kv_max + cap_diff_actual] = 1
        pos_diff = torch.arange(
            self.kv_max, self.kv_max + self.diff_max,
            dtype=torch.long, device=self.device,
        )
        self.position_ids.copy_(pos_diff.unsqueeze(0).expand(self.batch, -1))

        # Warmup runs（不 capture，让 cuDNN / kernel 选择稳定）
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
        # 保存对捕获时分配的输出张量的引用（fixed addresses）
        self.output_logits = out.logits  # (batch, 1)
        self.full_K = [layer.keys for layer in capture_dc.layers]
        self.full_V = [layer.values for layer in capture_dc.layers]
        # 保留 capture_dc 引用防止张量被释放
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
        用 CUDA graph 跑一次 HIT forward。

        Args:
          prefix_dc: DynamicCache，batch=1，kv_actual positions
          diff_ids: (batch, diff_actual)
          attn_mask_compact: (batch, kv_actual + diff_actual) — 原始 compact 布局，
                              本方法会重排到 bucket 的稀疏布局
          position_ids_actual: (batch, diff_actual) — 真实绝对位置 [kv_actual..]
          extension_len: diff 序列前 N 个 token 属于"extension"，
                          用来构造新的 prefix（next call 复用）

        Returns:
          logits: (batch, 1)
          new_prefix_dc: DynamicCache batch=1，kv_actual + extension_len positions

        失败（含 NaN、CUDA 错误等）时返回 None，由调用方回退到 eager。
        """
        try:
            return self._run_unsafe(prefix_dc, diff_ids, attn_mask_compact,
                                     position_ids_actual, extension_len)
        except Exception as e:
            # 任何异常（含 CUDA 异步错误延迟报告）都不允许 server 挂掉
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

        # 0) 输入 prefix_dc 完整性检查（debug：定位 NaN 来源） — 扫描所有层
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

        # 1) 写入 input_ids（padding 部分清零）
        self.input_ids.zero_()
        self.input_ids[:, :diff_actual].copy_(diff_ids)

        # 2) 写入 position_ids（padding 部分写为合法的连续位置）
        self.position_ids[:, :diff_actual].copy_(position_ids_actual)
        if diff_actual < self.diff_max:
            # 填充值随便，因为对应 attention_mask=0；用连续位置避免 RoPE 出错
            pad_pos = torch.arange(
                kv_actual + diff_actual, kv_actual + self.diff_max,
                dtype=torch.long, device=self.device,
            )
            self.position_ids[:, diff_actual:].copy_(pad_pos.unsqueeze(0).expand(self.batch, -1))

        # 3) 重排 attention_mask 到 bucket 稀疏布局
        #    [0:kv_actual] = real prefix mask（compact 输入的 [0:kv_actual]）
        #    [kv_actual:kv_max] = 0（padded prefix，被屏蔽）
        #    [kv_max:kv_max+diff_actual] = real diff mask
        #    [kv_max+diff_actual:] = 0
        self.attention_mask.zero_()
        self.attention_mask[:, :kv_actual].copy_(attn_mask_compact[:, :kv_actual])
        self.attention_mask[:, self.kv_max:self.kv_max + diff_actual].copy_(
            attn_mask_compact[:, kv_actual:kv_actual + diff_actual]
        )

        # 4) 把 prefix KV 拷到 buffer 的前 kv_actual 位置（batch=1 → batch=k 展开）
        #    关键：必须显式清零 padded 部分 [kv_actual:kv_max]，因为：
        #    - 这些位置可能保留上一次调用的真实 K/V 值（不是全零）
        #    - mem_efficient attention 的 max-shifted softmax 在 stale K 值
        #      上做 Q@K 可能产生数值不稳定（虽然理论上 mask=0 会屏蔽，
        #      但实测发现 stale 数据会污染整个 bucket 输出 → NaN）
        for i in range(self.num_layers):
            # 先清零整个 buffer，再写真实 prefix
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

        # 6) 取出 logits（clone 避免下次 replay 覆盖）
        logits = self.output_logits.clone()

        # 6.5) NaN/Inf safety check — 偶发 NaN 时返回 None 触发 eager fallback
        #      （根因待查；通常发生在 long-prefix MISS 之后，会污染所有后续 HIT）
        if not torch.isfinite(logits).all():
            nan_cnt = int((~torch.isfinite(logits)).sum().item())
            print(
                f"[RM-cg-warn] NaN/Inf in bucket output: nan_cnt={nan_cnt} "
                f"batch={batch} diff_actual={diff_actual} kv_actual={kv_actual} "
                f"extension_len={extension_len} → fallback to eager",
                flush=True,
            )
            return None

        # 7) 构造 new_prefix_dc（batch=1，kv_actual + extension_len positions）
        #    数据来源：
        #      [0:kv_actual]: full_K[0:1, :, 0:kv_actual, :] (原始 prefix，未变)
        #      [kv_actual:kv_actual+ext]: full_K[0:1, :, kv_max:kv_max+ext, :] (新算的 extension)
        new_prefix_len = kv_actual + extension_len
        from transformers.cache_utils import DynamicCache
        new_dc = DynamicCache()
        # 7a) 同时检查 extension 部分（new K/V）是否有 NaN：
        #     logits 可能因为 mask 偶然没受 NaN K/V 污染就返回干净的，
        #     但保存进 _req_kv 后下次 HIT 拿来用 → 一定 NaN。
        #     所以这里 stack 后扫一遍。
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
    """初始化 HIT bucket。失败时返回 False（调用方应回退到 eager）。

    关键技巧：SDPA backend 选择会被烧到 CUDA graph 里（replay 时不重新 dispatch）。
    所以我们临时切换到 mem_efficient 后端做 capture，capture 完成后恢复原状（Flash），
    让 eager 路径继续用 Flash（快），bucket replay 用 mem_efficient（正确处理 mask）。
    """
    global _hit_bucket, _mc_num_layers, _mc_num_kv_heads, _mc_head_dim, _mc_dtype

    # 保存原始 SDPA flags
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

    # 临时切换到 mem_efficient 后端（用于 capture）
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
        # 恢复原始 SDPA flags：让 eager 路径继续用 Flash（快）。
        # CUDA graph 已经把 capture 时的 mem_eff backend 烧到 graph 里了，
        # replay 时不会重新 dispatch，所以这个恢复不会影响 bucket。
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
# 核心：单次 forward 的 KV cache 打分
# ---------------------------------------------------------------------------

def _score_with_prefix_kv(
    prefix_dc,          # DynamicCache，batch=1，seq=stable_prefix_len
    stable_prefix_len: int,
    diff_seqs: list,    # list[list[int]]，每个候选的 diff token IDs（含 extension + cand）
    update_len: int,    # diff_seqs 中属于"extension"的前缀长度（用于更新 cache）
    request_id: Optional[str] = None,  # per-request bucket disable 判定用
) -> tuple:
    """
    单次 forward：batch=k, seq=max(diff_len), 使用 prefix_dc 作为 KV cache。
    返回 (scores: list[float], new_prefix_dc: DynamicCache | None)
      new_prefix_dc 包含 stable_prefix + extension 的 KV（batch=1）。
    """
    k = len(diff_seqs)
    max_diff = max(len(d) for d in diff_seqs)
    pad_id = _rm_tok.pad_token_id or 0

    _pf_on = _PROFILE_DETAIL
    _t0 = _pf_now() if _pf_on else 0.0

    # --- Phase: prep_tensor （构造 diff_ids + attn_mask） ---
    diff_ids = torch.full((k, max_diff), pad_id, dtype=torch.long, device=_rm_device)
    for i, d in enumerate(diff_seqs):
        diff_ids[i, :len(d)] = torch.tensor(d, dtype=torch.long)

    full_len = stable_prefix_len + max_diff
    attn_mask = torch.zeros((k, full_len), dtype=torch.long, device=_rm_device)
    attn_mask[:, :stable_prefix_len] = 1
    for i, d in enumerate(diff_seqs):
        attn_mask[i, stable_prefix_len:stable_prefix_len + len(d)] = 1

    _t1 = _pf_now() if _pf_on else 0.0

    # --- Phase: kv_expand （batch=1 → batch=k 展开 KV cache） ---
    batch_dc = _dc_expand_batch(prefix_dc, k)

    if _compile_mode:
        import torch._dynamo as _dynamo
        _dynamo.mark_dynamic(diff_ids, 1)
        _dynamo.mark_dynamic(attn_mask, 1)
        for _layer in batch_dc.layers:
            _dynamo.mark_dynamic(_layer.keys, 2)
            _dynamo.mark_dynamic(_layer.values, 2)

    _t2 = _pf_now() if _pf_on else 0.0

    # --- 优先尝试 CUDA graph bucket（HIT path 快路径） ---
    global _cg_consecutive_nan, _cg_runtime_disabled
    per_req_skip = (request_id is not None and request_id in _cg_per_req_disabled)
    if (_CUDA_GRAPH_ENABLED and _hit_bucket is not None
            and not _cg_runtime_disabled
            and not per_req_skip
            and _hit_bucket.can_fit(k, max_diff, stable_prefix_len)):
        # 构造真实绝对 position_ids: [stable_prefix_len .. stable_prefix_len+max_diff-1]
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
        # bucket 检测到 NaN/Inf 时返回 None，此处穿透到 eager 路径
        if cg_result is not None:
            _cg_consecutive_nan = 0  # 全局连续失败计数清零
            if request_id is not None:
                _cg_request_nan[request_id] = 0  # per-request 也清零
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
        # cg_result is None：bucket 输出有 NaN
        # 1) per-request 计数 + 该 request 累计太多就只关它
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
        # 2) 全局连续失败计数（跨 request）
        _cg_consecutive_nan += 1
        if _cg_consecutive_nan >= _CG_NAN_DISABLE_THRESHOLD and not _cg_runtime_disabled:
            _cg_runtime_disabled = True
            print(
                f"[RM] CUDA graph runtime-disabled after {_cg_consecutive_nan} "
                f"consecutive NaN failures (bucket appears broken; falling back to "
                f"eager permanently)",
                flush=True,
            )

    # --- 否则走原 eager 路径 ---
    # Phase: forward （主 forward，含全部 36 层）
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

    # --- Phase: score_extract （取出 logits） ---
    scores = out.logits.flatten().tolist()

    _t4 = _pf_now() if _pf_on else 0.0

    # --- Phase: kv_save （提取并 trim batch[0] 用于下一步） ---
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
    尝试用跨步 KV cache 打分。优先级：
      HIT ：1 pass (batch=k, seq≈8)                 最快，O(8×N)，与 N 无关
      MISS：2 pass (batch=1, seq=N) + (batch=k, seq≈8)  O(N/k + 8)，约 5× 快于 batch=k full
    返回 None → score() 降级到 baseline (batch=k, seq=N 完整 forward)。
    """
    global _req_kv

    _pf_on = _PROFILE_DETAIL
    _t_enter = _pf_now() if _pf_on else 0.0

    # ── Tokenize（不 padding，保持原始 IDs 用于前缀匹配） ──────────────────────
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

    # ── HIT path：复用上一步缓存的 prefix KV ──────────────────────────────────
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

    # ── Optimized MISS：batch=1 只处理 stable_prefix，再 batch=k 只处理 diff ────
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
# 模型加载函数
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
# torch.compile 辅助
# ---------------------------------------------------------------------------

def _apply_compile_and_warmup(model, tok, device: str):
    """
    对 model 应用 torch.compile 并执行 warmup forward，触发 JIT 编译。

    torch.compile 消除 PyTorch kernel launch overhead（~36ms → ~5ms），
    使 KV cache 的 short-sequence forward 真正受益。
    首次 warmup 约需 20-60s，后续调用无额外开销。
    """
    import torch._dynamo
    # 遇到不支持的 op 时使用 graph break 而非报错
    torch._dynamo.config.suppress_errors = True

    print("[RM] Applying torch.compile(dynamic=True, fullgraph=False)...", flush=True)
    compiled = torch.compile(model, dynamic=True, fullgraph=False)

    print("[RM] Running warmup pass 1/3: standard forward (may take 30-90s)...", flush=True)
    dummy_text = "Warmup " * 50   # ~50 tokens, pad 到 256
    dummy_texts = [dummy_text] * 5
    enc = tok(
        dummy_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)
    with torch.inference_mode():
        out1 = compiled(**enc, use_cache=True)
        _ = compiled(**enc, use_cache=True)

    # Pass 2: KV cache HIT 路径 — diff_len 必须 > _BPE_TRIM，且标记为 dynamic
    print("[RM] Running warmup pass 2/3: KV cache HIT forward (mark dynamic dims)...", flush=True)
    raw_kv = getattr(out1, 'past_key_values', None)
    if raw_kv is not None:
        try:
            from transformers.cache_utils import DynamicCache
            kv_dc = _to_dynamic_cache(raw_kv)
            if kv_dc is not None:
                diff_len = _BPE_TRIM + 3  # 8, 大于 _BPE_TRIM=5，覆盖实际最小 diff
                prefix_len = _dc_seq_len(kv_dc) - diff_len
                if prefix_len > 0:
                    prefix_dc = _dc_extract_batch0_trim(kv_dc, prefix_len)
                    if prefix_dc is not None:
                        batch_dc = _dc_expand_batch(prefix_dc, 5)
                        diff_ids = torch.zeros((5, diff_len), dtype=torch.long, device=device)
                        full_len = prefix_len + diff_len
                        attn_mask = torch.ones((5, full_len), dtype=torch.long, device=device)
                        # 标记 diff 维度为 fully symbolic，消除 shape guard
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

    # Pass 3: MISS 路径 — batch=1 full forward，seq 维度标记为 dynamic
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
    request_id: Optional[str] = None   # 提供时启用跨步 KV cache


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
    """对外暴露的 endpoint，加 top-level try/except 防止 server 进程被异常杀掉。"""
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
        # 把 NaN-safe 的 0 分数返回出去，至少让 LLM server 不会卡住
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
            # 真正的 fallback：不带 use_cache 的标准 batch forward
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
    """热切换 RM（同步，加载完成后才返回）。"""
    global _rm_model, _rm_tok, _req_kv, _hit_bucket, _CUDA_GRAPH_ENABLED
    with _rm_lock:
        _status["status"] = "reloading"
        print(f"[RM] Reloading: rm={req.rm}  rm_lora={req.rm_lora}", flush=True)

        # CUDA graph 与新模型不兼容（指针/形状变了），reload 后禁用
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
# CLI & 启动
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA RM Server")
    p.add_argument("--rm",        required=True,  help="RM 基础模型路径")
    p.add_argument("--rm_lora",   default=None,   help="RM LoRA checkpoint 路径")
    p.add_argument("--rm_device", default="cuda:0")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8001)
    p.add_argument("--compile",   action="store_true",
                   help="启用 torch.compile 加速推理（消除 kernel launch overhead，建议与 KV cache 一起使用）")
    p.add_argument("--cuda_graph", action="store_true",
                   help="启用 CUDA graph + 静态 bucketing 加速 HIT 路径（与 --compile 互斥）")
    p.add_argument("--cg_batch",    type=int, default=5,   help="CUDA graph bucket batch (默认匹配 --topk)")
    p.add_argument("--cg_diff_max", type=int, default=64,  help="CUDA graph bucket diff_max 上限")
    p.add_argument("--cg_kv_max",   type=int, default=2048, help="CUDA graph bucket kv_max 上限")
    args = p.parse_args()
    if args.compile and args.cuda_graph:
        p.error("--compile 与 --cuda_graph 互斥，二选一即可")
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

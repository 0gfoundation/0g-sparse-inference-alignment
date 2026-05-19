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
                past_key_values=None, use_cache=False, **kwargs):
        if hasattr(self.base_model, 'model'):
            backbone = self.base_model.model
        elif hasattr(self.base_model, 'transformer'):
            backbone = self.base_model.transformer
        else:
            backbone = self.base_model

        # 新版 HF backbone 要求 Cache 对象，不接受 legacy tuple
        kv_in = _to_dynamic_cache(past_key_values)
        prefix_kv_len = _dc_seq_len(kv_in)

        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_in,
            use_cache=use_cache,
            output_hidden_states=False,
        )

        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out.last_hidden_state

        token_rewards = self.token_reward_head(hidden.float()).squeeze(-1)  # (batch, seq)

        if past_key_values is not None:
            # KV cache 模式：找 diff 中最后一个有效 token
            if attention_mask is not None:
                diff_mask = attention_mask[:, prefix_kv_len:]
                diff_lens = diff_mask.sum(dim=1) - 1
                logits = token_rewards[
                    torch.arange(token_rewards.size(0)), diff_lens
                ].unsqueeze(-1)
            else:
                logits = token_rewards[:, -1].unsqueeze(-1)
        elif attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            logits = token_rewards[
                torch.arange(token_rewards.size(0)), seq_lens
            ].unsqueeze(-1)
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
    """将 batch=1 的 DynamicCache 扩展到 batch=k（.contiguous() 物化）。"""
    from transformers.cache_utils import DynamicCache
    new_dc = DynamicCache()
    for layer_idx in range(len(dc.key_cache)):
        K = dc.key_cache[layer_idx].expand(k, -1, -1, -1).contiguous()
        V = dc.value_cache[layer_idx].expand(k, -1, -1, -1).contiguous()
        new_dc.key_cache.append(K)
        new_dc.value_cache.append(V)
    return new_dc


def _dc_extract_batch0_trim(dc, keep_len: int):
    """
    从 batch>=1 的 DynamicCache 中取 batch[0]，并只保留前 keep_len 个位置。
    前缀 token 的 KV 对所有 batch 相同（causal attention 中 pos<keep_len 不依赖后续 token），
    因此取 batch[0] 是安全的。
    """
    try:
        from transformers.cache_utils import DynamicCache
        new_dc = DynamicCache()
        for layer_idx in range(len(dc.key_cache)):
            K = dc.key_cache[layer_idx][0:1, :, :keep_len, :].contiguous()
            V = dc.value_cache[layer_idx][0:1, :, :keep_len, :].contiguous()
            new_dc.key_cache.append(K)
            new_dc.value_cache.append(V)
        return new_dc
    except Exception:
        return None


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
) -> tuple:
    """
    单次 forward：batch=k, seq=max(diff_len), 使用 prefix_dc 作为 KV cache。
    返回 (scores: list[float], new_prefix_dc: DynamicCache | None)
      new_prefix_dc 包含 stable_prefix + extension 的 KV（batch=1）。
    """
    k = len(diff_seqs)
    max_diff = max(len(d) for d in diff_seqs)
    pad_id = _rm_tok.pad_token_id or 0

    diff_ids = torch.full((k, max_diff), pad_id, dtype=torch.long, device=_rm_device)
    for i, d in enumerate(diff_seqs):
        diff_ids[i, :len(d)] = torch.tensor(d, dtype=torch.long)

    full_len = stable_prefix_len + max_diff
    attn_mask = torch.zeros((k, full_len), dtype=torch.long, device=_rm_device)
    attn_mask[:, :stable_prefix_len] = 1
    for i, d in enumerate(diff_seqs):
        attn_mask[i, stable_prefix_len:stable_prefix_len + len(d)] = 1

    batch_dc = _dc_expand_batch(prefix_dc, k)

    with torch.no_grad():
        out = _rm_model(
            input_ids=diff_ids,
            attention_mask=attn_mask,
            past_key_values=batch_dc,
            use_cache=True,
        )

    scores = out.logits.flatten().tolist()

    # 从输出中提取更新后的 prefix KV（stable_prefix + extension 部分，batch[0]）
    new_prefix_dc = None
    raw_kv = getattr(out, 'past_key_values', None)
    if raw_kv is not None:
        new_len = stable_prefix_len + update_len
        kv_dc = _to_dynamic_cache(raw_kv)
        if kv_dc is not None and _dc_seq_len(kv_dc) >= new_len:
            new_prefix_dc = _dc_extract_batch0_trim(kv_dc, new_len)

    return scores, new_prefix_dc


def _try_score_with_kv_cache(texts: list, request_id: str) -> Optional[list]:
    """
    尝试用跨步 KV cache 打分。优先级：
      HIT ：1 pass (batch=k, seq≈8)                 最快，O(8×N)，与 N 无关
      MISS：2 pass (batch=1, seq=N) + (batch=k, seq≈8)  O(N/k + 8)，约 5× 快于 batch=k full
    返回 None → score() 降级到 baseline (batch=k, seq=N 完整 forward)。
    """
    global _req_kv

    # ── Tokenize（不 padding，保持原始 IDs 用于前缀匹配） ──────────────────────
    try:
        encoded_list = [
            _rm_tok(
                t, return_tensors="pt", truncation=True,
                max_length=2048, padding=False,
            ).input_ids[0].tolist()
            for t in texts
        ]
        raw_prefix = _find_common_prefix(encoded_list)
        stable_len = len(raw_prefix) - _BPE_TRIM
        if stable_len < 16:
            return None
        stable_prefix = raw_prefix[:stable_len]
        bpe_tail = raw_prefix[stable_len:]
    except Exception as e:
        print(f"[RM] tokenize failed ({e})", flush=True)
        return None

    # ── HIT path：复用上一步缓存的 prefix KV ──────────────────────────────────
    try:
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

                if all(len(d) > 0 for d in diff_seqs):
                    scores, new_dc = _score_with_prefix_kv(
                        prefix_dc, cached_len, diff_seqs,
                        update_len=len(extension),
                    )
                    if new_dc is not None:
                        _req_kv[request_id] = {
                            "prefix_ids": stable_prefix,
                            "kv": new_dc,
                        }
                    _kv_stats["hit"] += 1
                    return scores
    except Exception as e:
        print(f"[RM] KV HIT failed ({e}), trying MISS", flush=True)

    # ── Optimized MISS：batch=1 只处理 stable_prefix，再 batch=k 只处理 diff ────
    # 直接传 stable_prefix token IDs（不经 tokenizer，不含 bpe_tail/candidate），
    # 比传整段 texts[0] 更短，且不需要 _dc_extract_batch0_trim（KV 长度恰好等于 stable_len）。
    # 对 N=600：~180ms → ~(T_rm(stable_len)/5 + T_rm(8)) ≈ 17ms，约 10× 加速。
    try:
        prefix_ids_t = torch.tensor(
            [stable_prefix], dtype=torch.long, device=_rm_device,
        )
        with torch.no_grad():
            prefix_out = _rm_model(input_ids=prefix_ids_t, use_cache=True)

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

        scores, _ = _score_with_prefix_kv(
            prefix_dc, stable_len, diff_seqs, update_len=0,
        )

        if len(_req_kv) >= _MAX_REQ_KV:
            del _req_kv[next(iter(_req_kv))]
        _req_kv[request_id] = {"prefix_ids": stable_prefix, "kv": prefix_dc}

        _kv_stats["miss"] += 1
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

    print("[RM] Running warmup pass 1/2: standard forward (may take 20-60s)...", flush=True)
    dummy_texts = ["Warmup forward pass for torch.compile kernel fusion."] * 5
    enc = tok(
        dummy_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=64,
    ).to(device)
    with torch.no_grad():
        out1 = compiled(**enc, use_cache=True)
        _ = compiled(**enc, use_cache=True)

    # 用 pass-1 产出的 KV cache 触发 KV-cache HIT 路径的编译
    print("[RM] Running warmup pass 2/2: KV cache HIT forward...", flush=True)
    raw_kv = getattr(out1, 'past_key_values', None)
    if raw_kv is not None:
        try:
            from transformers.cache_utils import DynamicCache
            kv_dc = _to_dynamic_cache(raw_kv)
            if kv_dc is not None:
                prefix_len = _dc_seq_len(kv_dc) - 4  # 留 4 个 token 作为 diff
                if prefix_len > 0:
                    prefix_dc = _dc_extract_batch0_trim(kv_dc, prefix_len)
                    if prefix_dc is not None:
                        batch_dc = _dc_expand_batch(prefix_dc, 5)
                        diff_ids = torch.zeros((5, 4), dtype=torch.long, device=device)
                        full_len = prefix_len + 4
                        attn_mask = torch.ones((5, full_len), dtype=torch.long, device=device)
                        with torch.no_grad():
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
            print(f"[RM] KV cache warmup skipped ({e})", flush=True)

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
    t0 = time.time()
    with _rm_lock:
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
            encoded = _rm_tok(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(_rm_device)
            t_enc = time.time()
            with torch.no_grad():
                rm_out = _rm_model(**encoded)
            t_fwd = time.time()
            scores = rm_out.logits.flatten().tolist()
            seq_len = encoded["input_ids"].shape[1]
            path = "fallback"
        else:
            t_enc = t_tok
            t_fwd = time.time()
            seq_len = -1

    total = int((t_fwd - t0) * 1000)
    fwd = int((t_fwd - t_enc) * 1000)
    hit = _kv_stats["hit"]
    miss = _kv_stats["miss"]
    hit_rate = hit / (hit + miss) if (hit + miss) > 0 else 0.0
    print(
        f"[RM] score: path={path}  fwd={fwd}ms  total={total}ms"
        f"  seq={seq_len}  hit_rate={hit_rate:.1%}({hit}/{hit+miss})",
        flush=True,
    )
    return {"scores": scores}


@app.post("/reload")
def reload_rm(req: ReloadRequest):
    """热切换 RM（同步，加载完成后才返回）。"""
    global _rm_model, _rm_tok, _req_kv
    with _rm_lock:
        _status["status"] = "reloading"
        print(f"[RM] Reloading: rm={req.rm}  rm_lora={req.rm_lora}", flush=True)

        del _rm_model
        torch.cuda.empty_cache()
        _req_kv.clear()

        if req.rm_lora:
            _rm_model, _rm_tok = _load_rm_with_lora(req.rm, req.rm_lora, _rm_device)
        else:
            _rm_model, _rm_tok = _load_rm_base(req.rm, _rm_device)
        _rm_model.eval()
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
    return p.parse_args()


def main():
    global _rm_model, _rm_tok, _rm_device, _compile_mode
    args = parse_args()
    _rm_device = args.rm_device
    _compile_mode = args.compile

    print("=" * 60)
    print(f"RM       : {args.rm}  device={args.rm_device}")
    if args.rm_lora:
        print(f"RM LoRA  : {args.rm_lora}")
    print(f"Compile  : {args.compile}")
    print(f"Server   : http://{args.host}:{args.port}")
    print("=" * 60)

    if args.rm_lora:
        _rm_model, _rm_tok = _load_rm_with_lora(args.rm, args.rm_lora, args.rm_device)
    else:
        _rm_model, _rm_tok = _load_rm_base(args.rm, args.rm_device)
    _rm_model.eval()
    if args.compile:
        _rm_model = _apply_compile_and_warmup(_rm_model, _rm_tok, args.rm_device)

    _status["status"] = "ready"
    _status["rm"] = args.rm
    _status["rm_lora"] = args.rm_lora
    print("[RM] Ready.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

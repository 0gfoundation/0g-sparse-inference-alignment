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
            output_hidden_states=True,
        )

        if isinstance(out, tuple):
            hidden = out[0]
        elif hasattr(out, 'hidden_states') and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
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
    尝试用跨步 KV cache 打分。
    - HIT：只做 1 次 forward（batch=k, seq=extension+diff），比 baseline 更快
    - MISS：做 1 次标准 batch forward（与 baseline 相同），同时提取并缓存前缀 KV
    任何情况都不会做 2 次 forward。
    """
    global _req_kv

    try:
        # Step 1: tokenize（不 padding，保留原始长度）
        encoded_list = [
            _rm_tok(
                t, return_tensors="pt", truncation=True,
                max_length=2048, padding=False,
            ).input_ids[0].tolist()
            for t in texts
        ]

        # Step 2: 找 k 个候选的公共前缀，再修剪末尾 _BPE_TRIM 个 token 以消除边界影响
        raw_prefix = _find_common_prefix(encoded_list)
        stable_len = len(raw_prefix) - _BPE_TRIM
        if stable_len < 16:
            return None  # 前缀太短，不走 KV cache
        stable_prefix = raw_prefix[:stable_len]

        state = _req_kv.get(request_id)

        if state is not None:
            cached_ids = state["prefix_ids"]
            # 检查 stable_prefix 是否是缓存前缀的单调扩展
            if (len(cached_ids) <= len(stable_prefix)
                    and stable_prefix[:len(cached_ids)] == cached_ids):

                prefix_dc = state["kv"]
                cached_len = _dc_seq_len(prefix_dc)

                # extension = 从上次缓存位置到当前 stable_prefix 末尾的新 token
                extension = stable_prefix[len(cached_ids):]
                # diff_seqs = extension + per-candidate tokens
                # raw_prefix[stable_len:] 是被 trim 掉的 BPE 边界 token，也要放进 diff
                bpe_tail = raw_prefix[stable_len:]
                diff_seqs = [
                    extension + bpe_tail + e[len(raw_prefix):]
                    for e in encoded_list
                ]

                if any(len(d) == 0 for d in diff_seqs):
                    return None

                # 单次 forward：extension + diff，更新 cache 到 stable_prefix 末尾
                scores, new_dc = _score_with_prefix_kv(
                    prefix_dc,
                    cached_len,
                    diff_seqs,
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
        print(f"[RM] KV cache hit path failed ({e}), trying miss path", flush=True)

    # MISS：标准 batch forward（与 baseline 相同速度），但顺便提取前缀 KV
    try:
        encoded_padded = _rm_tok(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(_rm_device)

        with torch.no_grad():
            out = _rm_model(**encoded_padded, use_cache=True)

        scores = out.logits.flatten().tolist()

        # 提取 stable_prefix 的 KV 供下次使用
        raw_kv = getattr(out, 'past_key_values', None)
        if raw_kv is not None and request_id is not None:
            kv_dc = _to_dynamic_cache(raw_kv)
            if kv_dc is not None and _dc_seq_len(kv_dc) >= stable_len:
                prefix_dc = _dc_extract_batch0_trim(kv_dc, stable_len)
                if prefix_dc is not None:
                    if len(_req_kv) >= _MAX_REQ_KV:
                        del _req_kv[next(iter(_req_kv))]
                    _req_kv[request_id] = {
                        "prefix_ids": stable_prefix,
                        "kv": prefix_dc,
                    }

        _kv_stats["miss"] += 1
        return scores

    except Exception as e:
        print(f"[RM] KV cache miss path failed ({e})", flush=True)
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
            scores = _try_score_with_kv_cache(texts, req.request_id)
            if scores is not None:
                path = "kv_hit" if _kv_stats["hit"] > _kv_stats["miss"] else "kv_miss"
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
    return p.parse_args()


def main():
    global _rm_model, _rm_tok, _rm_device
    args = parse_args()
    _rm_device = args.rm_device

    print("=" * 60)
    print(f"RM       : {args.rm}  device={args.rm_device}")
    if args.rm_lora:
        print(f"RM LoRA  : {args.rm_lora}")
    print(f"Server   : http://{args.host}:{args.port}")
    print("=" * 60)

    if args.rm_lora:
        _rm_model, _rm_tok = _load_rm_with_lora(args.rm, args.rm_lora, args.rm_device)
    else:
        _rm_model, _rm_tok = _load_rm_base(args.rm, args.rm_device)
    _rm_model.eval()

    _status["status"] = "ready"
    _status["rm"] = args.rm
    _status["rm_lora"] = args.rm_lora
    print("[RM] Ready.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

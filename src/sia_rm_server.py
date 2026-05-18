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


class _ValueModelOutputWithPast:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


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

        # 在 backbone 调用前记录前缀长度（backbone 会原地扩展 cache，之后读就不准了）
        prefix_kv_len = _kv_seq_len(past_key_values)

        # 新版 HF backbone 要求 Cache 对象，不接受 legacy tuple
        kv_for_backbone = _to_dynamic_cache(past_key_values)

        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_for_backbone,
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
            # KV cache 模式：token_rewards 只包含 diff tokens 的 hidden states
            # 找 diff 中最后一个有效 token 的位置
            if attention_mask is not None:
                diff_mask = attention_mask[:, prefix_kv_len:]
                diff_lens = diff_mask.sum(dim=1) - 1
                logits = token_rewards[
                    torch.arange(token_rewards.size(0)), diff_lens
                ].unsqueeze(-1)
            else:
                logits = token_rewards[:, -1].unsqueeze(-1)
        elif attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1       # (batch,)
            logits = token_rewards[
                torch.arange(token_rewards.size(0)), seq_lens
            ].unsqueeze(-1)                                # (batch, 1)
        else:
            logits = token_rewards[:, -1].unsqueeze(-1)

        if use_cache:
            raw_past = getattr(out, 'past_key_values', None)
            new_past_kv = _to_legacy_tuple(raw_past)
            return _ValueModelOutputWithPast(logits=logits, past_key_values=new_past_kv)
        return _ValueModelOutput(logits=logits)


# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_rm_model = None
_rm_tok = None
_rm_device: str = "cuda:0"
_rm_lock = threading.Lock()
_status = {"status": "initializing", "rm": "", "rm_lora": None}

# KV prefix cache：request_id -> {"prefix_ids": list[int], "past_key_values": tuple}
_prefix_cache: dict = {}
_MAX_CACHE_ENTRIES = 64


# ---------------------------------------------------------------------------
# KV Cache 辅助函数
# ---------------------------------------------------------------------------

def _to_legacy_tuple(past_kv):
    """将 HF past_key_values（DynamicCache 或 tuple）统一转为 legacy tuple 格式。"""
    if past_kv is None:
        return None
    if isinstance(past_kv, tuple):
        return past_kv
    if hasattr(past_kv, 'to_legacy_cache'):
        return past_kv.to_legacy_cache()
    if hasattr(past_kv, 'key_cache'):
        return tuple(
            (past_kv.key_cache[i], past_kv.value_cache[i])
            for i in range(len(past_kv.key_cache))
        )
    return past_kv


def _to_dynamic_cache(past_kv):
    """将 legacy tuple 转为 DynamicCache（新版 HF backbone 要求 Cache 对象）。"""
    if past_kv is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return past_kv  # 旧版 HF，不需要转换
    if isinstance(past_kv, DynamicCache):
        return past_kv
    # legacy tuple: ((K0,V0), (K1,V1), ...) → DynamicCache
    if isinstance(past_kv, tuple):
        return DynamicCache.from_legacy_cache(past_kv)
    return past_kv


def _kv_seq_len(past_kv) -> int:
    """从 past_key_values（任意格式）中读取已缓存的序列长度。"""
    if past_kv is None:
        return 0
    if isinstance(past_kv, tuple) and len(past_kv) > 0:
        return past_kv[0][0].shape[2]
    if hasattr(past_kv, 'key_cache') and past_kv.key_cache:
        return past_kv.key_cache[0].shape[-2]
    if hasattr(past_kv, 'get_seq_length'):
        return past_kv.get_seq_length()
    return 0


def _find_common_prefix(seqs: list) -> list:
    """找到多个 token ID 序列的最长公共前缀。"""
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


def _get_prefix_kv(prefix_ids: list, request_id: str):
    """
    获取 prefix_ids 对应的 KV cache（legacy tuple 格式）。
    若已有缓存且新 prefix 是旧 prefix 的单调扩展，则增量前向；否则从头计算。
    返回 None 表示模型不支持 KV cache。
    """
    global _prefix_cache

    cached = _prefix_cache.get(request_id)
    if cached is not None:
        cached_ids = cached["prefix_ids"]
        # 新 prefix 是旧 prefix 的扩展（单调增长）
        if (len(cached_ids) <= len(prefix_ids)
                and prefix_ids[:len(cached_ids)] == cached_ids):
            if len(cached_ids) == len(prefix_ids):
                # 完全命中
                return cached["past_key_values"]

            # 增量：只对扩展部分做前向
            ext_ids = prefix_ids[len(cached_ids):]
            ext_tensor = torch.tensor([ext_ids], dtype=torch.long, device=_rm_device)
            full_len = len(prefix_ids)
            attn_mask = torch.ones((1, full_len), dtype=torch.long, device=_rm_device)

            with torch.no_grad():
                out = _rm_model(
                    input_ids=ext_tensor,
                    attention_mask=attn_mask,
                    past_key_values=_to_dynamic_cache(cached["past_key_values"]),
                    use_cache=True,
                )
            new_past_kv = _to_legacy_tuple(getattr(out, 'past_key_values', None))
            if new_past_kv is None:
                return None
            _prefix_cache[request_id] = {
                "prefix_ids": prefix_ids,
                "past_key_values": new_past_kv,
            }
            return new_past_kv

    # 全量计算
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=_rm_device)
    attn_mask = torch.ones((1, len(prefix_ids)), dtype=torch.long, device=_rm_device)

    with torch.no_grad():
        out = _rm_model(
            input_ids=prefix_tensor,
            attention_mask=attn_mask,
            use_cache=True,
        )
    new_past_kv = _to_legacy_tuple(getattr(out, 'past_key_values', None))
    if new_past_kv is None:
        return None

    # LRU 淘汰：超出上限时删除最旧条目
    if len(_prefix_cache) >= _MAX_CACHE_ENTRIES:
        oldest = next(iter(_prefix_cache))
        del _prefix_cache[oldest]
    _prefix_cache[request_id] = {
        "prefix_ids": prefix_ids,
        "past_key_values": new_past_kv,
    }
    return new_past_kv


def _score_with_kv_cache(prefix_kv: tuple, diff_seqs: list, prefix_len: int) -> list:
    """
    将 batch=1 的前缀 KV 展开为 batch=k，对各候选的 diff tokens 打分。
    diff_seqs: list of list[int]，右侧会 padding 到相同长度。
    """
    k = len(diff_seqs)
    max_diff_len = max(len(d) for d in diff_seqs)

    # right-pad diff sequences（用 pad_token_id）
    pad_id = _rm_tok.pad_token_id or 0
    diff_ids = torch.full((k, max_diff_len), pad_id, dtype=torch.long, device=_rm_device)
    for i, d in enumerate(diff_seqs):
        diff_ids[i, :len(d)] = torch.tensor(d, dtype=torch.long)

    # 完整 attention_mask：prefix 全 1 + diff 实际长度区域为 1，其余 0
    full_len = prefix_len + max_diff_len
    attn_mask = torch.zeros((k, full_len), dtype=torch.long, device=_rm_device)
    attn_mask[:, :prefix_len] = 1
    for i, d in enumerate(diff_seqs):
        attn_mask[i, prefix_len:prefix_len + len(d)] = 1

    # 将 batch=1 的 KV 扩展到 batch=k，并转为 DynamicCache（新版 HF 要求）
    # prefix_kv 是 legacy tuple: ((K0,V0),(K1,V1),...), K/V shape: (1,heads,seq,head_dim)
    batch_past_kv = _to_dynamic_cache(tuple(
        (k_val.expand(k, -1, -1, -1).contiguous(),
         v_val.expand(k, -1, -1, -1).contiguous())
        for k_val, v_val in prefix_kv
    ))

    with torch.no_grad():
        rm_out = _rm_model(
            input_ids=diff_ids,
            attention_mask=attn_mask,
            past_key_values=batch_past_kv,
        )

    return rm_out.logits.flatten().tolist()


def _try_score_with_kv_cache(texts: list, request_id: str) -> Optional[list]:
    """
    尝试用 KV cache 对候选打分。
    成功返回 scores list；前缀太短或出错则返回 None（退回 batch forward）。
    """
    try:
        # 逐条 tokenize（不 padding，取 token ID 列表）
        encoded_list = [
            _rm_tok(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False,
            ).input_ids[0].tolist()
            for t in texts
        ]

        # 找公共前缀
        prefix_ids = _find_common_prefix(encoded_list)
        min_diff_len = min(len(e) - len(prefix_ids) for e in encoded_list)

        # 前缀太短或 diff 为空，不值得走 KV cache 路径
        if len(prefix_ids) < 8 or min_diff_len < 1:
            return None

        # 获取 / 增量扩展前缀 KV
        prefix_kv = _get_prefix_kv(prefix_ids, request_id)
        if prefix_kv is None:
            return None

        diff_seqs = [e[len(prefix_ids):] for e in encoded_list]
        return _score_with_kv_cache(prefix_kv, diff_seqs, len(prefix_ids))

    except Exception as e:
        print(f"[RM] KV cache scoring failed ({e}), falling back to batch forward", flush=True)
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

app = FastAPI(title="SIA RM Server", version="0.2.0")


class ScoreRequest(BaseModel):
    user_content: str
    response_so_far: str
    candidate_texts: list[str]
    request_id: Optional[str] = None   # 保留兼容性，当前未使用


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
        encoded = _rm_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(_rm_device)
        t_enc = time.time()

        with torch.no_grad():
            rm_out = _rm_model(**encoded)
        t_fwd = time.time()

        scores = rm_out.logits.flatten().tolist()

    seq_len = encoded["input_ids"].shape[1]
    print(
        f"[RM] score: tok={int((t_tok-t0)*1000)}ms"
        f"  enc={int((t_enc-t_tok)*1000)}ms"
        f"  fwd={int((t_fwd-t_enc)*1000)}ms"
        f"  seq_len={seq_len}"
        f"  k={len(texts)}",
        flush=True,
    )
    return {"scores": scores}


@app.post("/reload")
def reload_rm(req: ReloadRequest):
    """热切换 RM（同步，加载完成后才返回）。"""
    global _rm_model, _rm_tok, _prefix_cache
    with _rm_lock:
        _status["status"] = "reloading"
        print(f"[RM] Reloading: rm={req.rm}  rm_lora={req.rm_lora}", flush=True)

        del _rm_model
        torch.cuda.empty_cache()
        _prefix_cache.clear()   # 换模型时清空 KV cache

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

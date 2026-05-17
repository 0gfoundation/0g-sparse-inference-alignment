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

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if hasattr(self.base_model, 'model'):
            backbone = self.base_model.model
        elif hasattr(self.base_model, 'transformer'):
            backbone = self.base_model.transformer
        else:
            backbone = self.base_model

        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if isinstance(out, tuple):
            hidden = out[0]
        elif hasattr(out, 'hidden_states') and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        else:
            hidden = out.last_hidden_state

        token_rewards = self.token_reward_head(hidden.float()).squeeze(-1)  # (batch, seq)
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1       # (batch,)
            logits = token_rewards[
                torch.arange(token_rewards.size(0)), seq_lens
            ].unsqueeze(-1)                                # (batch, 1)
        else:
            logits = token_rewards[:, -1].unsqueeze(-1)
        return _ValueModelOutput(logits=logits)


# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_rm_model = None
_rm_tok = None
_rm_device: str = "cuda:0"
_rm_lock = threading.Lock()
_status = {"status": "initializing", "rm": "", "rm_lora": None}


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

app = FastAPI(title="SIA RM Server", version="0.1.0")


class ScoreRequest(BaseModel):
    user_content: str
    response_so_far: str
    candidate_texts: list[str]


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

        encoded = _rm_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(_rm_device)

        with torch.no_grad():
            rm_out = _rm_model(**encoded)

        scores = rm_out.logits.flatten().tolist()

    return {"scores": scores}


@app.post("/reload")
def reload_rm(req: ReloadRequest):
    """热切换 RM（同步，加载完成后才返回）。"""
    global _rm_model, _rm_tok
    with _rm_lock:
        _status["status"] = "reloading"
        print(f"[RM] Reloading: rm={req.rm}  rm_lora={req.rm_lora}", flush=True)

        del _rm_model
        torch.cuda.empty_cache()

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

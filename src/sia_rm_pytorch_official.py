"""
SIA RM Server — 严格用官方 ValueModel.from_pretrained 加载, 跳过 vllm 服务路径

跟 src/sia_rm_server.py (用 _ValueModelWrapper 自定义实现) 的区别:
本 server 直接 import /workspace/SIA/git/SIA/src/value_model/model.ValueModel,
完全按官方代码加载 base + LoRA + token_reward_head, 不做任何转换。

Endpoint /score 跟现有 sia_rm_server.py 完全兼容, 主 LLM server 不用改任何代码,
只需 --rm_url 指向本 server。

启动:
  python src/sia_rm_pytorch_official.py \\
    --rm /workspace/SIA/models/Qwen3-4B \\
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \\
    --device cuda:0 --port 8001
"""
import argparse
import sys
import time
from typing import Optional

# 把官方 SIA 仓库路径加进来
sys.path.insert(0, '/workspace/SIA/git/SIA')

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.value_model.model import ValueModel


app = FastAPI()

# Globals
VM = None
TOK = None
DEVICE = "cuda:0"


class ScoreRequest(BaseModel):
    user_content: str
    response_so_far: str
    candidate_texts: list[str]
    request_id: Optional[str] = None


class ScoreTokenIdsRequest(BaseModel):
    """跟 vllm /classify (path A direct-token-ids) 完全同 payload schema 对齐:
    input = list[list[int]], 每个内 list = 一个候选完整序列 (prefix + gen + cand)。
    request_id 仅用于 dual-VM 日志关联。
    """
    input: list[list[int]]
    request_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "backend": "pytorch-official-value-model"}


@app.post("/score")
def score(req: ScoreRequest):
    """跟 sia_rm_server.py 同 schema, 但用官方 ValueModel 加载, 返回 raw logit (无 sigmoid)。"""
    try:
        return _score_impl(req)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[RM-error] {type(e).__name__}: {e}\n{tb}", flush=True)
        n = len(req.candidate_texts) if req.candidate_texts else 5
        return {"scores": [0.0] * n, "error": f"{type(e).__name__}: {e}"}


def _score_impl(req: ScoreRequest):
    """
    把 prefix + response_so_far + candidate (无 <|im_end|> close tag, Fix #1 后的格式)
    分别 tokenize, 通过 ValueModel forward, 取 attention_mask 指示的最后非 pad 位置的
    token_reward 作为 raw logit。
    """
    t0 = time.time()

    # 1. 构造 prefix (chat_template 渲染 user_msg + 假 assistant SENTINEL, split)
    convs = [
        {"role": "user", "content": req.user_content},
        {"role": "assistant", "content": "ZSIASENTINELZ"},
    ]
    text = TOK.apply_chat_template(convs, tokenize=False)
    prefix, _ = text.split("ZSIASENTINELZ", 1)
    if TOK.bos_token and prefix.startswith(TOK.bos_token):
        prefix = prefix[len(TOK.bos_token):]

    # 2. 拼 5 个 formatted_text (跟 _score_candidates_vllm Fix #1 后一致, 无 suffix)
    formatted = [prefix + req.response_so_far + ct for ct in req.candidate_texts]

    # 3. 批量 tokenize, padding 到最长
    enc = TOK(formatted, return_tensors='pt', padding=True,
              add_special_tokens=False).to(DEVICE)

    # 4. Forward 官方 ValueModel — 注意官方 forward 是 token_rewards[:, -1],
    #    不看 attention_mask。我们用 attention_mask 找最后非 pad 位置, 才能正确处理
    #    batch 内不同长度的输入 (用 backbone.forward 拿 token_rewards 后手动 pick)。
    with torch.no_grad():
        # ValueModel.forward 返回 token_rewards (batch, seq) 和 logits (batch, 1) = [:, -1]
        # 但 [:, -1] 对 padded 输入是错的, 我们手动从 token_rewards 中按 mask 取最后非 pad
        out = VM(input_ids=enc.input_ids,
                 attention_mask=enc.attention_mask)
        # token_rewards 形状: (batch, seq)
        token_rewards = out.token_rewards
        # 找每行最后一个非 pad 的位置
        seq_lens = enc.attention_mask.sum(dim=1) - 1   # (batch,)
        idx = torch.arange(token_rewards.size(0), device=DEVICE)
        last_rewards = token_rewards[idx, seq_lens]    # (batch,) raw logit

    scores = last_rewards.float().cpu().tolist()

    t_total = time.time() - t0
    return {"scores": scores, "elapsed_ms": round(t_total * 1000, 2)}


@app.post("/score_token_ids")
def score_token_ids(req: ScoreTokenIdsRequest):
    """token-ids 直传版本: input = list[list[int]] 候选完整序列。

    用法: dual-VM 对比 — 跟 vllm /classify (path A direct-token-ids) 接收完全
    一致的 payload (LLM 生成的 token_ids + 候选 cand_id), 跑官方 ValueModel.forward
    + 取最后非 pad token 的 reward, 返回 raw logit list[float]。

    跟 /score 的区别: 完全跳过 apply_chat_template / re-tokenize 路径, 用上游
    传来的 token_ids 直接 forward, 100% 对齐 vllm /classify 的输入分布。
    """
    try:
        t0 = time.time()
        token_id_lists = req.input
        if not token_id_lists:
            return {"scores": [], "elapsed_ms": 0.0}

        # Pad 到最长
        max_len = max(len(ids) for ids in token_id_lists)
        pad_id = TOK.pad_token_id
        input_ids = torch.full(
            (len(token_id_lists), max_len), pad_id,
            dtype=torch.long, device=DEVICE,
        )
        attention_mask = torch.zeros(
            (len(token_id_lists), max_len),
            dtype=torch.long, device=DEVICE,
        )
        for i, ids in enumerate(token_id_lists):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=DEVICE)
            attention_mask[i, :n] = 1

        with torch.no_grad():
            out = VM(input_ids=input_ids, attention_mask=attention_mask)
            token_rewards = out.token_rewards
            seq_lens = attention_mask.sum(dim=1) - 1   # (batch,)
            idx = torch.arange(token_rewards.size(0), device=DEVICE)
            last_rewards = token_rewards[idx, seq_lens]

        scores = last_rewards.float().cpu().tolist()
        t_total = time.time() - t0
        return {"scores": scores, "elapsed_ms": round(t_total * 1000, 2)}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[RM-token-ids-error] {type(e).__name__}: {e}\n{tb}", flush=True)
        n = len(req.input) if req.input else 5
        return {"scores": [0.0] * n, "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rm",      required=True,
                   help="Base 模型路径 (例: /workspace/SIA/models/Qwen3-4B)")
    p.add_argument("--rm_lora", required=True,
                   help="LoRA + token_reward_head 目录 "
                        "(例: /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base)")
    p.add_argument("--device",  default="cuda:0")
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=8001)
    args = p.parse_args()

    global VM, TOK, DEVICE
    DEVICE = args.device

    print(f"[RM-server] Loading official ValueModel.from_pretrained:")
    print(f"  base: {args.rm}")
    print(f"  lora: {args.rm_lora}")
    print(f"  device: {DEVICE}")
    VM = ValueModel.from_pretrained(
        base_model_path=args.rm,
        model_path=args.rm_lora,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).to(DEVICE).eval()

    TOK = AutoTokenizer.from_pretrained(args.rm)
    if TOK.pad_token_id is None:
        TOK.pad_token_id = TOK.eos_token_id

    print(f"[RM-server] VM ready. Starting FastAPI on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

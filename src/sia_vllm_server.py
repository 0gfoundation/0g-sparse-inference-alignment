"""
SIA vLLM OpenAI-compatible HTTP Server

将 sia_vllm_RM.py 包装为 OpenAI-compatible HTTP API，
兼容 0g-serving-broker 的 proxy 机制。

broker 会调用的 endpoints（来自 api/inference/const/const.go）：
  POST /chat/completions      (billing)
  POST /v1/chat/completions   (billing, /v1/ 前缀版本)
  GET  /v1/models             (broker 获取模型信息)
  GET  /health                (健康检查)

注意：streaming 目前为"先生成再分块发送"（simulated streaming），
因为 vllm.LLM 是同步接口。需要真·token-level streaming 时
须切换到 AsyncLLMEngine（后续工作）。

Usage:
  python sia_vllm_server.py \\
    --llm     /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_url  http://localhost:8001 \\
    --host 0.0.0.0 --port 8000 \\
    --llm_gpu_mem 0.3 --topk 5 --weight 1.0
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vllm import LLM, SamplingParams

# sia_vllm_RM.py 与本文件同目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sia_vllm_RM import make_sia_processor

# ---------------------------------------------------------------------------
# 全局状态（在 main() 里初始化）
# ---------------------------------------------------------------------------
_llm: Optional[LLM] = None
_model_id: str = ""        # 对外暴露的 model 名称（basename of llm path）
_args = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="SIA vLLM Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    sia_weight: Optional[float] = None  # SIA-specific: per-request RM score multiplier


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[ChatMessage], sia_weight: Optional[float] = None) -> str:
    """
    将 OpenAI messages 转为 LLM prompt 字符串。
    使用 Human/Assistant 格式（与 sia_vllm_RM.py CLI 行为一致，适合 Base 模型）。
    SIA processor 的 parse_conversation 和 RM 打分均基于此格式设计。
    若指定 sia_weight，在 prompt 开头注入 [SIA:weight=X] header 供处理器读取。
    """
    header = f"[SIA:weight={sia_weight}]\n" if sia_weight is not None else ""
    parts = []
    for m in messages:
        if m.role == "user":
            parts.append(f"Human:\n{m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant:\n{m.content}")
        elif m.role == "system":
            parts.append(m.content)
    parts.append("Assistant:\n")
    return header + "\n".join(parts)


def _build_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature if req.temperature is not None else 0.7,
        max_tokens=req.max_tokens if req.max_tokens is not None else 512,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=req.stop or [],
    )


def _make_chunk(
    request_id: str, created: int, model: str,
    content: str = "", finish_reason=None, role: str = None
) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "sia",
        }],
    }


async def _handle_chat(req: ChatCompletionRequest):
    """共用逻辑：将 messages 转为 prompt，调用 LLM 生成。"""
    prompt = _messages_to_prompt(req.messages, sia_weight=req.sia_weight)
    sampling_params = _build_sampling_params(req)
    model = req.model or _model_id
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream_sse(prompt, sampling_params, request_id, created, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 非流式：同步生成 ──────────────────────────────────────────
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None, lambda: _llm.generate([prompt], sampling_params)
    )
    out = outputs[0]
    text = out.outputs[0].text
    finish_reason = out.outputs[0].finish_reason or "stop"
    prompt_tokens = len(out.prompt_token_ids)
    completion_tokens = len(out.outputs[0].token_ids)

    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


async def _stream_sse(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    """先同步生成完整文本，再以 SSE chunk 格式发送（simulated streaming）。"""
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None, lambda: _llm.generate([prompt], sampling_params)
    )
    text = outputs[0].outputs[0].text
    finish_reason = outputs[0].outputs[0].finish_reason or "stop"

    # 第一个 chunk 带 role
    yield _make_chunk(request_id, created, model, role="assistant")

    # 按字符逐个发送（token-level 粒度最细）
    for ch in text:
        yield _make_chunk(request_id, created, model, content=ch)

    # 结束 chunk
    yield _make_chunk(request_id, created, model, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


# broker 的 TargetRoute 同时包含 /chat/completions 和 /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions_v1(req: ChatCompletionRequest):
    return await _handle_chat(req)


@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    return await _handle_chat(req)


# ---------------------------------------------------------------------------
# CLI & 启动
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA vLLM OpenAI-compatible HTTP Server")
    # LLM 参数
    p.add_argument("--llm",       required=True)
    p.add_argument("--rm_url",    default="http://localhost:8001",
                   help="RM server 地址（默认 http://localhost:8001）")
    p.add_argument("--llm_gpu_mem", type=float, default=0.5)
    p.add_argument("--topk",      type=int,   default=10)
    p.add_argument("--weight",    type=float, default=1.0)
    p.add_argument("--entropy_threshold", type=float, default=None)
    p.add_argument("--max_model_len", type=int, default=4096)
    # HTTP 服务器参数
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model_id", default=None,
                   help="对外暴露的 model 名称（默认取 --llm 的 basename）")
    return p.parse_args()


def main():
    global _llm, _tokenizer, _model_id, _args
    _args = parse_args()

    _model_id = _args.model_id or os.path.basename(_args.llm.rstrip("/"))

    print("=" * 60)
    print(f"Model ID : {_model_id}")
    print(f"LLM      : {_args.llm}")
    print(f"RM URL   : {_args.rm_url}")
    print(f"topk={_args.topk}  weight={_args.weight}  "
          f"entropy_threshold={_args.entropy_threshold}")
    print(f"Server   : http://{_args.host}:{_args.port}")
    print("=" * 60)

    # 构造 SIA LogitsProcessor 类
    SIAProcessor = make_sia_processor(
        rm_url=_args.rm_url,
        topk=_args.topk,
        weight=_args.weight,
        entropy_threshold=_args.entropy_threshold,
    )

    # 加载 LLM（含 SIA processor）
    print("Loading vLLM LLM...")
    _llm = LLM(
        model=_args.llm,
        max_model_len=_args.max_model_len,
        gpu_memory_utilization=_args.llm_gpu_mem,
        logits_processors=[SIAProcessor],
    )
    print("LLM loaded.\n")

    print(f"Starting HTTP server on http://{_args.host}:{_args.port} ...")
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()

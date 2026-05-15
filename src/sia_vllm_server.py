"""
SIA vLLM OpenAI-compatible HTTP Server

将 sia_vllm_RM.py 包装为 OpenAI-compatible HTTP API，
兼容 0g-serving-broker 的 proxy 机制。

broker 会调用的 endpoints（来自 api/inference/const/const.go）：
  POST /chat/completions      (billing)
  POST /v1/chat/completions   (billing, /v1/ 前缀版本)
  GET  /v1/models             (broker 获取模型信息)
  GET  /health                (健康检查)

Usage:
  python src/sia_vllm_server.py \\
    --llm     /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_url  http://localhost:8001 \\
    --host 0.0.0.0 --port 8000 \\
    --llm_gpu_mem 0.3 --topk 5 --weight 1.0
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from typing import AsyncIterator, Optional

# 匹配 "Answer:" + 可选空格 + A/B/C/D（不紧跟其他字母）
_ANSWER_RE = re.compile(r"Answer:\s*([ABCD])(?![a-zA-Z])")

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# sia_vllm_RM.py 与本文件同目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sia_vllm_RM import make_sia_processor

# ---------------------------------------------------------------------------
# 全局状态（在 main() 里初始化）
# ---------------------------------------------------------------------------
_engine: Optional[AsyncLLMEngine] = None
_llm_tok = None
_model_id: str = ""
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[ChatMessage]) -> dict:
    """
    将 OpenAI messages 转为 vLLM prompt dict（token IDs）。
    直接传 token IDs 给 vLLM，跳过二次 tokenize，避免 special token 处理歧义。
    """
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    token_ids = _llm_tok.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
    )
    return {"prompt_token_ids": token_ids}


def _build_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature if req.temperature is not None else 0.7,
        max_tokens=req.max_tokens if req.max_tokens is not None else 512,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=req.stop or [],
        repetition_penalty=1.3,
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


async def _log_rm_status():
    """查询 RM server /status 并打印到日志，失败时静默跳过。"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{_args.rm_url}/status")
            s = resp.json()
            lora = s.get("rm_lora") or "none"
            print(
                f"[SIA] RM status: {s.get('status')}  "
                f"rm={s.get('rm')}  rm_lora={lora}",
                flush=True,
            )
    except Exception as e:
        print(f"[SIA] RM status query failed: {e}", flush=True)


async def _handle_chat(req: ChatCompletionRequest):
    await _log_rm_status()
    prompt = _messages_to_prompt(req.messages)
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

    # 非流式：等待生成完成，超过 60 秒或检测到 Answer: 则立即中止
    REQUEST_TIMEOUT = 60
    start_time = time.time()
    final = None
    async for output in _engine.generate(prompt, sampling_params, request_id):
        final = output
        generated = output.outputs[0].text

        # 全文检测：只要 Answer: 后跟可选空格再接 A/B/C/D，立即停止
        if _ANSWER_RE.search(generated):
            await _engine.abort(request_id)
            break

        elapsed = time.time() - start_time
        if elapsed > REQUEST_TIMEOUT:
            print(
                f"[SERVER] {request_id} timed out after {elapsed:.1f}s, aborting",
                flush=True,
            )
            await _engine.abort(request_id)
            break

    text = final.outputs[0].text if final else ""
    finish_reason = (final.outputs[0].finish_reason or "stop") if final else "timeout"
    prompt_tokens = len(final.prompt_token_ids)
    completion_tokens = len(final.outputs[0].token_ids)

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
    prompt: dict,
    sampling_params: SamplingParams,
    request_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    """每生成一个 token 立即推送一个 SSE chunk（真·token-level streaming）。"""
    yield _make_chunk(request_id, created, model, role="assistant")

    prev_len = 0
    async for output in _engine.generate(prompt, sampling_params, request_id):
        new_text = output.outputs[0].text
        delta = new_text[prev_len:]
        prev_len = len(new_text)
        if delta:
            yield _make_chunk(request_id, created, model, content=delta)
        if output.finished:
            finish_reason = output.outputs[0].finish_reason or "stop"
            yield _make_chunk(request_id, created, model, finish_reason=finish_reason)
            break

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
    p.add_argument("--llm",       required=True)
    p.add_argument("--rm_url",    default="http://localhost:8001",
                   help="RM server 地址（默认 http://localhost:8001）")
    p.add_argument("--llm_gpu_mem", type=float, default=0.5)
    p.add_argument("--topk",      type=int,   default=10)
    p.add_argument("--weight",    type=float, default=1.0)
    p.add_argument("--entropy_threshold", type=float, default=None)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model_id", default=None,
                   help="对外暴露的 model 名称（默认取 --llm 的 basename）")
    return p.parse_args()


def main():
    global _engine, _llm_tok, _model_id, _args
    _args = parse_args()

    _model_id = _args.model_id or os.path.basename(_args.llm.rstrip("/"))

    print(f"Loading tokenizer: {_args.llm} ...")
    _llm_tok = AutoTokenizer.from_pretrained(_args.llm, trust_remote_code=True)
    print("Tokenizer loaded.")

    print("=" * 60)
    print(f"Model ID : {_model_id}")
    print(f"LLM      : {_args.llm}")
    print(f"RM URL   : {_args.rm_url}")
    print(f"topk={_args.topk}  weight={_args.weight}  "
          f"entropy_threshold={_args.entropy_threshold}")
    print(f"Server   : http://{_args.host}:{_args.port}")
    print("=" * 60)

    SIAProcessor = make_sia_processor(
        rm_url=_args.rm_url,
        topk=_args.topk,
        weight=_args.weight,
        entropy_threshold=_args.entropy_threshold,
    )

    print("Loading vLLM AsyncLLMEngine...")
    engine_args = AsyncEngineArgs(
        model=_args.llm,
        max_model_len=_args.max_model_len,
        gpu_memory_utilization=_args.llm_gpu_mem,
        logits_processors=[SIAProcessor],
        disable_log_stats=True,
    )
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("LLM loaded.\n")

    print(f"Starting HTTP server on http://{_args.host}:{_args.port} ...")
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()

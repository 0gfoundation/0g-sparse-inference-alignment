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
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    # 透传给 apply_chat_template 的额外 kwargs (e.g. {"enable_thinking": False})
    # 用法: 跟 OpenAI 兼容 — eval client 传 chat_template_kwargs={"enable_thinking": False}
    chat_template_kwargs: Optional[dict] = None


class CompletionRequest(BaseModel):
    """raw text completion request (跟 OpenAI /v1/completions 一致, 不走 chat_template)。
    适用场景: 需要 raw prompt (e.g. 'Human:\\n...\\nAssistant:\\n') 而非 chat-templated 的场景,
    例如对齐论文 evaluate.py 的 prompt 格式。"""
    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    bad_words: Optional[list[str]] = None  # vllm SamplingParams.bad_words —
                                            # 禁止 sampler 选这些 token
                                            # (e.g., ["<think>","</think>"] 强制不进 thinking 模式)
    n: Optional[int] = 1


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[ChatMessage],
                        chat_template_kwargs: Optional[dict] = None) -> dict:
    """
    将 OpenAI messages 转为 vLLM prompt dict（token IDs）。
    有 chat_template 时用标准 chat template（适合 Instruct 模型）；
    无 chat_template 时 fallback 到 Human/Assistant 纯文本格式（适合 Base 模型）。

    chat_template_kwargs: 透传给 apply_chat_template 的额外 kwargs。
        典型用法: {"enable_thinking": False} 对 Qwen3 Instruct 关掉 thinking 模式。
    """
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    if getattr(_llm_tok, "chat_template", None):
        extra = chat_template_kwargs or {}
        token_ids = _llm_tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            **extra,
        )
    else:
        parts = []
        for m in msgs:
            if m["role"] == "user":
                parts.append(f"Human:\n{m['content']}")
            elif m["role"] == "assistant":
                parts.append(f"Assistant:\n{m['content']}")
            elif m["role"] == "system":
                parts.append(m["content"])
        parts.append("Assistant:\n")
        token_ids = _llm_tok.encode("\n".join(parts), add_special_tokens=True)
    return {"prompt_token_ids": token_ids}


def _build_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    # repetition_penalty 默认 1.0 (paper-aligned, 2026-06-04 修复)。
    # 之前默认 1.3 跟 vllm v1 sampler 顺序 (SIA processor → penalties) 联动,
    # 直接污染 SIA 的 top-K 排序: 200Q Qwen3-14B Skywork mean reward 从 +3.09
    # 跳到 +11.48 (跟 Paper SIA +11.16 统计等价 p=0.43)。
    # 客户端若 model-specific 需要 repetition_penalty != 1.0, 显式传入即可。
    kwargs = dict(
        temperature=req.temperature if req.temperature is not None else 0.7,
        max_tokens=req.max_tokens if req.max_tokens is not None else 512,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=req.stop or [],
        repetition_penalty=(
            req.repetition_penalty if req.repetition_penalty is not None else 1.0
        ),
    )
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    return SamplingParams(**kwargs)


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
    """查询 RM server /status 并打印到日志，失败时静默跳过。b2 inproc 无 HTTP server，跳过。"""
    if _args.rm_backend == "b2":
        return
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
    prompt = _messages_to_prompt(req.messages, req.chat_template_kwargs)
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

    # 非流式：等待生成完成，超过 REQUEST_TIMEOUT 秒或检测到 Answer: 则立即中止
    # 2026-06-03: 60s 默认在 PyTorch RM backend (慢) 下会截断 AlpacaEval 长生成,
    # 导致 SIA arm 跟 noSIA arm (raw vllm 无此 timeout) 比较不公平 — 改 600s
    # 让 max_tokens=2048 真正能跑到; MMLU 类短答案不受影响 (Answer: A 几秒就触发 abort)
    REQUEST_TIMEOUT = 600
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


async def _handle_completion(req: CompletionRequest):
    """Raw text completion (跳过 chat_template)。SIA logits processor 仍按
    每个 decode step 触发, 跟 chat_completion 完全一样。"""
    await _log_rm_status()
    # 直接用 raw prompt — tokenize 走 vllm 内部 (它接受 string prompt)
    prompt = req.prompt
    # 构造 SamplingParams: 复用 _build_sampling_params 逻辑, 但 req 是 CompletionRequest
    # 不是 ChatCompletionRequest, 这里直接构造
    kwargs = dict(
        temperature=req.temperature if req.temperature is not None else 0.7,
        max_tokens=req.max_tokens if req.max_tokens is not None else 512,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=req.stop or [],
        repetition_penalty=(
            req.repetition_penalty if req.repetition_penalty is not None else 1.0
        ),
    )
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.bad_words:
        kwargs["bad_words"] = req.bad_words
    sampling_params = SamplingParams(**kwargs)
    model = req.model or _model_id
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    final = None
    async for output in _engine.generate(prompt, sampling_params, request_id):
        final = output

    text = final.outputs[0].text if final else ""
    finish_reason = (final.outputs[0].finish_reason or "stop") if final else "stop"
    prompt_tokens = len(final.prompt_token_ids)
    completion_tokens = len(final.outputs[0].token_ids)

    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.post("/v1/completions")
async def completions_v1(req: CompletionRequest):
    return await _handle_completion(req)


@app.post("/completions")
async def completions(req: CompletionRequest):
    return await _handle_completion(req)


# ---------------------------------------------------------------------------
# CLI & 启动
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA vLLM OpenAI-compatible HTTP Server")
    p.add_argument("--llm",       required=True)
    p.add_argument("--rm_url",    default="http://localhost:8001",
                   help="RM server 地址（默认 http://localhost:8001）")
    p.add_argument("--rm_backend",
                   choices=["pytorch", "vllm", "b2"], default="pytorch",
                   help="RM 后端：pytorch=src/sia_rm_server.py（自定义 /score）；"
                        "vllm=vllm serve（/classify，推荐：更快更稳）；"
                        "b2=in-process RMClient（嵌套 vLLM 实例，无 HTTP 开销）")
    p.add_argument("--rm_model",  default=None,
                   help="rm_backend ∈ {vllm, b2} 时必填: RM 模型路径")
    p.add_argument("--rm_b2_gpu_mem", type=float, default=0.3,
                   help="b2 backend 下 RM vLLM 实例的 gpu_memory_utilization (默认 0.3)")
    p.add_argument("--llm_gpu_mem", type=float, default=0.5)
    p.add_argument("--topk",      type=int,   default=10)
    p.add_argument("--weight",    type=float, default=1.0)
    p.add_argument("--entropy_threshold", type=float, default=None)
    p.add_argument("--use_token_ids", action="store_true",
                   help="rm_backend=vllm 时：客户端预先 tokenize 并发 token_ids 到 RM，"
                        "省服务端 re-tokenize（~3-5ms/call）。需要 RM server 用 "
                        "scripts/vllm_serve_with_token_ids.py 启动以打 Pydantic 补丁。")
    p.add_argument("--max_model_len", type=int, default=4096)
    # vllm Automatic Prefix Caching (APC). Default ON (生产推荐):
    # 多 request 共享前缀时 2-10x prefill 加速; SIA 跟 APC 正交不冲突。
    # 想关传 --disable_prefix_caching.
    p.add_argument("--enable_prefix_caching", dest="enable_prefix_caching",
                   action="store_true", default=True,
                   help="Enable vllm APC (default).")
    p.add_argument("--disable_prefix_caching", dest="enable_prefix_caching",
                   action="store_false",
                   help="Disable vllm APC (not recommended for production).")
    p.add_argument("--enable_thinking", choices=["true", "false"], default=None,
                   help="透传 enable_thinking 给 RM prefix 构造, 跟 LLM 实际看到的 prompt 100% 一致。"
                        "Qwen3 Instruct 模型 default=true (含 <think> 注入); "
                        "传 false 跟 eval client --disable_thinking 一致; "
                        "default=None 表示不传 (用 chat_template 默认值, 也即模型本身默认)。")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model_id", default=None,
                   help="对外暴露的 model 名称（默认取 --llm 的 basename）")
    p.add_argument("--dummy_processor", choices=[None, "noop", "sync"],
                   default=None,
                   help="校准用: 替换 SIA processor 为 dummy. "
                        "noop=完全 no-op (测 hook 开销); "
                        "sync=做 entropy+cpu sync 但不做决策 (测 sync 开销). "
                        "默认 None=用真实 SIA processor.")
    args = p.parse_args()
    if args.rm_backend in ("vllm", "b2") and not args.rm_model:
        p.error(f"--rm_backend {args.rm_backend} 必须同时指定 --rm_model")
    return args


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
    print(f"RM mode  : {_args.rm_backend}"
          + (f"  model={_args.rm_model}" if _args.rm_backend == "vllm" else ""))
    print(f"topk={_args.topk}  weight={_args.weight}  "
          f"entropy_threshold={_args.entropy_threshold}  "
          f"use_token_ids={_args.use_token_ids}")
    print(f"Server   : http://{_args.host}:{_args.port}")
    print("=" * 60)

    if _args.dummy_processor is not None:
        # Calibration mode: replace SIA processor with a dummy (no RM call).
        # 用来分离 "SIA sync 开销" 跟 "SIA RM 调用 + 决策开销"。
        from dummy_lp import make_dummy_processor
        SIAProcessor = make_dummy_processor(
            _args.dummy_processor, topk=_args.topk
        )
        print(f"[CALIB] dummy_processor={_args.dummy_processor}, "
              f"NO RM call, NO logits modification", flush=True)
    else:
        # 把 "true"/"false" 字符串转成 bool / None
        _enable_thinking = (
            True if _args.enable_thinking == "true"
            else False if _args.enable_thinking == "false"
            else None
        )
        SIAProcessor = make_sia_processor(
            rm_url=_args.rm_url,
            topk=_args.topk,
            weight=_args.weight,
            entropy_threshold=_args.entropy_threshold,
            rm_backend=_args.rm_backend,
            rm_model=_args.rm_model,
            use_token_ids=_args.use_token_ids,
            rm_b2_gpu_mem=_args.rm_b2_gpu_mem,
            enable_thinking=_enable_thinking,
        )

    print("Loading vLLM AsyncLLMEngine...")
    engine_kwargs = dict(
        model=_args.llm,
        max_model_len=_args.max_model_len,
        gpu_memory_utilization=_args.llm_gpu_mem,
        logits_processors=[SIAProcessor],
        disable_log_stats=True,
        enable_prefix_caching=_args.enable_prefix_caching,
    )
    print(f"[SIA] main LLM prefix_caching = {_args.enable_prefix_caching}")
    # Optional main-LLM cudagraph override. Default = vllm's default (keeps
    # legacy Qwen14B / vllm 0.10 path untouched). Set to "piecewise" when
    # running a nested-vllm RM under vllm >= 0.15: FULL_AND_PIECEWISE on the
    # main LLM raises a global cudagraph-capturing flag that breaks RM forward.
    llm_cg = os.environ.get("SIA_LLM_CUDAGRAPH", "default").lower()
    if llm_cg in ("piecewise", "none", "eager"):
        try:
            from vllm.config import CompilationConfig  # vllm >= 0.13
            if llm_cg == "piecewise":
                engine_kwargs["compilation_config"] = CompilationConfig(cudagraph_mode=1)
                print("[SIA] main LLM cudagraph_mode=PIECEWISE (SIA_LLM_CUDAGRAPH=piecewise)")
            else:
                engine_kwargs["enforce_eager"] = True
                print("[SIA] main LLM enforce_eager=True (SIA_LLM_CUDAGRAPH=none)")
        except ImportError:
            print("[SIA] SIA_LLM_CUDAGRAPH ignored (vllm too old to support compilation_config)")
    engine_args = AsyncEngineArgs(**engine_kwargs)
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("LLM loaded.\n")

    print(f"Starting HTTP server on http://{_args.host}:{_args.port} ...")
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()

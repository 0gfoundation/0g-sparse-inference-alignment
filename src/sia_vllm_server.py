"""
SIA vLLM OpenAI-compatible HTTP Server

Wraps sia_vllm_RM.py as an OpenAI-compatible HTTP API,
compatible with the 0g-serving-broker proxy mechanism.

Endpoints called by the broker (from api/inference/const/const.go):
  POST /chat/completions      (billing)
  POST /v1/chat/completions   (billing, /v1/ prefix variant)
  GET  /v1/models             (broker fetches model info)
  GET  /health                (health check)

Usage:
  python src/sia_vllm_server.py \\
    --llm     /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_url  http://localhost:8001 \\
    --host 0.0.0.0 --port 8000 \\
    --llm_gpu_mem 0.3 --topk 5 --weight 1.0
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import uuid
from typing import AsyncIterator, Optional, Union

# Match "Answer:" + optional whitespace + A/B/C/D (not immediately followed by other letters)
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

# sia_vllm_RM.py is in the same directory as this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sia_vllm_RM import make_sia_processor

# ---------------------------------------------------------------------------
# Global state (initialized in main())
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
    content: Union[str, list]  # list for multimodal (OpenAI image_url format)


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
    # Extra kwargs passed through to apply_chat_template (e.g. {"enable_thinking": False})
    # Usage: OpenAI-compatible — eval client passes chat_template_kwargs={"enable_thinking": False}
    chat_template_kwargs: Optional[dict] = None
    stream_options: Optional[dict] = None          # e.g. {"include_usage": true} — router force-injects this field
    # Per-request SIA overrides. None = use server default (from --weight / --topk / --entropy_threshold).
    sia_weight: Optional[float] = None            # 0.0 = disable SIA for this request
    sia_topk: Optional[int] = None                # override number of RM candidates
    sia_entropy_threshold: Optional[float] = None  # override entropy gate; 0 = always intervene
    tools: Optional[list] = None                  # not supported; triggers 400 if set
    tool_choice: Optional[object] = None          # not supported; triggers 400 if set


class CompletionRequest(BaseModel):
    """Raw text completion request (compatible with OpenAI /v1/completions, bypasses chat_template).
    Use case: when a raw prompt (e.g. 'Human:\\n...\\nAssistant:\\n') is needed instead of
    chat-templated format, e.g. the prompt format used by alignment paper evaluate.py."""
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
                                            # prevents sampler from selecting these tokens
                                            # (e.g., ["<think>","</think>"] forces non-thinking mode)
    n: Optional[int] = 1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _has_image(messages: list[ChatMessage]) -> bool:
    """Return True if any message contains image content (OpenAI image_url format)."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


async def _messages_to_multimodal_prompt(
    messages: list[ChatMessage],
    chat_template_kwargs: Optional[dict] = None,
) -> dict:
    """Build vLLM multimodal prompt dict for requests that contain images.

    Converts OpenAI image_url content parts to PIL Images, builds the chat
    template string with image placeholders, and returns a vLLM prompt dict
    suitable for VL models.
    """
    try:
        from PIL import Image as _PILImage
    except ImportError:
        raise RuntimeError("Pillow is required for multimodal inputs: pip install Pillow")

    images = []
    qwen_msgs = []

    for msg in messages:
        if isinstance(msg.content, str):
            qwen_msgs.append({"role": msg.role, "content": msg.content})
            continue

        content_parts = []
        for part in msg.content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "image_url":
                # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    # base64-encoded data URI: data:<mime>;base64,<data>
                    _, b64data = url.split(",", 1)
                    img = _PILImage.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
                else:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        img = _PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                images.append(img)
                content_parts.append({"type": "image"})  # placeholder for chat template
            elif ptype == "image":
                # Qwen-native format: {"type": "image", "image": url_or_path}
                img_ref = part.get("image") or ""
                if img_ref.startswith("data:"):
                    _, b64data = img_ref.split(",", 1)
                    img = _PILImage.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
                else:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.get(img_ref)
                        resp.raise_for_status()
                        img = _PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                images.append(img)
                content_parts.append({"type": "image"})  # placeholder for chat template
            elif ptype == "text":
                content_parts.append({"type": "text", "text": part.get("text", "")})
        qwen_msgs.append({"role": msg.role, "content": content_parts})

    extra = chat_template_kwargs or {}
    text = _llm_tok.apply_chat_template(
        qwen_msgs,
        tokenize=False,
        add_generation_prompt=True,
        **extra,
    )

    prompt: dict = {"prompt": text}
    if images:
        prompt["multi_modal_data"] = {"image": images}
    return prompt


def _messages_to_prompt(messages: list[ChatMessage],
                        chat_template_kwargs: Optional[dict] = None) -> dict:
    """
    Convert OpenAI messages to a vLLM prompt dict (token IDs).
    Uses the standard chat template when available (suitable for Instruct models);
    falls back to Human/Assistant plain-text format when no chat_template is set
    (suitable for Base models).

    chat_template_kwargs: extra kwargs passed through to apply_chat_template.
        Typical use: {"enable_thinking": False} to disable thinking mode for Qwen3 Instruct.
    """
    msgs = []
    for m in messages:
        if isinstance(m.content, str):
            msgs.append({"role": m.role, "content": m.content})
        else:
            # Multimodal content — extract text parts only (images go via multimodal path)
            text = " ".join(
                p.get("text", "") for p in m.content
                if isinstance(p, dict) and p.get("type") == "text"
            )
            msgs.append({"role": m.role, "content": text})
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
    # repetition_penalty defaults to 1.0 (paper-aligned, fixed 2026-06-04).
    # The previous default of 1.3 interacted with vllm v1 sampler ordering
    # (SIA processor → penalties), directly polluting SIA's top-K ranking:
    # 200Q Qwen3-14B Skywork mean reward jumped from +3.09 to +11.48
    # (statistically equivalent to Paper SIA +11.16 at p=0.43).
    # Clients that need model-specific repetition_penalty != 1.0 can pass it explicitly.
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
    sia_extra: dict = {}
    if req.sia_weight is not None:
        sia_extra["sia_weight"] = req.sia_weight
    if req.sia_topk is not None:
        sia_extra["sia_topk"] = req.sia_topk
    if req.sia_entropy_threshold is not None:
        sia_extra["sia_entropy_threshold"] = req.sia_entropy_threshold
    if sia_extra:
        kwargs["extra_args"] = sia_extra
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
            "owned_by": "0G Foundation",
        }],
    }


def _check_model(model: Optional[str]) -> Optional[JSONResponse]:
    """Validate the model field: None / empty string / _model_id / full path are all valid;
    anything else returns a 404 JSONResponse.
    Caller: if err := _check_model(req.model): return err
    """
    if not model:
        return None
    if model in (_model_id, _args.llm):
        return None
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"The model `{model}` does not exist or is not loaded.",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        },
    )


async def _log_rm_status():
    """Query RM server /status and print to log; silently skip on failure. b2 inproc has no HTTP server, so skip."""
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
    if err := _check_model(req.model):
        return err
    if not req.messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "[] is too short - 'messages'",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )
    if req.tools:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Tool calls are not yet implemented in this server.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )
    await _log_rm_status()

    if _has_image(req.messages):
        # VM is text-only: bypass SIA entirely for image-containing requests.
        # sia_weight=0 triggers the early-exit path in SIALogitsProcessor (no VM call).
        prompt = await _messages_to_multimodal_prompt(req.messages, req.chat_template_kwargs)
        req = req.model_copy(update={"sia_weight": 0.0})
        print("[SIA] multimodal request detected — SIA bypassed (VM is text-only)", flush=True)
    else:
        prompt = _messages_to_prompt(req.messages, req.chat_template_kwargs)

    sampling_params = _build_sampling_params(req)
    model = _model_id
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        include_usage = bool(
            req.stream_options and req.stream_options.get("include_usage")
        )
        return StreamingResponse(
            _stream_sse(prompt, sampling_params, request_id, created, model,
                        include_usage=include_usage),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: wait for generation to complete; abort immediately if
    # REQUEST_TIMEOUT seconds elapse or "Answer:" is detected.
    # 2026-06-03: the 60s default truncated long AlpacaEval generations under
    # PyTorch RM backend (slow), making the SIA arm unfairly compared to the
    # noSIA arm (raw vllm has no such timeout) — changed to 600s so
    # max_tokens=2048 can actually run to completion; MMLU-style short answers
    # are unaffected (Answer: A triggers abort within a few seconds).
    REQUEST_TIMEOUT = 600
    start_time = time.time()
    final = None
    try:
        async for output in _engine.generate(prompt, sampling_params, request_id):
            final = output
            generated = output.outputs[0].text

            # Full-text check: stop immediately if Answer: is followed by optional whitespace then A/B/C/D
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
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error",
                                "param": None, "code": None}},
        )

    text = final.outputs[0].text if final else ""
    finish_reason = (final.outputs[0].finish_reason or "stop") if final else "timeout"
    prompt_tokens = len(final.prompt_token_ids)
    completion_tokens = len(final.outputs[0].token_ids)
    cached_tokens = (final.num_cached_tokens or 0) if final else 0

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
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    })


async def _stream_sse(
    prompt: dict,
    sampling_params: SamplingParams,
    request_id: str,
    created: int,
    model: str,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Push one SSE chunk immediately per generated token (true token-level streaming).

    When include_usage=True, an extra usage chunk is sent before [DONE] (OpenAI V3 billing requirement):
      data: {"id":...,"choices":[],"usage":{"prompt_tokens":...,"completion_tokens":...,"total_tokens":...}}
    The router force-injects stream_options.include_usage=true for streaming requests,
    so this path is always triggered in production.
    """
    yield _make_chunk(request_id, created, model, role="assistant")

    prev_len = 0
    final_output = None
    async for output in _engine.generate(prompt, sampling_params, request_id):
        new_text = output.outputs[0].text
        delta = new_text[prev_len:]
        prev_len = len(new_text)
        if delta:
            yield _make_chunk(request_id, created, model, content=delta)
        if output.finished:
            final_output = output
            finish_reason = output.outputs[0].finish_reason or "stop"
            yield _make_chunk(request_id, created, model, finish_reason=finish_reason)
            break

    if include_usage and final_output is not None:
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(final_output.outputs[0].token_ids)
        cached_tokens = final_output.num_cached_tokens or 0
        usage_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


# broker's TargetRoute includes both /chat/completions and /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions_v1(req: ChatCompletionRequest):
    return await _handle_chat(req)


@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    return await _handle_chat(req)


async def _handle_completion(req: CompletionRequest):
    """Raw text completion (bypasses chat_template). SIA logits processor still
    fires on every decode step, identical to chat_completion."""
    if err := _check_model(req.model):
        return err
    await _log_rm_status()
    # Use raw prompt directly — tokenization is handled by vllm internally (it accepts string prompts)
    prompt = req.prompt
    # Build SamplingParams: mirrors _build_sampling_params logic, but req is a CompletionRequest
    # not a ChatCompletionRequest, so we construct it directly here
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
    model = _model_id
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
# CLI & startup
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA vLLM OpenAI-compatible HTTP Server")
    p.add_argument("--llm",       required=True)
    p.add_argument("--rm_url",    default="http://localhost:8001",
                   help="RM server address (default http://localhost:8001)")
    p.add_argument("--rm_backend",
                   choices=["pytorch", "vllm", "b2"], default="pytorch",
                   help="RM backend: pytorch=src/sia_rm_server.py (custom /score); "
                        "vllm=vllm serve (/classify, recommended: faster and more stable); "
                        "b2=in-process RMClient (nested vLLM instance, no HTTP overhead)")
    p.add_argument("--rm_model",  default=None,
                   help="Required when rm_backend is vllm or b2: path to the RM model")
    p.add_argument("--rm_b2_gpu_mem", type=float, default=0.3,
                   help="gpu_memory_utilization for the RM vLLM instance under b2 backend (default 0.3)")
    p.add_argument("--llm_gpu_mem", type=float, default=0.5)
    p.add_argument("--topk",      type=int,   default=10)
    p.add_argument("--weight",    type=float, default=1.0)
    p.add_argument("--entropy_threshold", type=float, default=None)
    p.add_argument("--logit_gap_threshold", type=float, default=None,
                   help="Dual-gate: skip VM scoring when top1-top2 logit gap >= this value "
                        "(None=disabled). Reduces intervention rate without extra GPU sync. "
                        "Validate with AlpacaEval A/B before enabling in production.")
    p.add_argument("--eager_vm_prefill", action="store_true", default=False,
                   help="b2 backend: after each decode step pre-warm the VM KV cache with the "
                        "predicted next token for all active sessions. Reduces tail prefill "
                        "cost at the next INTERVENE step.")
    p.add_argument("--vm_head_type", default="scalar",
                   choices=["scalar", "vocab_lowrank"],
                   help="b2 backend: VM reward head type. "
                        "'scalar' (default): K RM forwards per INTERVENE step. "
                        "'vocab_lowrank': 1 RM forward per step (FaRMA, ~K× speedup). "
                        "Requires checkpoint converted with convert_rm_for_vllm.py --head_type vocab_lowrank.")
    p.add_argument("--vm_head_rank", type=int, default=64,
                   help="Rank for vocab_lowrank head (default 64, ignored for scalar head).")
    p.add_argument("--vm_topk", type=int, default=None,
                   help="Limit candidates sent to VM per request (default: same as --topk). "
                        "Keeps --topk for entropy gate so intervention rate is unchanged. "
                        "Used for FaRMA proxy benchmarking: --topk 10 --vm_topk 1 reduces "
                        "VM batch from N×10 to N×1 while preserving the intervention rate.")
    p.add_argument("--use_token_ids", action="store_true",
                   help="When rm_backend=vllm: pre-tokenize on the client and send token_ids to RM, "
                        "saving server-side re-tokenize (~3-5ms/call). Requires RM server to be "
                        "started with scripts/vllm_serve_with_token_ids.py to apply Pydantic patch.")
    p.add_argument("--max_model_len", type=int, default=4096)
    # Mamba layer prefix caching mode (only relevant for Hybrid models like 0GM-35B).
    # Default "none" sets mamba_block_size to max_model_len, making lcm_block_size
    # very large so no requests can hit APC in practice. "align" aligns mamba_block_size
    # to the attention layer block_size (~1056 tokens), enabling cached_tokens reporting
    # for prompts >= 1056 tokens. The Qwen3.5 code comments for 0GM-35B explicitly say
    # "please use align".
    p.add_argument("--mamba_cache_mode", default=None,
                   choices=["all", "align", "none"],
                   help="Passed through to vllm --mamba-cache-mode. Recommended: align for Hybrid models (0GM-35B). "
                        "default=None means not passed (vllm uses its default value none).")
    # vllm Automatic Prefix Caching (APC). Default ON (recommended for production):
    # 2-10x prefill speedup when requests share a prefix; SIA and APC are orthogonal.
    # Pass --disable_prefix_caching to turn it off.
    p.add_argument("--enable_prefix_caching", dest="enable_prefix_caching",
                   action="store_true", default=True,
                   help="Enable vllm APC (default).")
    p.add_argument("--disable_prefix_caching", dest="enable_prefix_caching",
                   action="store_false",
                   help="Disable vllm APC (not recommended for production).")
    p.add_argument("--enable_thinking", choices=["true", "false"], default=None,
                   help="Passed through to RM prefix construction to match the prompt seen by the LLM exactly. "
                        "Qwen3 Instruct models default=true (includes <think> injection); "
                        "pass false to match eval client --disable_thinking; "
                        "default=None means not passed (uses chat_template default, i.e. model's own default).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model_id", default=None,
                   help="Model name exposed externally (defaults to the basename of --llm)")
    p.add_argument("--dummy_processor", choices=[None, "noop", "sync"],
                   default=None,
                   help="Calibration: replace SIA processor with a dummy. "
                        "noop=complete no-op (measure hook overhead); "
                        "sync=perform entropy+cpu sync but no decision (measure sync overhead). "
                        "Default None=use the real SIA processor.")
    args = p.parse_args()
    if args.rm_backend in ("vllm", "b2") and not args.rm_model:
        p.error(f"--rm_backend {args.rm_backend} requires --rm_model to be specified")
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
    print(f"topk={_args.topk}  vm_topk={_args.vm_topk}  weight={_args.weight}  "
          f"entropy_threshold={_args.entropy_threshold}  "
          f"logit_gap_threshold={_args.logit_gap_threshold}  "
          f"use_token_ids={_args.use_token_ids}  "
          f"eager_vm_prefill={_args.eager_vm_prefill}")
    print(f"Server   : http://{_args.host}:{_args.port}")
    print("=" * 60)

    if _args.dummy_processor is not None:
        # Calibration mode: replace SIA processor with a dummy (no RM call).
        # Used to isolate "SIA sync overhead" from "SIA RM call + decision overhead".
        from dummy_lp import make_dummy_processor
        SIAProcessor = make_dummy_processor(
            _args.dummy_processor, topk=_args.topk
        )
        print(f"[CALIB] dummy_processor={_args.dummy_processor}, "
              f"NO RM call, NO logits modification", flush=True)
    else:
        # Convert "true"/"false" strings to bool / None
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
            logit_gap_threshold=_args.logit_gap_threshold,
            rm_backend=_args.rm_backend,
            rm_model=_args.rm_model,
            use_token_ids=_args.use_token_ids,
            rm_b2_gpu_mem=_args.rm_b2_gpu_mem,
            rm_max_model_len=_args.max_model_len,
            enable_thinking=_enable_thinking,
            eager_vm_prefill=_args.eager_vm_prefill,
            vm_topk=_args.vm_topk,
            vm_head_type=_args.vm_head_type,
            vm_head_rank=_args.vm_head_rank,
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
    if _args.mamba_cache_mode is not None:
        engine_kwargs["mamba_cache_mode"] = _args.mamba_cache_mode
    print(f"[SIA] main LLM prefix_caching = {_args.enable_prefix_caching}, "
          f"mamba_cache_mode = {_args.mamba_cache_mode or '(vllm default)'}")
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

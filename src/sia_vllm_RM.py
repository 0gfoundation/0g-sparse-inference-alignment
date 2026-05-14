"""
SIA per-token 干预（HTTP RM Server 版本）- vLLM v1 engine 版本

架构：
  - LLM: vllm (v1 engine) 负责生成，GPU 由 vllm 管理
  - RM:  独立 RM server（sia_rm_server.py），通过 HTTP /score 调用
  - SIALogitsProcessor 运行在 EngineCore 子进程中，通过 LLM(logits_processors=[...]) 注册

干预逻辑（per-token）：
  1. 从 LLM logits 中取 top-k candidate token IDs
  2. 将当前生成序列 + 各 candidate 解码成文本
  3. 通过 HTTP POST /score 批量发送到 RM server 打分
  4. combined_logits[topk_indices] += rm_scores * weight
  5. 返回 combined_logits

用法示例：
  # 先启动 RM server（sia_rm_server.py），再启动本脚本
  python sia_vllm_RM.py \\
    --llm  /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_url http://localhost:8001 \\
    --llm_gpu_mem 0.3 --topk 5 --weight 0.1 --max_tokens 20
"""

import argparse
import re
import requests
import time
from typing import Optional

import torch
import torch.distributions as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_SIA_WEIGHT_RE = re.compile(r'^\[SIA:weight=([-+]?[0-9]*\.?[0-9]+)\]\n?')


def _parse_sia_header(prompt_text: str):
    """
    检查 prompt 开头是否有 [SIA:weight=X] header（由 HTTP server 注入）。
    返回 (weight, cleaned_text)；无 header 则 weight 为 None。
    """
    m = _SIA_WEIGHT_RE.match(prompt_text)
    if m:
        return float(m.group(1)), prompt_text[m.end():]
    return None, prompt_text


def parse_conversation(text: str):
    """
    将对话文本解析为 conversations list。支持以下格式：
      - 'Human:\\nQ\\nAssistant:\\nA'（带冒号格式）
      - '[INST] Q [/INST] A'（Llama 2 / Mistral 格式）
      - 'user\\nQ\\nassistant\\nA'（ChatML stripped，如 Qwen3 / Llama 3）
      - 'user\\nQ\\nmodel\\nA'（Gemma，model 为 assistant 别名）
    """
    text = text.strip()

    # Format 1: Human: / Assistant: 带冒号
    parts = re.split(r'(Human|Assistant):\s*', text, flags=re.IGNORECASE)
    conversations = []
    current_role = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.lower() == 'human':
            current_role = 'user'
        elif part.lower() == 'assistant':
            current_role = 'assistant'
        else:
            if current_role:
                conversations.append({'role': current_role, 'content': part})
    if conversations:
        return conversations

    # Format 2: [INST] ... [/INST] (Llama 2 / Mistral)
    if '[INST]' in text:
        inst_matches = re.findall(
            r'\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|$)', text, re.DOTALL
        )
        if inst_matches:
            conversations = []
            for user_part, asst_part in inst_matches:
                user_part = user_part.strip()
                asst_part = asst_part.strip()
                if user_part:
                    conversations.append({'role': 'user', 'content': user_part})
                if asst_part:
                    conversations.append({'role': 'assistant', 'content': asst_part})
            if conversations:
                return conversations

    # Format 3: role 单独一行（ChatML stripped / Gemma 等）
    # 'model' 是 Gemma 对 assistant 的别名
    parts = re.split(r'\n(user|assistant|system|human|model)\n', '\n' + text, flags=re.IGNORECASE)
    conversations = []
    current_role = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lower = part.lower()
        if lower in ('user', 'human'):
            current_role = 'user'
        elif lower in ('assistant', 'model'):
            current_role = 'assistant'
        elif lower == 'system':
            current_role = 'system'
        else:
            if current_role:
                conversations.append({'role': current_role, 'content': part})
    if conversations:
        return conversations

    return [{'role': 'user', 'content': text}]


def extract_user_content(prompt_text: str) -> str:
    """从 prompt 文本中提取最后一个 user turn 内容。"""
    user_content = None
    for c in parse_conversation(prompt_text):
        if c['role'] == 'user':
            user_content = c['content']
    return user_content if user_content is not None else prompt_text


# ---------------------------------------------------------------------------
# LogitsProcessor 工厂
# ---------------------------------------------------------------------------

def make_sia_processor(
    rm_url: str = "http://localhost:8001",
    topk: int = 10,
    weight: float = 1.0,
    entropy_threshold: Optional[float] = None,
):
    """
    返回一个 SIALogitsProcessor 类（不是实例）。
    vllm 引擎以 (vllm_config, device, is_pin_memory) 参数实例化该类。
    RM 打分通过 HTTP 调用独立的 sia_rm_server.py 完成。
    """

    class SIALogitsProcessor(LogitsProcessor):
        _RM_URL = rm_url
        _TOPK = topk
        _WEIGHT = weight
        _ENTROPY_THRESHOLD = entropy_threshold

        # ----------------------------------------------------------------
        # 初始化：在 EngineCore 子进程里执行
        # ----------------------------------------------------------------
        def __init__(self, vllm_config, device: torch.device,
                     is_pin_memory: bool) -> None:
            self._llm_device = device

            # LLM tokenizer（用于解码候选 token、提取 user content）
            llm_model_path = vllm_config.model_config.model
            print(f"[SIA] Loading LLM tokenizer: {llm_model_path}", flush=True)
            self._llm_tok = AutoTokenizer.from_pretrained(
                llm_model_path, trust_remote_code=True
            )

            # HTTP session（复用 TCP 连接到 RM server）
            self._rm_session = requests.Session()
            print(f"[SIA] RM URL: {self._RM_URL}", flush=True)

            # 预计算哨兵边界（启动时一次，用于 _extract_user_content）
            self._sentinel_prefix, self._sentinel_suffix = self._compute_sentinel_bounds()

            # per-request 状态
            self._prompt_user: dict[int, str] = {}
            self._output_ids: dict[int, list] = {}
            self._weight_per_req: dict[int, float] = {}
            # 干预统计：每个请求的 total token steps 和实际干预次数
            self._total_steps: dict[int, int] = {}
            self._intervened_steps: dict[int, int] = {}

        # ----------------------------------------------------------------
        # 从 prompt token IDs 中提取 user 内容
        # ----------------------------------------------------------------
        def _compute_sentinel_bounds(self):
            """
            启动时调用一次：用哨兵字符串推算 user turn 的 prefix / suffix token 序列。
            成功返回 (prefix, suffix) list；tokenizer 无 chat_template 或推算失败返回 (None, None)。
            """
            SENTINEL = "XSIASENTINELX"
            try:
                if not getattr(self._llm_tok, 'chat_template', None):
                    return None, None
                test_ids = self._llm_tok.apply_chat_template(
                    [{"role": "user", "content": SENTINEL}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                sentinel_ids = self._llm_tok.encode(SENTINEL, add_special_tokens=False)
                test_list = list(test_ids)
                sentinel_list = list(sentinel_ids)
                sentinel_pos = next(
                    (i for i in range(len(test_list) - len(sentinel_list) + 1)
                     if test_list[i:i + len(sentinel_list)] == sentinel_list),
                    None,
                )
                if sentinel_pos is None:
                    return None, None
                prefix = test_list[:sentinel_pos]
                suffix = test_list[sentinel_pos + len(sentinel_list):]
                if not prefix or not suffix:
                    return None, None
                return prefix, suffix
            except Exception:
                return None, None

        def _extract_user_content(self, prompt_ids: list) -> str:
            """
            主方案：在 token ID 层用预计算的 prefix/suffix 定位最后一个 user turn，
            不依赖格式假设，适用于任何有 chat_template 的 tokenizer。
            回退方案：基于正则的文本解析（覆盖 Llama 2 / Gemma / Human: 格式）。
            两种方案均无声切换，不打印警告。
            """
            prefix = self._sentinel_prefix
            suffix = self._sentinel_suffix
            if prefix is not None and suffix is not None:
                prompt_list = list(prompt_ids)
                # 最后一次出现 prefix（处理多轮对话）
                last_prefix_pos = next(
                    (i for i in range(len(prompt_list) - len(prefix), -1, -1)
                     if prompt_list[i:i + len(prefix)] == prefix),
                    None,
                )
                if last_prefix_pos is not None:
                    content_start = last_prefix_pos + len(prefix)
                    suffix_pos = next(
                        (i for i in range(content_start, len(prompt_list) - len(suffix) + 1)
                         if prompt_list[i:i + len(suffix)] == suffix),
                        None,
                    )
                    if suffix_pos is not None:
                        user_ids = prompt_list[content_start:suffix_pos]
                        content = self._llm_tok.decode(
                            user_ids, skip_special_tokens=True
                        ).strip()
                        if content:
                            return content

            # Fallback：文本解析（raw string prompt / 无 chat_template 时的正常路径）
            prompt_text = self._llm_tok.decode(prompt_ids, skip_special_tokens=True)
            return extract_user_content(prompt_text)

        # ----------------------------------------------------------------
        # RM 打分：批量发送到 RM server，一次 HTTP 调用
        # ----------------------------------------------------------------
        def _score_candidates(
            self,
            user_content: str,
            response_so_far: str,
            candidate_token_ids: list[int],
        ) -> torch.Tensor:
            """返回 shape (topk,) float32 tensor（CPU）。"""
            candidate_texts = [
                self._llm_tok.decode([tid], skip_special_tokens=False)
                for tid in candidate_token_ids
            ]
            resp = self._rm_session.post(
                f"{self._RM_URL}/score",
                json={
                    "user_content": user_content,
                    "response_so_far": response_so_far,
                    "candidate_texts": candidate_texts,
                },
                timeout=30,
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
            return torch.tensor(scores, dtype=torch.float32)

        # ----------------------------------------------------------------
        # apply：每个 token 生成前被 vllm 调用一次
        # ----------------------------------------------------------------
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            batch_size = logits.shape[0]

            for i in range(batch_size):
                output_ids = list(self._output_ids.get(i, []))
                user_content = self._prompt_user.get(i, "")

                self._total_steps[i] = self._total_steps.get(i, 0) + 1
                req_step = self._total_steps[i]

                topk_logits, topk_indices = torch.topk(logits[i], self._TOPK)

                # entropy 过滤
                probs = F.softmax(topk_logits.float(), dim=-1)
                entropy = dist.Categorical(probs=probs).entropy().item()
                if self._ENTROPY_THRESHOLD is not None:
                    if entropy < self._ENTROPY_THRESHOLD:
                        print(
                            f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
                            f"step={req_step:3d} req={i} "
                            f"SKIP (entropy={entropy:.3f} < {self._ENTROPY_THRESHOLD})",
                            flush=True,
                        )
                        continue

                response_so_far = self._llm_tok.decode(
                    output_ids, skip_special_tokens=True
                )

                try:
                    rm_scores = self._score_candidates(
                        user_content, response_so_far, topk_indices.tolist()
                    )
                except Exception as e:
                    print(
                        f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
                        f"step={req_step:3d} req={i} RM error: {e}",
                        flush=True,
                    )
                    continue

                # 归一化：减均值，使得 topk 内有相对排序，
                # 避免全负分时把 topk 全部压低、让 topk 外 token 意外胜出
                rm_scores = rm_scores - rm_scores.mean()
                effective_weight = self._weight_per_req.get(i, self._WEIGHT)
                logits[i, topk_indices] = (
                    logits[i, topk_indices]
                    + rm_scores.to(logits.device) * effective_weight
                )

                self._intervened_steps[i] = self._intervened_steps.get(i, 0) + 1

                print(
                    f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
                    f"step={req_step:3d} req={i} INTERVENE "
                    f"entropy={entropy:.3f} "
                    f"gen_len={len(output_ids)} "
                    f"rm=[{rm_scores.min():.3f}, {rm_scores.max():.3f}]",
                    flush=True,
                )

            return logits

        def is_argmax_invariant(self) -> bool:
            return False

        # ----------------------------------------------------------------
        # update_state
        # ----------------------------------------------------------------
        def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
            if batch_update is None:
                return

            for idx in batch_update.removed:
                total = self._total_steps.pop(idx, 0)
                intervened = self._intervened_steps.pop(idx, 0)
                ratio = intervened / total if total > 0 else 0.0
                print(
                    f"[SIA] req={idx} DONE  "
                    f"intervened={intervened}/{total}  ratio={ratio:.1%}",
                    flush=True,
                )
                self._output_ids.pop(idx, None)
                self._prompt_user.pop(idx, None)
                self._weight_per_req.pop(idx, None)

            if batch_update.moved:
                old_out = dict(self._output_ids)
                old_prompt = dict(self._prompt_user)
                old_weight = dict(self._weight_per_req)
                old_total = dict(self._total_steps)
                old_intervened = dict(self._intervened_steps)
                for i1, i2, directionality in batch_update.moved:
                    if directionality == MoveDirectionality.UNIDIRECTIONAL:
                        self._output_ids[i2] = old_out.get(i1, [])
                        self._prompt_user[i2] = old_prompt.get(i1, "")
                        if i1 in old_weight:
                            self._weight_per_req[i2] = old_weight[i1]
                        self._total_steps[i2] = old_total.get(i1, 0)
                        self._intervened_steps[i2] = old_intervened.get(i1, 0)
                    else:  # SWAP
                        self._output_ids[i1] = old_out.get(i2, [])
                        self._output_ids[i2] = old_out.get(i1, [])
                        self._prompt_user[i1] = old_prompt.get(i2, "")
                        self._prompt_user[i2] = old_prompt.get(i1, "")
                        if i2 in old_weight:
                            self._weight_per_req[i1] = old_weight[i2]
                        else:
                            self._weight_per_req.pop(i1, None)
                        if i1 in old_weight:
                            self._weight_per_req[i2] = old_weight[i1]
                        else:
                            self._weight_per_req.pop(i2, None)
                        self._total_steps[i1] = old_total.get(i2, 0)
                        self._total_steps[i2] = old_total.get(i1, 0)
                        self._intervened_steps[i1] = old_intervened.get(i2, 0)
                        self._intervened_steps[i2] = old_intervened.get(i1, 0)

            for idx, params, prompt_ids, output_ids in batch_update.added:
                self._output_ids[idx] = output_ids
                prompt_text = self._llm_tok.decode(prompt_ids, skip_special_tokens=True)
                sia_weight, prompt_text_clean = _parse_sia_header(prompt_text)
                if sia_weight is not None:
                    self._weight_per_req[idx] = sia_weight
                self._prompt_user[idx] = extract_user_content(prompt_text_clean)

    return SIALogitsProcessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA vllm 干预（HTTP RM Server 版本）")
    p.add_argument("--llm",       required=True,  help="LLM 模型路径")
    p.add_argument("--rm_url",    default="http://localhost:8001",
                   help="RM server 地址（默认 http://localhost:8001）")
    p.add_argument("--llm_gpu_mem", type=float, default=0.5,
                   help="vllm gpu_memory_utilization（默认 0.5）")
    p.add_argument("--topk",      type=int,   default=10)
    p.add_argument("--weight",    type=float, default=1.0)
    p.add_argument("--max_tokens", type=int,  default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--entropy_threshold", type=float, default=None)
    p.add_argument("--prompt", type=str,
                   default="Human:\nTell me a joke.\nAssistant:\n")
    p.add_argument("--max_model_len", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"LLM      : {args.llm}")
    print(f"RM URL   : {args.rm_url}")
    print(f"topk={args.topk}  weight={args.weight}  "
          f"entropy_threshold={args.entropy_threshold}")
    print("=" * 60)

    SIAProcessor = make_sia_processor(
        rm_url=args.rm_url,
        topk=args.topk,
        weight=args.weight,
        entropy_threshold=args.entropy_threshold,
    )

    print("Loading vllm LLM...")
    llm = LLM(
        model=args.llm,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.llm_gpu_mem,
        logits_processors=[SIAProcessor],
    )
    print("LLM loaded.\n")

    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
    )

    print("=" * 60)
    print(f"Prompt: {repr(args.prompt)}")
    print("=" * 60)
    print("\n[Generated]")
    print(outputs[0].outputs[0].text)
    print("=" * 60)


if __name__ == "__main__":
    main()

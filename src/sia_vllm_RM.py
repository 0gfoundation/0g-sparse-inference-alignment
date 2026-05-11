"""
SIA per-token 干预（接入真实 RM）- vLLM v1 engine 版本

架构：
  - LLM: vllm (v1 engine) 负责生成，GPU 由 vllm 管理
  - RM:  HuggingFace AutoModelForSequenceClassification（可选 LoRA），在 apply() 里同步调用
  - SIALogitsProcessor 运行在 EngineCore 子进程中，通过 LLM(logits_processors=[...]) 注册

干预逻辑（per-token）：
  1. 从 LLM logits 中取 top-k candidate token IDs
  2. 将当前生成序列 + 各 candidate 解码成文本
  3. 用 RM tokenizer apply_chat_template 逐个格式化并送入 RM 打分
  4. combined_logits[topk_indices] += rm_scores * weight
  5. 返回 combined_logits

用法示例：
  # 不带 LoRA
  python sia_vllm_RM.py \\
    --llm  /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm   /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_device cuda:0 --llm_gpu_mem 0.3 --topk 5 --weight 0.1 --max_tokens 20

  # 带 LoRA（ValueModel checkpoint）
  python sia_vllm_RM.py \\
    --llm     /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm      /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_lora /workspace/SIA/models/SIA-checkpoints/VM-Qwen3-1.7B-Base \\
    --rm_device cuda:0 --llm_gpu_mem 0.3 --topk 5 --weight 1.0 --max_tokens 64
"""

import argparse
import os
import re
from typing import Optional

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


# ---------------------------------------------------------------------------
# ValueModel 轻量包装（仅在 --rm_lora 时使用）
# 复现 src/value_model/model.py 中 ValueModel 的推理路径，
# 不依赖 SIA 源码，避免跨模块 import 问题。
# ---------------------------------------------------------------------------

class _ValueModelOutput:
    def __init__(self, logits):
        self.logits = logits  # (batch, 1)


class _ValueModelWrapper(nn.Module):
    """
    加载 ValueModel checkpoint（base RM + LoRA + token_reward_head）后的推理封装。
    forward() 返回与 AutoModelForSequenceClassification 兼容的输出对象（.logits）。
    """

    def __init__(self, base_model, token_reward_head: nn.Linear):
        super().__init__()
        self.base_model = base_model
        self.token_reward_head = token_reward_head
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # 取 transformer backbone（跳过分类头）
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
            hidden = out.last_hidden_state  # (batch, seq, hidden)

        token_rewards = self.token_reward_head(hidden.float()).squeeze(-1)  # (batch, seq)
        logits = token_rewards[:, -1].unsqueeze(-1)                         # (batch, 1)
        return _ValueModelOutput(logits=logits)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

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
    rm_path: str,
    rm_lora_path: Optional[str] = None,
    topk: int = 10,
    weight: float = 1.0,
    rm_device: str = "cuda:0",
    entropy_threshold: Optional[float] = None,
):
    """
    返回一个 SIALogitsProcessor 类（不是实例）。
    vllm 引擎以 (vllm_config, device, is_pin_memory) 参数实例化该类。
    """

    class SIALogitsProcessor(LogitsProcessor):
        _RM_PATH = rm_path
        _RM_LORA_PATH = rm_lora_path
        _TOPK = topk
        _WEIGHT = weight
        _RM_DEVICE = rm_device
        _ENTROPY_THRESHOLD = entropy_threshold

        # ----------------------------------------------------------------
        # 初始化：在 EngineCore 子进程里执行
        # ----------------------------------------------------------------
        def __init__(self, vllm_config, device: torch.device,
                     is_pin_memory: bool) -> None:
            self._llm_device = device

            # LLM tokenizer
            llm_model_path = vllm_config.model_config.model
            print(f"[SIA] Loading LLM tokenizer: {llm_model_path}", flush=True)
            self._llm_tok = AutoTokenizer.from_pretrained(
                llm_model_path, trust_remote_code=True
            )

            # RM tokenizer
            print(f"[SIA] Loading RM tokenizer: {self._RM_PATH}", flush=True)
            self._rm_tok = AutoTokenizer.from_pretrained(
                self._RM_PATH, trust_remote_code=True
            )

            # RM model
            if self._RM_LORA_PATH:
                self._rm_model = self._load_rm_with_lora()
            else:
                self._rm_model = self._load_rm_base()

            self._rm_model.eval()
            print("[SIA] RM loaded.", flush=True)

            # per-request 状态
            self._prompt_user: dict[int, str] = {}
            self._output_ids: dict[int, list] = {}
            self._step = 0

        def _load_rm_base(self):
            """加载普通 AutoModelForSequenceClassification RM。"""
            print(f"[SIA] Loading RM (base): {self._RM_PATH}  device={self._RM_DEVICE}",
                  flush=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._RM_PATH,
                num_labels=1,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self._RM_DEVICE)
            # 设置 pad_token_id，否则 sequence classification pooler 报错
            model.config.pad_token_id = self._rm_tok.eos_token_id
            return model

        def _load_rm_with_lora(self):
            """
            加载 ValueModel checkpoint：base RM + LoRA + token_reward_head。
            checkpoint 目录结构（与 src/value_model/model.py 保持一致）：
              {rm_lora_path}/
                lora_weights/        # PeftModel adapter
                token_reward_head.pt # {'token_reward_head': state_dict}
                model_config.json    # 可选
            """
            from peft import PeftModel

            print(f"[SIA] Loading RM base for LoRA: {self._RM_PATH}", flush=True)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self._RM_PATH,
                num_labels=1,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            lora_dir = os.path.join(self._RM_LORA_PATH, "lora_weights")
            print(f"[SIA] Loading LoRA from: {lora_dir}", flush=True)
            base_model = PeftModel.from_pretrained(base_model, lora_dir)
            base_model = base_model.merge_and_unload()

            hidden_size = base_model.config.hidden_size
            token_reward_head = nn.Linear(hidden_size, 1)  # float32，与 checkpoint 一致
            head_path = os.path.join(self._RM_LORA_PATH, "token_reward_head.pt")
            print(f"[SIA] Loading token_reward_head from: {head_path}", flush=True)
            head_state = torch.load(head_path, map_location='cpu', weights_only=True)
            token_reward_head.load_state_dict(head_state['token_reward_head'])

            model = _ValueModelWrapper(base_model, token_reward_head).to(self._RM_DEVICE)
            return model

        # ----------------------------------------------------------------
        # 从 prompt token IDs 中提取 user 内容
        # ----------------------------------------------------------------
        def _extract_user_content(self, prompt_ids: list) -> str:
            """
            主方案：用 apply_chat_template 注入哨兵字符串，在 token ID 层定位
            最后一个 user turn 的边界，不依赖任何格式假设。
            适用于任何带有 chat_template 的 tokenizer（Qwen3 / Llama 3 / Mistral v3 等）。
            回退方案：基于正则的文本解析（覆盖 Llama 2 / Gemma / Human: 格式）。
            """
            SENTINEL = "XSIASENTINELX"
            try:
                if not getattr(self._llm_tok, 'chat_template', None):
                    raise ValueError("no chat_template")

                test_ids = self._llm_tok.apply_chat_template(
                    [{"role": "user", "content": SENTINEL}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                sentinel_ids = self._llm_tok.encode(SENTINEL, add_special_tokens=False)

                test_list = list(test_ids)
                sentinel_list = list(sentinel_ids)

                # 找哨兵在 test_ids 中的位置，确定 prefix / suffix
                sentinel_pos = next(
                    (i for i in range(len(test_list) - len(sentinel_list) + 1)
                     if test_list[i:i + len(sentinel_list)] == sentinel_list),
                    None,
                )
                if sentinel_pos is None:
                    raise ValueError("sentinel not found in template output")

                prefix = test_list[:sentinel_pos]
                suffix = test_list[sentinel_pos + len(sentinel_list):]
                if not suffix:
                    raise ValueError("empty suffix — cannot determine user content end")

                prompt_list = list(prompt_ids)

                # 找 prefix 在 prompt_ids 中最后一次出现（处理多轮对话）
                last_prefix_pos = next(
                    (i for i in range(len(prompt_list) - len(prefix), -1, -1)
                     if prompt_list[i:i + len(prefix)] == prefix),
                    None,
                )
                if last_prefix_pos is None:
                    raise ValueError("prefix not found in prompt_ids")

                content_start = last_prefix_pos + len(prefix)

                # 找 suffix 紧随其后的位置
                suffix_pos = next(
                    (i for i in range(content_start, len(prompt_list) - len(suffix) + 1)
                     if prompt_list[i:i + len(suffix)] == suffix),
                    None,
                )
                if suffix_pos is None:
                    raise ValueError("suffix not found after prefix")

                user_ids = prompt_list[content_start:suffix_pos]
                content = self._llm_tok.decode(user_ids, skip_special_tokens=True).strip()
                if content:
                    return content
                raise ValueError("decoded user content is empty")

            except Exception as e:
                print(f"[SIA] _extract_user_content sentinel failed ({e}), fallback to text parsing",
                      flush=True)

            prompt_text = self._llm_tok.decode(prompt_ids, skip_special_tokens=True)
            return extract_user_content(prompt_text)

        # ----------------------------------------------------------------
        # RM 打分：逐个 candidate forward，避免 fast tokenizer padding 问题
        # ----------------------------------------------------------------
        def _score_candidates(
            self,
            user_content: str,
            response_so_far: str,
            candidate_token_ids: list[int],
        ) -> torch.Tensor:
            """返回 shape (topk,) float32 tensor（CPU）。"""
            bos = self._rm_tok.bos_token
            scores = []
            for tid in candidate_token_ids:
                cand_text = self._llm_tok.decode([tid], skip_special_tokens=False)
                response_with_cand = response_so_far + cand_text
                convs = [
                    {'role': 'user',      'content': user_content},
                    {'role': 'assistant', 'content': response_with_cand},
                ]
                rm_text = self._rm_tok.apply_chat_template(convs, tokenize=False)
                if bos and rm_text.startswith(bos):
                    rm_text = rm_text[len(bos):]

                encoded = self._rm_tok(
                    rm_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=2048,
                ).to(self._RM_DEVICE)

                with torch.no_grad():
                    rm_out = self._rm_model(**encoded)
                scores.append(rm_out.logits.flatten()[0].item())
                del rm_out, encoded

            return torch.tensor(scores, dtype=torch.float32)

        # ----------------------------------------------------------------
        # apply：每个 token 生成前被 vllm 调用一次
        # ----------------------------------------------------------------
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            self._step += 1
            batch_size = logits.shape[0]

            for i in range(batch_size):
                output_ids = list(self._output_ids.get(i, []))
                user_content = self._prompt_user.get(i, "")

                topk_logits, topk_indices = torch.topk(logits[i], self._TOPK)

                # entropy 过滤
                if self._ENTROPY_THRESHOLD is not None:
                    probs = F.softmax(topk_logits.float(), dim=-1)
                    entropy = dist.Categorical(probs=probs).entropy().item()
                    if entropy < self._ENTROPY_THRESHOLD:
                        print(
                            f"[SIA] step={self._step:3d} req={i} "
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
                    print(f"[SIA] step={self._step:3d} req={i} RM error: {e}",
                          flush=True)
                    continue

                # 归一化：减均值，使得 topk 内有相对排序，
                # 避免全负分时把 topk 全部压低、让 topk 外 token 意外胜出
                rm_scores = rm_scores - rm_scores.mean()
                logits[i, topk_indices] = (
                    logits[i, topk_indices]
                    + rm_scores.to(logits.device) * self._WEIGHT
                )

                print(
                    f"[SIA] step={self._step:3d} req={i} "
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
                self._output_ids.pop(idx, None)
                self._prompt_user.pop(idx, None)

            if batch_update.moved:
                old_out = dict(self._output_ids)
                old_prompt = dict(self._prompt_user)
                for i1, i2, directionality in batch_update.moved:
                    if directionality == MoveDirectionality.UNIDIRECTIONAL:
                        self._output_ids[i2] = old_out.get(i1, [])
                        self._prompt_user[i2] = old_prompt.get(i1, "")
                    else:  # SWAP
                        self._output_ids[i1] = old_out.get(i2, [])
                        self._output_ids[i2] = old_out.get(i1, [])
                        self._prompt_user[i1] = old_prompt.get(i2, "")
                        self._prompt_user[i2] = old_prompt.get(i1, "")

            for idx, params, prompt_ids, output_ids in batch_update.added:
                self._output_ids[idx] = output_ids
                self._prompt_user[idx] = self._extract_user_content(prompt_ids)

    return SIALogitsProcessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA vllm 干预（真实 RM，可选 LoRA）")
    p.add_argument("--llm",       required=True,  help="LLM 模型路径")
    p.add_argument("--rm",        required=True,  help="RM 基础模型路径")
    p.add_argument("--rm_lora",   default=None,
                   help="RM LoRA checkpoint 目录（ValueModel 格式，含 lora_weights/ 和 token_reward_head.pt）")
    p.add_argument("--rm_device", default="cuda:0")
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
    print(f"RM       : {args.rm}  (device={args.rm_device})")
    if args.rm_lora:
        print(f"RM LoRA  : {args.rm_lora}")
    print(f"topk={args.topk}  weight={args.weight}  "
          f"entropy_threshold={args.entropy_threshold}")
    print("=" * 60)

    SIAProcessor = make_sia_processor(
        rm_path=args.rm,
        rm_lora_path=args.rm_lora,
        topk=args.topk,
        weight=args.weight,
        rm_device=args.rm_device,
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

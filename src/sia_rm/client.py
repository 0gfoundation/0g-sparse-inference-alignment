"""
RMClient: in-process stateful Reward Model client for SIA.

Wraps a vLLM `LLM` instance (synchronous) with session state for SIA's
incremental decode pattern:

    rm = RMClient(model_path=...)
    sid = rm.new_session(prompt_token_ids)
    # for each SIA-generated token:
    rm.fix_a_token(sid, generated_token_id)
    # whenever SIA wants to score candidates:
    rewards = rm.score_candidates(sid, [t1, t2, t3, t4, t5])

The session prefix is just a Python list of token ids. We rely on vLLM's
automatic prefix caching for KV reuse — `fix_a_token` does NOT call vLLM,
it only appends to the prefix; the next `score_candidates` call sends
prompts of the form `prefix + [candidate]`, and vLLM's prefix cache makes
the per-call forward effectively decode-only (~7-15ms for batch=5).

Reward retrieval: vLLM EngineCore is a separate subprocess, so the score
head writes its output to a `/dev/shm` file. We set
`SIA_REWARD_FILE_ID` to a unique value per RMClient instance to avoid
collisions across concurrent clients.

NOTE: vLLM may split a single `generate()` call into multiple compute_logits
invocations (e.g. one prompt prefills, then the remaining N-1 batch
together). The score head writes one record per call (append mode); we
concatenate all records and assume the order matches the prompt order. This
held in step 1 verify (5 prompts → 2 records in [1, 4] split, concat
recovered the original [0..4] order). If a future vLLM version reorders,
the assert in `score_candidates` will catch it.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from .qwen3_with_score import (
    read_all_rewards,
    read_rewards,
    truncate_rewards,
)


class RMClient:
    # X-7: vLLM 默认 cuda graph capture sizes 步长为 [1,2,4,8,16,...]，
    # 不包含 5；SIA topk=5 时, score_candidates 用 batch=5 prompts,
    # 落到 batch=8 graph 要 padding 3 个 dummy token, 多算一些 forward。
    # 这里显式列出包含 5 的 sizes; 用法是 list (多个值时直接采用), 单值会
    # 走 vLLM 内部 [1,2,4]+range(8,N+1,8) 推断（坑 #98 vllm-rm-experiment-
    # report.md §3.1 已踩过）。
    _DEFAULT_CUDA_GRAPH_SIZES: list = [1, 2, 4, 5, 8, 16, 32, 64, 128]

    def __init__(
        self,
        model_path: str,
        gpu_mem: float = 0.3,
        max_model_len: int = 4096,
        reward_file_id: Optional[str] = None,
        cuda_graph_sizes: Optional[list] = None,
    ):
        # Unique reward file id per instance — must be set BEFORE we
        # import vllm (the env var is read by qwen3_with_score in the
        # EngineCore subprocess at compute_logits time, so as long as the
        # var is set before generate() it's fine; but setting it before
        # LLM() init is cleanest)
        self._fid = reward_file_id or f"client_{uuid.uuid4().hex[:8]}"
        os.environ["SIA_REWARD_FILE_ID"] = self._fid

        # Trigger ModelRegistry.register_model (idempotent)
        from . import qwen3_with_score  # noqa: F401

        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            enforce_eager=False,
            disable_log_stats=True,
            cuda_graph_sizes=(
                self._DEFAULT_CUDA_GRAPH_SIZES if cuda_graph_sizes is None
                else cuda_graph_sizes
            ),
        )
        self._sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
        )

        # Session state
        self._sessions: dict[int, list[int]] = {}
        self._next_id = 0

        # Clean any stale reward file from previous runs
        truncate_rewards()

    # ---- session management ----

    def new_session(self, prompt_token_ids) -> int:
        sid = self._next_id
        self._next_id += 1
        self._sessions[sid] = list(prompt_token_ids)
        return sid

    def fix_a_token(self, sid: int, token_id: int) -> None:
        """Commit a token to the session prefix. Does NOT call vLLM."""
        if sid not in self._sessions:
            raise ValueError(f"Unknown session id {sid}")
        self._sessions[sid].append(int(token_id))

    def end_session(self, sid: int) -> None:
        self._sessions.pop(sid, None)

    def session_length(self, sid: int) -> int:
        return len(self._sessions[sid])

    # ---- scoring ----

    def score_candidates(
        self,
        sid: int,
        candidate_token_ids,
    ) -> list[float]:
        """Score N candidates by running forward on N prompts (prefix+[c])
        and reading the score head outputs. Returns N floats."""
        if sid not in self._sessions:
            raise ValueError(f"Unknown session id {sid}")
        prefix = self._sessions[sid]
        from vllm import TokensPrompt

        prompts = [
            TokensPrompt(prompt_token_ids=prefix + [int(c)])
            for c in candidate_token_ids
        ]
        n_expected = len(candidate_token_ids)

        # Clear reward channel before this call, then generate, then read.
        truncate_rewards()
        _ = self.llm.generate(prompts, self._sp, use_tqdm=False)
        rewards = read_rewards()  # concatenates all records

        if rewards is None or rewards.numel() != n_expected:
            n_got = 0 if rewards is None else int(rewards.numel())
            records = read_all_rewards()
            shapes = [tuple(r.shape) for r in records]
            raise RuntimeError(
                f"reward channel returned {n_got} values, expected "
                f"{n_expected}. records seen: {shapes}. "
                f"Likely SIA_REWARD_FILE_ID mismatch or vLLM reordered "
                f"prompts."
            )
        return rewards.tolist()

    # ---- bench / introspection ----

    def time_score_candidates(
        self,
        sid: int,
        candidate_token_ids,
    ) -> tuple[list[float], float]:
        """Like score_candidates but also returns wall-clock latency in ms."""
        t0 = time.perf_counter()
        out = self.score_candidates(sid, candidate_token_ids)
        return out, (time.perf_counter() - t0) * 1000.0

    def __del__(self):
        try:
            truncate_rewards()
        except Exception:
            pass

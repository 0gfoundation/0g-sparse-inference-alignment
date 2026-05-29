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
    read_all_timings,
    truncate_timings,
)


class RMClient:
    def __init__(
        self,
        model_path: str,
        gpu_mem: float = 0.3,
        max_model_len: int = 4096,
        reward_file_id: Optional[str] = None,
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

        from vllm import LLM, SamplingParams, TokensPrompt

        # X-5: cache TokensPrompt class ref to avoid re-importing in hot path
        # (sub-microsecond per call but 100% deterministic, zero risk)
        self._TokensPrompt = TokensPrompt

        self.llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            enforce_eager=False,
            disable_log_stats=True,
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

        # Profile-mode state: how many tokens of each session's prefix have
        # already been pushed through a vLLM forward (thus KV-cached).
        # Used by score_candidates_profiled to know how many new tokens
        # need a separate "prefill" generate call.
        self._last_prefilled: dict[int, int] = {}

        # Clean any stale reward + timing files from previous runs
        truncate_rewards()
        truncate_timings()

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
        TP = self._TokensPrompt  # X-5: local alias avoids self.__dict__ lookup × N

        # X-5: candidate_token_ids comes from SIA processor topk (list[int]),
        # already Python ints — skip redundant `int(c)` cast.
        prompts = [
            TP(prompt_token_ids=prefix + [c])
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

    # ---- split-stage profiling for parallelization analysis ----

    def score_candidates_profiled(
        self,
        sid: int,
        candidate_token_ids,
    ) -> dict:
        """Profile-mode score_candidates: splits the work into two
        separate llm.generate() calls so each stage can be timed
        independently. Designed to feed a "can we overlap RM with main
        LLM forward?" analysis.

        Stage A — "non-intervention KV":
            Generate(prefix_only). vLLM uses prefix caching to skip
            already-processed tokens and only forwards the K tokens
            added since the last RM call (the SKIP-token tail). The
            forward time is the cost of catching the RM KV cache up
            with the latest LLM-generated tokens — this is the work
            that could potentially overlap with the main LLM forward.

        Stage B — "5 candidate KV + score":
            Generate(prefix+[c_i] for each c in candidates). All
            prefix KV is now cached (after Stage A) so vLLM only
            forwards the 5 candidate tokens. The score head matmul is
            timed separately inside compute_logits via cuda.Event.

        Requires SIA_RM_PROFILE=1 env var when launching vLLM, otherwise
        the timing channel is not populated and GPU-side stats will be
        zero (wall-clock is always returned).

        Returns dict with all stage times:
            {
              'rewards': list[float],          # the 5 reward scalars
              'n_new_prefix': int,             # K (tokens forwarded in Stage A)
              'n_candidates': int,             # always 5
              # Wall-clock (client side, includes vLLM dispatch overhead)
              'a_wall_ms': float,
              'b_wall_ms': float,
              # GPU time (from cuda.Event inside compute_logits)
              'a_forward_gpu_ms': float,       # KV computation for K new tokens
              'b_forward_gpu_ms': float,       # KV computation for 5 candidates
              'b_score_gpu_ms': float,         # score head matmul on 5 hidden
            }
        """
        if sid not in self._sessions:
            raise ValueError(f"Unknown session id {sid}")
        prefix = self._sessions[sid]
        TP = self._TokensPrompt

        # Determine how many tokens of prefix vLLM has already KV-cached
        # for this session. First call: none cached, so Stage A processes
        # the entire prefix. Subsequent calls: only tokens added since
        # last call (fix_a_token chain).
        n_already = self._last_prefilled.get(sid, 0)
        n_new = len(prefix) - n_already

        # Clear the side channels so the timings we read are only from
        # this score call.
        truncate_rewards()
        truncate_timings()

        # ---- Stage A: prefill new prefix tokens (KV cache catchup) ----
        a_wall_ms = 0.0
        if n_new > 0:
            a_t0 = time.perf_counter()
            _ = self.llm.generate(
                [TP(prompt_token_ids=prefix)],
                self._sp, use_tqdm=False,
            )
            a_wall_ms = (time.perf_counter() - a_t0) * 1000.0
        # Snapshot timing records from Stage A (forward_ms + score_ms,
        # but score_ms here is on the *single* prefix prompt, not useful)
        a_timings = read_all_timings()
        a_forward_gpu_ms = sum(t['forward_ms'] for t in a_timings)

        # ---- Stage B: forward 5 candidates (prefix is now KV-cached) ----
        # Truncate rewards again so Stage B's reward records start fresh.
        # (Stage A's reward record is the dummy 1-prompt one, not useful.)
        truncate_rewards()
        # Note: don't truncate timings here — we want to subtract Stage A
        # timings from total to get Stage B incremental. But simpler: take
        # all timings after Stage A.
        n_timings_after_a = len(a_timings)

        b_t0 = time.perf_counter()
        prompts = [TP(prompt_token_ids=prefix + [c])
                   for c in candidate_token_ids]
        _ = self.llm.generate(prompts, self._sp, use_tqdm=False)
        b_wall_ms = (time.perf_counter() - b_t0) * 1000.0

        all_timings = read_all_timings()
        b_timings = all_timings[n_timings_after_a:]
        b_forward_gpu_ms = sum(t['forward_ms'] for t in b_timings)
        # Score head is only meaningful on the 5-sample call; sum across
        # all forward passes that wrote a record in Stage B.
        b_score_gpu_ms = sum(t['score_ms'] for t in b_timings)

        rewards = read_rewards()
        n_expected = len(candidate_token_ids)
        if rewards is None or rewards.numel() != n_expected:
            raise RuntimeError(
                f"reward channel returned "
                f"{0 if rewards is None else rewards.numel()} "
                f"values, expected {n_expected} (Stage B). "
                f"Likely SIA_REWARD_FILE_ID mismatch or vLLM reordered."
            )

        # Update prefix-prefill bookkeeping. Stage B forwarded prefix
        # + 1 candidate token per prompt; the candidate tokens are NOT
        # part of the session prefix (they are scoring branches that
        # get dropped). So the session prefix is still len(prefix) and
        # all of it is now KV-cached.
        self._last_prefilled[sid] = len(prefix)

        return {
            'rewards': rewards.tolist(),
            'n_new_prefix': n_new,
            'n_candidates': n_expected,
            'a_wall_ms': a_wall_ms,
            'b_wall_ms': b_wall_ms,
            'a_forward_gpu_ms': a_forward_gpu_ms,
            'b_forward_gpu_ms': b_forward_gpu_ms,
            'b_score_gpu_ms': b_score_gpu_ms,
        }

    def __del__(self):
        try:
            truncate_rewards()
        except Exception:
            pass

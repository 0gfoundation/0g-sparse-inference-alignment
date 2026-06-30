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

import torch

from .qwen3_with_score import (
    read_all_rewards,
    read_rewards,
    truncate_rewards,
    read_all_timings,
    truncate_timings,
    set_inproc_reward_mode,
    clear_inproc_rewards,
    take_inproc_rewards,
    _REWARD_BUFFERS,
)


class RMClient:
    def __init__(
        self,
        model_path: str,
        gpu_mem: float = 0.3,
        max_model_len: int = 4096,
        reward_file_id: Optional[str] = None,
        cuda_graph_sizes: Optional[list] = None,
        multiprocessing: bool = False,
        llm_tokenizer=None,
        max_num_batched_tokens: Optional[int] = None,
    ):
        """
        cuda_graph_sizes: optional list of batch sizes to capture in the
        vLLM CUDA graph. When None (default), vLLM uses its built-in
        default ([1, 2, 4, 8, 16, ...]) which does NOT include batch=5 —
        SIA's typical topk=5 will then be padded up to the batch=8 graph.
        Pass e.g. [1, 2, 4, 5, 8, 16, 32, 64, 128] to get a dedicated
        batch=5 graph.

        multiprocessing: when False (default for SIA), force vLLM v1 to use
        InprocClient (EngineCore in this process; no ZMQ; no busy-loop
        subprocess). For SIA's pattern of submitting 5 shared-prefix prompts
        per intervention, this eliminates the 1+4 split that the
        multiprocess client produces — when EngineCoreProc runs in a
        separate subprocess, the 5 add_request calls serialize over ZMQ,
        and req#1 arrives at the EngineCore before req#2-5 do, triggering
        a single-prompt scheduler step on req#1 alone. InprocClient adds
        all 5 requests synchronously before _run_engine starts, so the
        scheduler picks up all 5 in one schedule() and runs a single
        batch=5 forward (measured: Stage B p50 4.78 ms vs 13.29 ms,
        wall-clock p50 9.04 ms vs 23.96 ms — see doc/inproc-vs-mp.md).
        """
        # InprocClient must be enabled via env var BEFORE vllm is imported
        # (LLMEngine.from_engine_args reads VLLM_ENABLE_V1_MULTIPROCESSING
        # to decide whether to spawn EngineCoreProc).
        if not multiprocessing:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        self._multiprocessing = multiprocessing

        # Unique reward file id per instance — must be set BEFORE we
        # import vllm (the env var is read by qwen3_with_score at
        # compute_logits time. In multiprocess mode this is the EngineCore
        # subprocess; in inproc mode it's this process. Either way the
        # mechanism still works since /dev/shm is process-agnostic.)
        self._fid = reward_file_id or f"client_{uuid.uuid4().hex[:8]}"
        os.environ["SIA_REWARD_FILE_ID"] = self._fid

        # Trigger ModelRegistry.register_model (idempotent)
        from . import qwen3_with_score  # noqa: F401

        # B-1: select reward channel BEFORE LLM(...) runs cudagraph capture.
        # capture invokes compute_logits with dummy inputs at every cuda
        # graph batch size; those calls must already see the correct flag.
        set_inproc_reward_mode(not multiprocessing)

        from vllm import LLM, SamplingParams, TokensPrompt

        # X-5: cache TokensPrompt class ref to avoid re-importing in hot path
        # (sub-microsecond per call but 100% deterministic, zero risk)
        self._TokensPrompt = TokensPrompt

        # vllm 0.19 cudagraph 冲突 workaround.
        # 主 LLM 用 vllm 0.19 默认 FULL_AND_PIECEWISE cudagraph mode;
        # RM 跟主 LLM 同进程同 stream, RM 也用 FULL cudagraph 会冲突:
        #   "CUDA graph capturing detected at an inappropriate time"
        # 三种 RM 配置 (env var):
        #   SIA_RM_CUDAGRAPH=full  (默认 0.19 行为)  — 跟主 LLM 冲突, 失败
        #   SIA_RM_CUDAGRAPH=piecewise  (推荐 for vllm 0.19) — 只 capture splitting ops,
        #                                跟主 LLM FULL 不冲突, 大部分加速保留 (~80%)
        #   SIA_RM_CUDAGRAPH=none  (eager) — 没 cudagraph, ~50-70% 慢
        # vllm 0.10/0.16 默认就是 PIECEWISE, 没冲突, 这里仍 use full default behavior.
        rm_cg_mode = os.environ.get("SIA_RM_CUDAGRAPH", "default").lower()
        compilation_config = None
        enforce_eager = False
        if rm_cg_mode == "none" or rm_cg_mode == "eager":
            enforce_eager = True
            print(f"[RMClient] enforce_eager=True (SIA_RM_CUDAGRAPH={rm_cg_mode})",
                  flush=True)
        elif rm_cg_mode == "piecewise":
            compilation_config = {"cudagraph_mode": 1}  # 1 = PIECEWISE
            print("[RMClient] cudagraph_mode=PIECEWISE", flush=True)
        elif rm_cg_mode == "full":
            compilation_config = {"cudagraph_mode": 2}  # 2 = FULL
            print("[RMClient] cudagraph_mode=FULL", flush=True)

        # SIA_RM_PREFIX_CACHING=0 disables APC on the RM vLLM instance.
        # Needed for vllm 0.18.0 + PIECEWISE: APC generates new batch_descriptors
        # at inference time, triggering RuntimeError in PIECEWISE graph mode.
        # vllm 0.17.1 (AOT capture) is unaffected — leave at default 1 for VL-30B.
        rm_apc = os.environ.get("SIA_RM_PREFIX_CACHING", "1").strip() not in ("0", "false", "no")
        print(f"[RMClient] enable_prefix_caching={rm_apc} (SIA_RM_PREFIX_CACHING="
              f"{'1' if rm_apc else '0'})", flush=True)

        llm_kwargs = dict(
            model=model_path,
            hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
            dtype="bfloat16",
            enable_prefix_caching=rm_apc,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
        )
        # Prevent chunked-prefill reward count mismatch.
        # 当 APC 关闭时, 每个候选都需全序列 forward; 10 候选 × max_model_len 可能
        # 超过 vllm 默认 max_num_batched_tokens=16384, 导致 batch 被拆成多步,
        # compute_logits 对 partial prefill 序列也写 reward → numel mismatch.
        # 修复: 设 max_num_batched_tokens ≥ max_model_len × topk.
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
            print(f"[RMClient] max_num_batched_tokens={max_num_batched_tokens} "
                  f"(caller-specified)", flush=True)
        elif not rm_apc and not multiprocessing:
            safe_mbt = max_model_len * 32  # covers topk ≤ 32
            llm_kwargs["max_num_batched_tokens"] = safe_mbt
            print(f"[RMClient] APC disabled: max_num_batched_tokens={safe_mbt} "
                  f"(prevents chunked-prefill reward count mismatch)", flush=True)
        if compilation_config is not None:
            llm_kwargs["compilation_config"] = compilation_config
        if cuda_graph_sizes is not None:
            llm_kwargs["cuda_graph_sizes"] = cuda_graph_sizes
        self.llm = LLM(**llm_kwargs)
        self._sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
        )

        # Session state
        self._sessions: dict[int, list[int]] = {}
        self._next_id = 0

        # Cross-tokenizer incremental prefix cache (P-3).
        # Maps sid → (prefix_len_at_last_encode, stable_rm_prefix_ids).
        # Updated every time _get_stable_rm_prefix() is called; invalidated
        # in end_session().  Only used when _cross_tokenizer is True.
        self._rm_prefix_cache: dict[int, tuple[int, list[int]]] = {}

        # Cross-tokenizer bridge: when LLM tokenizer != RM tokenizer (e.g.
        # 0GM-35B uses 248K-vocab Qwen3.5 tokenizer, RM uses 151K-vocab
        # Qwen3 tokenizer; only 0.2% of overlapping ids match), score_candidates
        # has to go through string: LLM-token-id → text (LLM tokenizer)
        # → RM-token-id (RM tokenizer). Otherwise stay on the zero-overhead
        # token-id-direct path (Qwen3-14B + VM-Qwen3-4B case).
        #
        # Compatibility detection: if the caller passes the LLM tokenizer,
        # compare vocab size + sample 100 ids. If they fully match, leave
        # cross-tokenizer mode off (no overhead).
        self._cross_tokenizer = False
        self._llm_tok = None
        self._rm_tok = None
        if llm_tokenizer is not None:
            from transformers import AutoTokenizer
            rm_tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            same = llm_tokenizer.vocab_size == rm_tok.vocab_size
            if same:
                step = max(1, llm_tokenizer.vocab_size // 100)
                for i in range(0, llm_tokenizer.vocab_size, step):
                    if (llm_tokenizer.convert_ids_to_tokens(i)
                            != rm_tok.convert_ids_to_tokens(i)):
                        same = False
                        break
            if not same:
                self._cross_tokenizer = True
                self._llm_tok = llm_tokenizer
                self._rm_tok = rm_tok
                print(
                    f"[RMClient] cross-tokenizer bridge ON: "
                    f"LLM vocab={llm_tokenizer.vocab_size}, "
                    f"RM vocab={rm_tok.vocab_size}. "
                    f"score_candidates will decode→encode via text.",
                    flush=True,
                )

        # Profile-mode state: how many tokens of each session's prefix have
        # already been pushed through a vLLM forward (thus KV-cached).
        # Used by score_candidates_profiled to know how many new tokens
        # need a separate "prefill" generate call.
        self._last_prefilled: dict[int, int] = {}

        # Clean any stale reward + timing files from previous runs
        truncate_rewards()
        truncate_timings()
        # B-1: cudagraph capture above wrote ~67 dummy GPU tensors into
        # the inproc reward buffer; drop them now so they don't pin GPU
        # memory until the first real score_candidates() call.
        if not multiprocessing:
            clear_inproc_rewards(self._fid)

        # vllm 0.18.0 CUDAGraph 跨实例清除修复:
        # gpu_model_runner.py 在 profiling 阶段结束时调用
        # CUDAGraphWrapper.clear_all_graphs(), 它通过类级 WeakSet _all_instances
        # 清空进程内 ALL CUDAGraphWrapper 的 concrete_cudagraph_entries——包括
        # RM 的。此后 RM inference 遇任何 batch size 都找不到捕获图, 触发重新
        # capture → global flag=False → RuntimeError.
        #
        # 修复: RM init 完毕后立即把 _all_instances 替换为新空 WeakSet。
        # 主 LLM 的 wrappers 将注册到新 WeakSet; clear_all_graphs() 只清 RM-free
        # 的新 WeakSet。RM 的 wrappers 保留 concrete_cudagraph_entries, inference
        # 时正常 replay。
        if not multiprocessing and rm_cg_mode in ("piecewise", "full"):
            try:
                import weakref
                from vllm.compilation.cuda_graph import CUDAGraphWrapper
                n = len(list(CUDAGraphWrapper._all_instances))
                CUDAGraphWrapper._all_instances = weakref.WeakSet()
                print(f"[RMClient] Isolated {n} RM CUDAGraphWrapper instances from "
                      f"global registry (prevents main LLM clear_all_graphs from "
                      f"clearing RM captured graphs).", flush=True)
            except (ImportError, AttributeError):
                pass  # vllm 0.10/0.16/0.17 没有这个问题

        # vllm 0.17+ workspace lock 跨实例冲突修复:
        # vllm 在每个 GPUModelRunner 完成 cudagraph capture 后会 lock 全局
        # workspace (gpu_model_runner.py:6040 + workspace.py:58). 因为
        # workspace_manager 是 process-level singleton, InprocClient 模式下
        # RM 先 init 完, lock 了 workspace; 主 LLM 后续 init 时 MoE kernel
        # 想 grow workspace (e.g. 0GM-35B MoE 需要 128 MB) → AssertionError.
        #
        # RM 自己的 cudagraph 已 captured (用已分配的 size, replay 时不 grow),
        # unlock 安全; 主 LLM 加载完会自己再 lock 一次。
        if not multiprocessing:
            try:
                from vllm.v1.worker.workspace import unlock_workspace
                unlock_workspace()
            except ImportError:
                pass  # vllm 0.10/0.16 没有这个机制, 没有 lock 问题

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
        self._rm_prefix_cache.pop(sid, None)

    def session_length(self, sid: int) -> int:
        return len(self._sessions[sid])

    def _get_stable_rm_prefix(self, sid: int, prefix: list[int]) -> list[int]:
        """Return stable RM prefix IDs for cross-tokenizer sessions (P-3).

        Encodes only the new LLM tokens since the last call and appends them
        to the cached RM prefix, instead of re-encoding the full prefix text.
        delta is typically ~4-5 (SKIP steps between INTERVENEs at 20% rate).
        Encoding each token individually keeps every previous APC block's hash
        unchanged across steps, maximising cross-step KV-cache reuse in the VM.

        Falls back to full re-encode on first call or unexpected negative delta.
        """
        cached = self._rm_prefix_cache.get(sid)
        cur_len = len(prefix)
        if cached is not None:
            cached_len, cached_ids = cached
            delta = cur_len - cached_len
            if delta == 0:
                return cached_ids
            if delta > 0:
                # Encode each new LLM token individually and append.
                # delta is typically the number of SKIP steps since the last
                # INTERVENE (~4-5 with entropy_threshold=1.0).  Encoding one
                # token at a time avoids BPE merges at the old prefix boundary,
                # keeping every previous APC block's hash unchanged across steps.
                stable = cached_ids
                for i in range(delta):
                    tok_id = prefix[cached_len + i]
                    try:
                        new_text = self._llm_tok.decode(
                            [tok_id], skip_special_tokens=False
                        )
                    except (OverflowError, ValueError) as _e:
                        print(f"[RMClient] WARNING: prefix decode failed for token {tok_id} ({_e})",
                              flush=True)
                        new_text = ""
                    new_rm_ids = self._rm_tok.encode(
                        new_text, add_special_tokens=False
                    )
                    stable = stable + new_rm_ids
                self._rm_prefix_cache[sid] = (cur_len, stable)
                return stable
        # First call for this session, or unexpected negative delta — full re-encode.
        try:
            prefix_text = self._llm_tok.decode(prefix, skip_special_tokens=False)
        except (OverflowError, ValueError) as _e:
            vocab_size = self._llm_tok.vocab_size
            bad = [t for t in prefix if not (isinstance(t, int) and 0 <= t < vocab_size)]
            print(f"[RMClient] WARNING: full prefix decode failed ({_e}); bad ids={bad[:5]}, "
                  f"filtering {len(bad)} of {len(prefix)} tokens", flush=True)
            safe_prefix = [t for t in prefix if isinstance(t, int) and 0 <= t < vocab_size]
            prefix_text = self._llm_tok.decode(safe_prefix, skip_special_tokens=False)
        stable = self._rm_tok.encode(prefix_text, add_special_tokens=False)
        self._rm_prefix_cache[sid] = (cur_len, stable)
        return stable

    # ---- scoring ----

    def score_candidates(
        self,
        sid: int,
        candidate_token_ids,
    ) -> torch.Tensor:
        """Score N candidates by running forward on N prompts (prefix+[c])
        and reading the score head outputs.

        Returns: shape (N,) float32 tensor.
        - Inproc mode (default, multiprocessing=False): GPU tensor, same
          device as the RM. Caller can do mean-norm + index_add_ entirely
          on GPU.
        - Multiprocess mode: CPU float32 tensor (read from /dev/shm).

        D-1: 直接返回 tensor (而非 list[float]) — 调用方可以直接做 mean-norm /
        scalar mul / .to(gpu) 一次性完成, 避免 `torch.tensor(list)` 重建 + 多次
        host→device transfer。
        B-1: inproc mode 进一步把 .cpu() sync + /dev/shm 文件 IPC 也省掉。
        """
        if sid not in self._sessions:
            raise ValueError(f"Unknown session id {sid}")
        prefix = self._sessions[sid]
        TP = self._TokensPrompt  # X-5: local alias avoids self.__dict__ lookup × N
        n_expected = len(candidate_token_ids)

        if self._cross_tokenizer:
            # P-1: token-id bridge for incompatible tokenizers (e.g. 0GM-35B
            # + VM-Qwen3-4B). Decode the LLM-side prefix + each candidate to
            # text, then re-encode with the RM tokenizer.
            #
            # P-2: 稳定前缀 — prefix 编码一次，K 个候选共享，同步内 APC 全命中。
            # P-3: 增量缓存 — 每步只编码新增的 delta 个 LLM token（逐 token），
            #   append 到缓存前缀；delta ≈ 4-5（两次 INTERVENE 间的 SKIP 步数）。
            #   效果：跨步的完整 block 哈希完全不变 → APC 跨步 100% 命中，
            #   每个 prompt 仅需 forward tail (P mod 16) + candidate ≈ 8 tokens，
            #   而不是全量重编后 last-block-miss 的 ~25 tokens。
            # 近似误差：增量拼接在 BPE 边界处差 1-2 token，但 SIA 只看相对排名，
            # 边界噪声可忽略（与 P-2 的 split 编码误差同性质）。
            stable_rm_prefix_ids = self._get_stable_rm_prefix(sid, prefix)
            prompts = []
            for c in candidate_token_ids:
                try:
                    cand_text = self._llm_tok.decode([c], skip_special_tokens=False)
                except (OverflowError, ValueError) as _e:
                    # Invalid candidate token ID (out of u32 range or similar edge case).
                    # Log for diagnosis; fall back to empty text to keep prompt count aligned.
                    print(f"[RMClient] WARNING: decode failed for candidate {c} ({_e})",
                          flush=True)
                    cand_text = ""
                cand_rm_ids = self._rm_tok.encode(
                    cand_text, add_special_tokens=False
                )
                prompts.append(
                    TP(prompt_token_ids=stable_rm_prefix_ids + cand_rm_ids)
                )
        else:
            # Zero-overhead path: tokenizer-compatible (Qwen3-14B + VM-Qwen3-4B).
            # X-5: candidate_token_ids comes from SIA processor topk (list[int]),
            # already Python ints — skip redundant `int(c)` cast.
            prompts = [
                TP(prompt_token_ids=prefix + [c])
                for c in candidate_token_ids
            ]

        # B-1: dual reward channel.
        # Clear before generate (defensive — handles stale state if a
        # prior call raised between generate() and take/read).
        if self._multiprocessing:
            truncate_rewards()
        else:
            clear_inproc_rewards(self._fid)

        _ = self.llm.generate(prompts, self._sp, use_tqdm=False)

        if self._multiprocessing:
            rewards = read_rewards()  # CPU tensor, concatenates all records
        else:
            rewards = take_inproc_rewards(self._fid)  # GPU tensor (or None)

        if rewards is None or rewards.numel() != n_expected:
            n_got = 0 if rewards is None else int(rewards.numel())
            channel = "/dev/shm" if self._multiprocessing else "inproc buffer"
            if self._multiprocessing:
                records = read_all_rewards()
                shapes = [tuple(r.shape) for r in records]
                detail = f"records seen: {shapes}"
            else:
                detail = ""
            raise RuntimeError(
                f"reward channel ({channel}) returned {n_got} values, "
                f"expected {n_expected}. {detail} "
                f"Likely SIA_REWARD_FILE_ID mismatch or vLLM reordered "
                f"prompts."
            )
        return rewards

    def score_candidates_batch(
        self,
        requests: list,
    ) -> list:
        """Score candidates for multiple sessions in a single llm.generate() call.

        Args:
            requests: list of (sid, candidate_token_ids) pairs.
                      fix_a_token must already have been called for all new
                      output tokens before this call (same contract as
                      score_candidates).

        Returns:
            list of Tensor, one per request, each shape (K_i,).
            Order matches the input list.  Same device/dtype contract as
            score_candidates() — GPU tensor in inproc mode, CPU in multiprocess.
        """
        if not requests:
            return []
        # Diagnostic: always print entry so we know the function is reached.
        print(f"[RM-diag] score_candidates_batch n={len(requests)}", flush=True)
        # Single-request fast-path: delegate to the existing method so all
        # its error-handling and edge cases remain in one place.
        if len(requests) == 1:
            sid, cands = requests[0]
            return [self.score_candidates(sid, cands)]

        TP = self._TokensPrompt
        all_prompts: list = []
        offsets: list = []   # (start, end) index into all_prompts per request

        for sid, candidate_token_ids in requests:
            if sid not in self._sessions:
                raise ValueError(f"Unknown session id {sid}")
            prefix = self._sessions[sid]
            start = len(all_prompts)
            n = len(candidate_token_ids)

            if self._cross_tokenizer:
                # P-2 + P-3: stable prefix (same encoding for all K candidates)
                # with incremental cross-step caching — see score_candidates().
                stable_rm_prefix_ids = self._get_stable_rm_prefix(sid, prefix)
                for c in candidate_token_ids:
                    try:
                        cand_text = self._llm_tok.decode([c], skip_special_tokens=False)
                    except (OverflowError, ValueError) as _e:
                        print(f"[RMClient] WARNING: decode failed for candidate {c} ({_e})",
                              flush=True)
                        cand_text = ""
                    cand_rm_ids = self._rm_tok.encode(
                        cand_text, add_special_tokens=False
                    )
                    all_prompts.append(
                        TP(prompt_token_ids=stable_rm_prefix_ids + cand_rm_ids)
                    )
            else:
                for c in candidate_token_ids:
                    all_prompts.append(TP(prompt_token_ids=prefix + [c]))

            offsets.append((start, start + n))

        n_total = len(all_prompts)

        if self._multiprocessing:
            truncate_rewards()
        else:
            clear_inproc_rewards(self._fid)

        _ = self.llm.generate(all_prompts, self._sp, use_tqdm=False)

        if self._multiprocessing:
            all_rewards = read_rewards()
        else:
            # SIA_RM_BATCH_DEBUG=1: print how many compute_logits calls vLLM
            # made internally and the sample count of each — directly reveals
            # whether N sessions are batched as one call (n_calls=1, samples=N×K)
            # or serialized (n_calls=N, samples=K each).
            if os.environ.get("SIA_RM_BATCH_DEBUG", "0") == "1":
                raw = _REWARD_BUFFERS.get(self._fid, [])
                shapes = [int(t.numel()) for t in raw]
                n_sessions = len(requests)
                print(
                    f"[RM-batch-debug] sessions={n_sessions} "
                    f"total_prompts={n_total} "
                    f"compute_logits_calls={len(shapes)} "
                    f"samples_per_call={shapes}",
                    flush=True,
                )
            all_rewards = take_inproc_rewards(self._fid)

        if all_rewards is None or all_rewards.numel() != n_total:
            n_got = 0 if all_rewards is None else int(all_rewards.numel())
            channel = "/dev/shm" if self._multiprocessing else "inproc buffer"
            if self._multiprocessing:
                records = read_all_rewards()
                shapes = [tuple(r.shape) for r in records]
                detail = f"records seen: {shapes}"
            else:
                detail = ""
            raise RuntimeError(
                f"reward channel ({channel}) returned {n_got} values, "
                f"expected {n_total} (batch of {len(requests)} sessions). "
                f"{detail}"
                f"Likely SIA_REWARD_FILE_ID mismatch or vLLM reordered prompts."
            )

        # ORDERING ASSUMPTION: offsets slicing is correct only if vLLM processes
        # prompts in the order they were submitted to llm.generate().  This holds
        # because vLLM v1's InprocClient adds all requests synchronously (FIFO) and
        # the scheduler's WAITING queue is drained in arrival order (verified
        # empirically for single-session K-prompt batches; applies to multi-session
        # N×K batches as long as vLLM does not introduce APC-aware request reordering
        # in a future release).  The numel check above catches count mismatches but
        # NOT ordering errors — a silent wrong-score bug would result if vLLM ever
        # reorders.
        return [all_rewards[start:end] for (start, end) in offsets]

    # ---- bench / introspection ----

    def time_score_candidates(
        self,
        sid: int,
        candidate_token_ids,
    ) -> tuple[torch.Tensor, float]:
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

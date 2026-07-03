"""
SIA per-token intervention (HTTP RM Server version) - vLLM v1 engine version

Architecture:
  - LLM: vllm (v1 engine) handles generation; GPU managed by vllm
  - RM:  standalone RM server (sia_rm_server.py), called via HTTP /score
  - SIALogitsProcessor runs in the EngineCore subprocess, registered via LLM(logits_processors=[...])

Intervention logic (per-token):
  1. Extract top-k candidate token IDs from LLM logits
  2. Decode the current generation sequence + each candidate to text
  3. Batch-send to the RM server for scoring via HTTP POST /score
  4. combined_logits[topk_indices] += rm_scores * weight
  5. Return combined_logits

Usage example:
  # First start the RM server (sia_rm_server.py), then run this script
  python sia_vllm_RM.py \\
    --llm  /workspace/SIA/models/Qwen3-1.7B-Base \\
    --rm_url http://localhost:8001 \\
    --llm_gpu_mem 0.3 --topk 5 --weight 0.1 --max_tokens 20
"""

import argparse
import hashlib
import os
import re
import requests
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Make sia_rm importable in EngineCore subprocess (spawned via python -m,
# does not inherit parent's sys.path). Adding src/ to PYTHONPATH lets the
# subprocess find sia_rm when SIALogitsProcessor.__init__ instantiates
# RMClient (which transitively reimports sia_rm in the RM EngineCore).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_pp = os.environ.get("PYTHONPATH", "")
if _THIS_DIR not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_THIS_DIR + ":" + _pp) if _pp else _THIS_DIR
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

_SIA_WEIGHT_RE = re.compile(r'^\[SIA:weight=([-+]?[0-9]*\.?[0-9]+)\]\n?')


def _parse_sia_header(prompt_text: str):
    """
    Check whether the prompt starts with a [SIA:weight=X] header (injected by the HTTP server).
    Returns (weight, cleaned_text); weight is None if no header is present.
    """
    m = _SIA_WEIGHT_RE.match(prompt_text)
    if m:
        return float(m.group(1)), prompt_text[m.end():]
    return None, prompt_text


def parse_conversation(text: str):
    """
    Parse conversation text into a conversations list. Supports the following formats:
      - 'Human:\\nQ\\nAssistant:\\nA'  (colon-separated format)
      - '[INST] Q [/INST] A'          (Llama 2 / Mistral format)
      - 'user\\nQ\\nassistant\\nA'    (ChatML stripped, e.g. Qwen3 / Llama 3)
      - 'user\\nQ\\nmodel\\nA'        (Gemma; 'model' is an alias for 'assistant')
    """
    text = text.strip()

    # Format 1: Human: / Assistant: with colon
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

    # Format 3: role on its own line (ChatML stripped / Gemma etc.)
    # 'model' is Gemma's alias for 'assistant'
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
    """Extract the content of the last user turn from the prompt text."""
    user_content = None
    for c in parse_conversation(prompt_text):
        if c['role'] == 'user':
            user_content = c['content']
    return user_content if user_content is not None else prompt_text


# ---------------------------------------------------------------------------
# LogitsProcessor factory
# ---------------------------------------------------------------------------

def make_sia_processor(
    rm_url: str = "http://localhost:8001",
    topk: int = 10,
    weight: float = 1.0,
    entropy_threshold: Optional[float] = None,
    logit_gap_threshold: Optional[float] = None,
    rm_backend: str = "pytorch",       # "pytorch" / "vllm" / "b2"
    rm_model: Optional[str] = None,    # required when rm_backend ∈ {"vllm","b2"}: RM model path
    use_token_ids: bool = False,       # C2: client pre-tokenizes and sends token_ids directly to RM
    rm_b2_gpu_mem: float = 0.3,        # b2 backend: gpu_memory_utilization for the RM vLLM instance
    rm_max_model_len: int = 4096,      # b2 backend: max_model_len for the RM vLLM instance
    enable_thinking: Optional[bool] = None,   # chat_template enable_thinking forwarded to RM prefix construction,
                                              # 100% consistent with what the LLM actually sees (None=not passed)
    eager_vm_prefill: bool = False,    # b2 backend: after each decode step, prefill predicted next token
                                       # into VM KV cache so the next INTERVENE finds warm prefix (no tail prefill)
    vm_topk: Optional[int] = None,     # limit candidates sent to VM (None = same as topk).
                                       # Useful for FaRMA proxy benchmarking: keep topk=10 for entropy
                                       # gate (preserving intervention rate) but send only vm_topk=1
                                       # candidates to VM, reducing VM batch from N×topk to N×1.
    vm_head_type: str = "scalar",      # b2 backend: VM reward head architecture.
                                       # "scalar" (default): K forwards per step (original)
                                       # "vocab_lowrank": 1 forward per step + vocab indexing (FaRMA)
    vm_head_rank: int = 64,            # b2 backend: rank for vocab_lowrank head
):
    """
    Returns a SIALogitsProcessor class (not an instance).
    The vllm engine instantiates the class with (vllm_config, device, is_pin_memory) arguments.

    rm_backend:
      - "pytorch": calls the POST /score endpoint of src/sia_rm_server.py (hand-written FastAPI)
      - "vllm":    calls the /classify endpoint of a running vllm serve instance (HTTP, must be started separately)
      - "b2":      in-process stateful RM (sia_rm.RMClient), shares GPU with the LLM.
                    No HTTP/IPC overhead; relies on vLLM prefix caching so multi-step scoring
                    is nearly decode-only (~7-15ms vs HTTP /classify ~30ms).
                    rm_model is required: VM model path (architecture is coerced to
                    Qwen3WithScoreForCausalLM via hf_overrides).

    use_token_ids (only effective when rm_backend="vllm"):
      - False: client sends strings; vLLM server re-tokenizes
      - True:  client pre-tokenizes and sends list[list[int]]

    SIA_PROFILE env var (default "1", enabled): accumulates per-phase timing for each INTERVENE;
    prints a [SIA-pf-summary] aggregate p50/p95/max every 100 INTERVENE calls.
    """
    if rm_backend in ("vllm", "b2") and not rm_model:
        raise ValueError(
            f"rm_backend='{rm_backend}' requires --rm_model"
        )

    class SIALogitsProcessor(LogitsProcessor):
        _RM_URL = rm_url
        _RM_BACKEND = rm_backend
        _RM_MODEL = rm_model
        _TOPK = topk
        _WEIGHT = weight
        _ENTROPY_THRESHOLD = entropy_threshold
        _GAP_THRESHOLD = logit_gap_threshold   # second gate: SKIP when top1-top2 logit gap >= this
        _USE_TOKEN_IDS = use_token_ids
        _RM_B2_GPU_MEM = rm_b2_gpu_mem
        _RM_MAX_MODEL_LEN = rm_max_model_len
        _ENABLE_THINKING = enable_thinking   # None / True / False
        _EAGER_VM_PREFILL = eager_vm_prefill
        _VM_TOPK: Optional[int] = vm_topk   # None = use _TOPK; int = limit VM candidates per request
        _VM_HEAD_TYPE: str = vm_head_type   # "scalar" or "vocab_lowrank"
        _VM_HEAD_RANK: int = vm_head_rank

        # Maximum number of sessions per score_candidates_batch() call (b2 backend).
        # Prevents RM KV-cache OOM at high concurrency: a single apply() with
        # batch_size=128 and topk=10 would otherwise submit 1280 prompts at once,
        # exhausting the RM's pre-allocated KV block pool.
        # Set SIA_RM_BATCH_CHUNK=0 to disable chunking (original unlimited behavior).
        _B2_BATCH_CHUNK: int = int(os.environ.get("SIA_RM_BATCH_CHUNK", "8"))

        # SIA client-side profiling (distinct from the RM_PROFILE server-side profiling)
        _PROFILE_DETAIL: bool = os.environ.get("SIA_PROFILE", "1") == "1"
        _PF_STATS_INTERVAL: int = int(os.environ.get("SIA_PF_INTERVAL", "100"))
        # D-1: per-step INTERVENE/SKIP/error print switch
        #   verbose — full log retained, debug-friendly, one extra cpu().tolist() sync
        #   quiet (default) — skips per-step print, saves ~0.5-1ms/step (includes format + stdout flush
        #                     + topk_values cpu sync). Flip detection uses GPU argmax path;
        #                     flip count is still tracked (1 int sync per batch, not 5 floats)
        #   flip stats / startup messages / DONE / SIA-pf-summary are unaffected by this var
        _LOG_LEVEL: str = os.environ.get("SIA_LOG_LEVEL", "quiet")

        # ===== Dual-VM validation hook =====
        # When SIA_DUAL_VM_URL is set (e.g. http://localhost:8002/score_token_ids),
        # _score_candidates_vllm, after completing the vllm /classify call in path A (same vocab),
        # forwards the same payload to that URL (expected to be the sia_rm_pytorch_official.py
        # /score_token_ids endpoint) and writes both sets of scores to SIA_DUAL_VM_LOG (JSONL)
        # for offline comparison analysis.
        # Default empty string => disabled, zero overhead. Logging only; does not affect SIA intervention.
        _DUAL_VM_URL: str = os.environ.get("SIA_DUAL_VM_URL", "")
        _DUAL_VM_LOG: str = os.environ.get(
            "SIA_DUAL_VM_LOG", "/tmp/sia_dual_vm.jsonl",
        )

        # ===== SIA DEBUG HISTOGRAM START =====
        # For diagnosing "why is the intervention rate low". When SIA_DEBUG_HIST=1, prints at each request DONE:
        #   - ENTROPY_HIST: 16-bin histogram of top-5 re-normalized entropy [0..1.6 step=0.1]
        #   - GAP_HIST: 16-bin histogram of top-1 vs top-2 logit gap [0..8.0 step=0.5]
        #   - ENTROPY_INTHINK / ENTROPY_OUTTHINK: bucketed by <think>/</think> boundary
        # Default SIA_DEBUG_HIST=0 -> zero hot-path cost.
        # This entire block is bracketed by SIA DEBUG HISTOGRAM START/END comments for easy manual rollback (grep + sed).
        _DEBUG_HIST: bool = os.environ.get("SIA_DEBUG_HIST", "0") == "1"
        # ===== SIA DEBUG HISTOGRAM END =====

        # ----------------------------------------------------------------
        # Initialization: runs inside the EngineCore subprocess
        # ----------------------------------------------------------------
        def __init__(self, vllm_config, device: torch.device,
                     is_pin_memory: bool) -> None:
            self._llm_device = device
            print(f"[SIA] __init__ _EAGER_VM_PREFILL={self._EAGER_VM_PREFILL} _VM_TOPK={self._VM_TOPK} _RM_BACKEND={self._RM_BACKEND}", flush=True)

            # LLM tokenizer (for decoding candidate tokens and extracting user content)
            llm_model_path = vllm_config.model_config.model
            print(f"[SIA] Loading LLM tokenizer: {llm_model_path}", flush=True)
            self._llm_tok = AutoTokenizer.from_pretrained(
                llm_model_path, trust_remote_code=True
            )

            # HTTP session (reuse TCP connection to RM server)
            self._rm_session = requests.Session()
            print(f"[SIA] RM URL: {self._RM_URL}", flush=True)

            # Pre-compute sentinel bounds once at startup (used by _extract_user_content)
            self._sentinel_prefix, self._sentinel_suffix = self._compute_sentinel_bounds()

            # per-request state
            self._prompt_user: dict[int, str] = {}
            self._output_ids: dict[int, list] = {}
            self._weight_per_req: dict[int, float] = {}
            self._topk_per_req: dict[int, int] = {}
            self._entropy_per_req: dict[int, Optional[float]] = {}
            # Intervention stats: total token steps and actual intervention count per request
            self._total_steps: dict[int, int] = {}
            self._intervened_steps: dict[int, int] = {}
            # Number of times intervention caused top-1 token to flip (pre top-1 != post top-1)
            self._flipped_steps: dict[int, int] = {}
            # response_so_far incremental decode cache (avoids O(n) full re-decode each step)
            self._decoded_text: dict[int, str] = {}
            self._decoded_token_count: dict[int, int] = {}
            # C1: chat template prefix/suffix cache (rendered once per request)
            self._chat_prefix_per_req: dict[int, Optional[str]] = {}
            self._chat_suffix_per_req: dict[int, Optional[str]] = {}
            # C3 (2026-06-03): prefix_ids cache for direct token_ids path when tokenizers are the same
            # (once per request, avoids re-encoding the user turn at each INTERVENE step)
            self._manual_prefix_ids_per_req: dict[int, Optional[list[int]]] = {}

            # C2: RM tokenizer
            # The vllm backend always loads the RM tokenizer:
            #   - used to detect whether LLM/RM share the same vocab (decides between
            #     direct-token-ids fast path and text round-trip fallback)
            #   - not needed on the same-vocab path (LLM tok == RM tok), used only for sanity comparison
            #   - required on the cross-vocab path (e.g. 0GM-35B + VM-Qwen3-4B) for re-encoding text
            self._rm_tok = None
            self._same_tokenizer = False  # True => LLM/RM share the same vocab; token IDs can be passed directly
            if self._RM_BACKEND == "vllm":
                print(f"[SIA] Loading RM tokenizer: {self._RM_MODEL}", flush=True)
                self._rm_tok = AutoTokenizer.from_pretrained(
                    self._RM_MODEL, trust_remote_code=True,
                )
                self._same_tokenizer = self._tokenizers_compatible(
                    self._llm_tok, self._rm_tok,
                )
                print(
                    f"[SIA] Tokenizer compat: same_vocab={self._same_tokenizer} "
                    f"(LLM vocab={self._llm_tok.vocab_size}, "
                    f"RM vocab={self._rm_tok.vocab_size}) → "
                    f"{'direct-token-ids fast path' if self._same_tokenizer else 'text round-trip (cross-tokenizer)'}",
                    flush=True,
                )

            # Client-side profiling accumulation
            # Note: N in SIA-pf-summary @N is _pf_intervene_calls (INTERVENE-only count).
            # SKIP / apply_entry and similar phases are recorded every token (not just on INTERVENE),
            # so those phases will have sample counts >> N. p50/p95 are still percentiles of each phase's own samples.
            self._pf_stats: dict[str, list] = {
                # vllm http backend phases (legacy, not used by b2)
                "format_chat":      [],
                "tokenize_client":  [],
                "http_post":        [],
                "parse_response":   [],
                # b2 backend phases (RMClient call internals — within one INTERVENE call)
                "b2_prefix_adv":    [],   # _prepare_b2_session (session init + fix_a_token)
                "b2_score_call":    [],   # RMClient.score_candidates end-to-end (amortized per-request)
                "b2_batch_wall_abs":[],   # score_candidates_batch() absolute wall (not amortized; per INTERVENE step)
                "intv_batch_size":  [],   # number of sessions batched per INTERVENE step (len(_b2_reqs))
                "b2_eager_wall":    [],   # eager_prefill_batch() wall per apply() call (only when --eager_vm_prefill)
                "total":            [],   # one INTERVENE call end-to-end

                # === apply() sub-phases (per-token) — added 2026-06-01 ===
                # Phases recorded on every apply() call:
                "apply_total":      [],   # apply() end-to-end (includes INTERVENE or SKIP)
                "apply_topk_ent":   [],   # torch.topk + log_softmax + entropy (GPU)
                "apply_cpu_sync":   [],   # entropy.cpu().tolist() (GPU->CPU sync wait)
                "skip_step":        [],   # SKIP path only (from sync completion to apply return)
                # Additional sub-phases for INTERVENE path only (subset of apply() time):
                "apply_intv_step":  [],   # apply() total wall on steps where ≥1 request intervenes
                "intv_prepare":     [],   # topk_indices.cpu().tolist() + output_ids/user_content retrieval
                "intv_apply_logits":[],   # mean-norm + .to(gpu) + index_add_ + flip
            }
            self._pf_intervene_calls: int = 0
            self._pf_apply_calls: int = 0    # incremented on every apply call (used to set per-token level interval)
            self._pf_total_req_steps: int = 0  # sum of batch_size across all apply() calls (denominator for rate)

            # Dual-VM log counter (used to rate-limit error messages)
            self._dual_vm_call_count: int = 0
            if self._DUAL_VM_URL:
                print(
                    f"[SIA] DUAL_VM enabled: forwarding token_ids payload to "
                    f"{self._DUAL_VM_URL}, logging to {self._DUAL_VM_LOG}",
                    flush=True,
                )

            # ===== SIA DEBUG HISTOGRAM START =====
            # debug state (only used when SIA_DEBUG_HIST=1; empty dicts are zero-cost)
            self._dbg_entropy_hist: dict[int, list[int]] = {}        # req_idx -> 16 bins [0..1.6]
            self._dbg_gap_hist: dict[int, list[int]] = {}            # req_idx -> 16 bins [0..8.0]
            self._dbg_entropy_hist_inthink: dict[int, list[int]] = {}  # entropy inside <think>...</think> only
            self._dbg_entropy_hist_outthink: dict[int, list[int]] = {} # entropy after </think>
            self._dbg_in_think: dict[int, bool] = {}                 # req_idx -> whether currently inside think block
            self._think_open_ids: set[int] = set()
            self._think_close_ids: set[int] = set()
            if self._DEBUG_HIST:
                # Detect whether <think> / </think> are single tokens (they usually are in Qwen3.5 thinking models).
                # If multi-token, the bucketing logic will skip them (last_tok never in set) and all go to inthink.
                for marker, target_set in (
                    ("<think>", self._think_open_ids),
                    ("</think>", self._think_close_ids),
                ):
                    try:
                        ids = self._llm_tok.encode(marker, add_special_tokens=False)
                        if len(ids) == 1:
                            target_set.add(ids[0])
                    except Exception:
                        pass
                print(
                    f"[SIA-debug] SIA_DEBUG_HIST=1 enabled; "
                    f"think_open_ids={self._think_open_ids}, think_close_ids={self._think_close_ids}",
                    flush=True,
                )
            # ===== SIA DEBUG HISTOGRAM END =====

            # B2 backend: in-process RMClient + per-request session table
            # Assumes LLM and RM use the same tokenizer (prerequisite for the B2 path), so token IDs are interchangeable
            self._rm: Optional[object] = None
            self._b2_sessions: dict[int, int] = {}        # req_idx -> rm_sid
            self._b2_chat_prefix_len: dict[int, int] = {}  # req_idx -> chat_prefix token count
            if self._RM_BACKEND == "b2":
                print(f"[SIA-b2] Starting RMClient(model={self._RM_MODEL}, "
                      f"gpu_mem={self._RM_B2_GPU_MEM}) ...", flush=True)
                # This forks another vLLM EngineCore subprocess (for the RM).
                # The process runs inside the LLM EngineCore subprocess — nested but viable.
                from sia_rm import RMClient
                # vllm 0.19 + MoE main LLM (e.g. 0GM-35B): InprocClient mode lets RM
                # share the global workspace with the main LLM in the same process. RM inits first
                # and locks the workspace; the main LLM MoE forward needs a larger workspace but
                # grow is blocked -> AssertionError.
                # Use env var SIA_RM_MULTIPROCESS=1 to force RM into a subprocess to isolate the workspace.
                # Trade-off: 1+4 split revives, b2_score_call ~9ms -> ~24ms.
                rm_mp = os.environ.get("SIA_RM_MULTIPROCESS", "0") == "1"
                if rm_mp:
                    print(f"[SIA-b2] RMClient using multiprocessing=True "
                          f"(workspace isolation for MoE main LLM)", flush=True)
                # Pass the LLM tokenizer into RMClient: if it differs from the RM's own tokenizer
                # (e.g. 0GM-35B uses Qwen3.5 248K vocab vs RM Qwen3 151K),
                # RMClient internally enables the cross-tokenizer bridge (decode->encode).
                # When they match (e.g. Qwen3-14B + VM-Qwen3-4B), the bridge is disabled with zero overhead.
                self._rm = RMClient(
                    model_path=self._RM_MODEL,
                    gpu_mem=self._RM_B2_GPU_MEM,
                    max_model_len=self._RM_MAX_MODEL_LEN,
                    multiprocessing=rm_mp,
                    llm_tokenizer=self._llm_tok,
                    head_type=self._VM_HEAD_TYPE,
                    head_rank=self._VM_HEAD_RANK,
                )
                print(f"[SIA-b2] RMClient ready", flush=True)

        # ----------------------------------------------------------------
        # Extract user content from prompt token IDs
        # ----------------------------------------------------------------
        def _compute_sentinel_bounds(self):
            """
            Called once at startup: uses a sentinel string to determine the prefix / suffix token
            sequence of a user turn.
            Returns (prefix, suffix) lists on success; returns (None, None) if the tokenizer
            has no chat_template or if detection fails.
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
            Primary method: locates the last user turn at the token ID level using pre-computed
            prefix/suffix bounds; makes no format assumptions and works with any tokenizer that
            has a chat_template.
            Fallback: regex-based text parsing (covers Llama 2 / Gemma / Human: formats).
            Both methods switch silently with no warnings printed.
            """
            prefix = self._sentinel_prefix
            suffix = self._sentinel_suffix
            if prefix is not None and suffix is not None:
                prompt_list = list(prompt_ids)
                # Last occurrence of prefix (handles multi-turn conversations)
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

            # Fallback: text parsing (normal path for raw string prompts / no chat_template)
            prompt_text = self._llm_tok.decode(prompt_ids, skip_special_tokens=True)
            return extract_user_content(prompt_text)

        # ----------------------------------------------------------------
        # C3 helper: detect whether LLM/RM share the same vocab + cached prefix_ids consistent with official sia.py
        # ----------------------------------------------------------------
        @staticmethod
        def _tokenizers_compatible(llm_tok, rm_tok) -> bool:
            """Check whether the LLM and RM tokenizers are token-ID compatible (same vocabulary).

            Same vocab => LLM output token IDs can be passed directly to RM, skipping text round-trip
            and eliminating precision loss from irreversible BPE boundary decode-encode.

            Examples:
              - Qwen3-14B + Qwen3-4B -> same (151,936 vocab)
              - Qwen3-VL-30B-A3B-Instruct + Qwen3-4B -> same (Qwen3 vocab 151,936)
              - 0GM-1.0-35B-A3B-0427 + Qwen3-4B -> not same (0GM extended to 248k)

            Strategy: vocab_size matches + encoding IDs are identical for several sampled strings.
            """
            try:
                if llm_tok.vocab_size != rm_tok.vocab_size:
                    return False
            except Exception:
                return False
            # Sample various string types, covering ASCII / CJK / chat-special / numeric
            samples = [
                "Hello world",
                "你好世界",
                "<|im_start|>user\n",
                "<|im_end|>\n<|im_start|>assistant\n",
                "1234567890",
                " a quick brown fox jumps over the lazy dog.",
            ]
            for s in samples:
                try:
                    if (
                        llm_tok.encode(s, add_special_tokens=False)
                        != rm_tok.encode(s, add_special_tokens=False)
                    ):
                        return False
                except Exception:
                    return False
            return True

        def _get_manual_prefix_ids(
            self, req_idx: int, user_content: str,
        ) -> Optional[list[int]]:
            """Build a manual chat prefix fully consistent with the official sia.py, encode it, and return it cached.

            Format (matches official src/sia.py; does **not** use apply_chat_template to avoid Qwen3
            chat_template auto-injecting <think>\\n\\n</think> empty block when enable_thinking=False,
            which would cause RM to see an OOD prefix):
                <|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n

            Only called when _same_tokenizer=True (LLM tok == RM tok); encodes using the LLM tokenizer
            (equivalent to encoding with the RM tokenizer).

            Note: assumes Qwen3 chat template (ChatML). If the LLM is non-Qwen3 but still vocab-compatible
            with the RM tokenizer (extremely rare), the encoded prefix may not be recognized as a
            conversation start by the model — this case is already filtered by _tokenizers_compatible
            (all our SIA value models are Qwen3-series).
            """
            cached = self._manual_prefix_ids_per_req.get(req_idx)
            if cached is not None:
                return cached
            try:
                prefix_text = (
                    "<|im_start|>user\n"
                    + user_content
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )
                prefix_ids = self._llm_tok.encode(
                    prefix_text, add_special_tokens=False,
                )
                if not prefix_ids:
                    self._manual_prefix_ids_per_req[req_idx] = None
                    return None
            except Exception:
                self._manual_prefix_ids_per_req[req_idx] = None
                return None
            self._manual_prefix_ids_per_req[req_idx] = prefix_ids
            return prefix_ids

        # ----------------------------------------------------------------
        # C1 helper: cached chat template prefix/suffix
        # ----------------------------------------------------------------
        def _get_chat_template_parts(self, req_idx: int, user_content: str):
            """Renders the chat template once per request and caches a prefix 100% consistent with what the LLM sees.

            2026-06-03 fix: the previous approach using [user, assistant=SENTINEL] + split produced a prefix
            that **did not include** the chat_template's generation_prompt injection (e.g. Qwen3's `<think>\\n`
            when enable_thinking is on, or the `<think>\\n\\n</think>\\n\\n` empty block when off).
            This caused the RM to see a context inconsistent with what the LLM sees at generation time (OOD).

            New approach: use `apply_chat_template(user, add_generation_prompt=True, enable_thinking=X)`
            to get a prefix 100% consistent with the LLM side. suffix is always "" (Fix #1 removed close tag).

            Subsequent INTERVENE steps only need `prefix + response_so_far + candidate` string concatenation.
            Returns (prefix, suffix=""); returns (None, None) triggering fallback if apply_chat_template fails.
            """
            if req_idx in self._chat_prefix_per_req:
                return self._chat_prefix_per_req[req_idx], self._chat_suffix_per_req[req_idx]

            try:
                # Key: same apply_chat_template call as the LLM (add_generation_prompt=True
                # + same enable_thinking), so the prefix the RM sees is 100% consistent with what the LLM sees
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if self._ENABLE_THINKING is not None:
                    kwargs["enable_thinking"] = self._ENABLE_THINKING
                prefix = self._llm_tok.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    **kwargs,
                )
            except Exception:
                self._chat_prefix_per_req[req_idx] = None
                self._chat_suffix_per_req[req_idx] = None
                return None, None

            # Strip leading BOS (if present)
            bos = self._llm_tok.bos_token
            if bos and prefix.startswith(bos):
                prefix = prefix[len(bos):]
            # Suffix is always empty (after Fix #1 we no longer append a close tag at the end of RM input)
            suffix = ""
            self._chat_prefix_per_req[req_idx] = prefix
            self._chat_suffix_per_req[req_idx] = suffix
            return prefix, suffix

        # ----------------------------------------------------------------
        # Profiling utilities
        # ----------------------------------------------------------------
        def _pf_record(self, phase: str, dur_ms: float) -> None:
            arr = self._pf_stats.get(phase)
            if arr is None:
                return
            arr.append(dur_ms)
            # Prevent unbounded growth: drop the first half when size exceeds 2000
            if len(arr) > 2000:
                del arr[:1000]

        def _pf_summary_if_due(self) -> None:
            self._pf_intervene_calls += 1
            if self._pf_intervene_calls % self._PF_STATS_INTERVAL != 0:
                return
            # Rate header: intervention rate and average batch size per step
            intv_rate = (
                self._pf_intervene_calls / self._pf_total_req_steps * 100
                if self._pf_total_req_steps > 0 else 0.0
            )
            bs_arr = self._pf_stats.get("intv_batch_size", [])
            avg_batch = sum(bs_arr) / len(bs_arr) if bs_arr else 0.0
            print(
                f"[SIA-pf-rate @{self._pf_intervene_calls}] "
                f"intv_calls={self._pf_intervene_calls} "
                f"apply_steps={self._pf_apply_calls} "
                f"req_slots={self._pf_total_req_steps} "
                f"intv_rate={intv_rate:.1f}% "
                f"avg_intv_batch={avg_batch:.2f}",
                flush=True,
            )
            parts = []
            for phase, arr in self._pf_stats.items():
                if not arr:
                    continue
                arr_sorted = sorted(arr)
                n = len(arr_sorted)
                p50 = arr_sorted[n // 2]
                p95_idx = max(0, min(n - 1, int(n * 0.95)))
                p95 = arr_sorted[p95_idx]
                mx = arr_sorted[-1]
                parts.append(f"{phase}: p50={p50:.2f} p95={p95:.2f} max={mx:.2f}")
            print(
                f"[SIA-pf-summary @{self._pf_intervene_calls}] " + " | ".join(parts),
                flush=True,
            )

        # ----------------------------------------------------------------
        # RM scoring: batch-send to RM server in a single HTTP call
        # ----------------------------------------------------------------
        def _score_candidates(
            self,
            req_idx: int,
            user_content: str,
            response_so_far: str,
            candidate_token_ids: list[int],
            output_ids: list[int],
        ) -> torch.Tensor:
            """Returns a shape (topk,) float32 tensor (CPU).

            Dispatches to different implementations based on rm_backend:
              - pytorch: POST /score (sia_rm_server.py custom protocol)
              - vllm:    POST /classify (vLLM native protocol)
              - b2:      in-process RMClient (token_ids path, skips chat template / HTTP)
            """
            if self._RM_BACKEND == "b2":
                return self._score_candidates_b2(
                    req_idx, user_content, output_ids, candidate_token_ids
                )
            if self._RM_BACKEND == "vllm":
                # vllm backend: same vocab takes the direct-token-ids fast path (skipping decode);
                # cross-vocab falls back to text round-trip. Both branches are inside _score_candidates_vllm.
                # Decode is deferred until a text path is actually needed.
                return self._score_candidates_vllm(
                    req_idx, user_content, response_so_far,
                    candidate_token_ids, output_ids,
                )
            # pytorch /score always requires text candidates
            candidate_texts = [
                self._llm_tok.decode([tid], skip_special_tokens=False)
                for tid in candidate_token_ids
            ]
            return self._score_candidates_pytorch(
                user_content, response_so_far, candidate_texts
            )

        def _prepare_b2_session(
            self,
            req_idx: int,
            user_content: str,
            output_ids: list,
        ) -> int:
            """Initialize or advance the b2 RM session for req_idx.

            Pure Python — no GPU call.  Must be called before
            score_candidates / score_candidates_batch.

            Returns the session id (sid).
            """
            sid = self._b2_sessions.get(req_idx)
            if sid is None:
                chat_prefix_tokens = self._llm_tok.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                bos = self._llm_tok.bos_token_id
                if (bos is not None and chat_prefix_tokens
                        and chat_prefix_tokens[0] == bos):
                    chat_prefix_tokens = chat_prefix_tokens[1:]
                sid = self._rm.new_session(chat_prefix_tokens)
                self._b2_sessions[req_idx] = sid
                self._b2_chat_prefix_len[req_idx] = len(chat_prefix_tokens)

            cur_len = self._rm.session_length(sid)
            chat_prefix_len = self._b2_chat_prefix_len[req_idx]
            n_already = cur_len - chat_prefix_len
            if n_already < 0:
                n_already = 0
            for tid in output_ids[n_already:]:
                self._rm.fix_a_token(sid, int(tid))

            return sid

        def _score_candidates_b2(
            self,
            req_idx: int,
            user_content: str,
            output_ids: list[int],
            candidate_token_ids: list[int],
        ) -> torch.Tensor:
            """In-process RMClient path: session init/advance + scoring.

            Delegates session lifecycle to _prepare_b2_session (pure Python),
            then calls rm.score_candidates for the GPU forward.
            """
            assert self._rm is not None, "RMClient not initialized"
            pf_on = self._PROFILE_DETAIL
            t0 = time.perf_counter() if pf_on else 0.0

            sid = self._prepare_b2_session(req_idx, user_content, output_ids)
            t_advance = time.perf_counter() if pf_on else 0.0

            rewards = self._rm.score_candidates(sid, candidate_token_ids)
            t_score = time.perf_counter() if pf_on else 0.0

            if pf_on:
                self._pf_record("b2_prefix_adv",   (t_advance - t0)          * 1000)
                self._pf_record("b2_score_call",   (t_score   - t_advance)   * 1000)
                self._pf_record("total",           (t_score   - t0)          * 1000)
                self._pf_summary_if_due()
            # D-1: score_candidates already returns a CPU float32 tensor; no longer need to rebuild from list
            return rewards

        def _score_candidates_pytorch(
            self, user_content: str, response_so_far: str,
            candidate_texts: list[str],
        ) -> torch.Tensor:
            """Call sia_rm_server.py POST /score"""
            resp = self._rm_session.post(
                f"{self._RM_URL}/score",
                json={
                    "user_content": user_content,
                    "response_so_far": response_so_far,
                    "candidate_texts": candidate_texts,
                    "request_id": hashlib.md5(user_content.encode()).hexdigest()[:16],
                },
                timeout=120,
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
            return torch.tensor(scores, dtype=torch.float32)

        def _score_candidates_vllm(
            self, req_idx: int, user_content: str, response_so_far: str,
            candidate_token_ids: list[int], output_ids: list[int],
        ) -> torch.Tensor:
            """Call vllm serve POST /classify (activation=false to get raw logits).

            Two paths are selected automatically based on whether LLM/RM tokenizers share the same vocab:

            **A. Direct-token-ids fast path (same vocab, e.g. Qwen3-VL-30B + VM-Qwen3-4B)**
              - Sends `prefix_ids + output_ids + [cand_id]` directly to vllm /classify
              - prefix_ids from _get_manual_prefix_ids: hand-written ChatML, consistent with official sia.py,
                does **not** go through apply_chat_template (avoids Qwen3 enable_thinking=False
                auto-injecting `<think>\\n\\n</think>` empty block causing RM to see OOD prefix)
              - Skips decode/encode; zero BPE boundary perturbation
              - This is the path validated by the 2026-06-03 dual-VM experiment as "98%+ top-1 agreement with official PyTorch VM"

            **B. Text round-trip fallback (cross vocab, e.g. 0GM-35B + VM-Qwen3-4B)**
              - Must decode candidate tokens / accumulated output to text, then let vllm re-encode with RM tokenizer
              - Known BPE boundary precision loss ~1.2 logit, top-1 agreement rate ~82-88%
              - prefix from _get_chat_template_parts (includes _ENABLE_THINKING forwarded to chat_template)

            Fix #1 (2026-06-03): Neither path has an `<|im_end|>` close tag at the end of the prefix,
            consistent with the official ValueModel forward path (avoids partial-response OOD).
            """
            pf_on = self._PROFILE_DETAIL
            t0 = time.perf_counter() if pf_on else 0.0

            if self._same_tokenizer:
                # ==== Path A: direct token IDs (no text round-trip) ====
                prefix_ids = self._get_manual_prefix_ids(req_idx, user_content)
                if prefix_ids is not None:
                    input_payload = [
                        list(prefix_ids) + list(output_ids) + [int(cid)]
                        for cid in candidate_token_ids
                    ]
                    t_format = time.perf_counter() if pf_on else 0.0
                    t_tokenize = t_format  # no separate tokenize step
                else:
                    # encode failed (extremely rare), fall back to text path
                    input_payload = None  # fall through to text path
            else:
                input_payload = None  # cross-vocab -> text path

            if input_payload is None:
                # ==== Path B: text round-trip (cross-tokenizer fallback) ====
                candidate_texts = [
                    self._llm_tok.decode([int(tid)], skip_special_tokens=False)
                    for tid in candidate_token_ids
                ]
                prefix, _suffix_unused = self._get_chat_template_parts(
                    req_idx, user_content,
                )
                if prefix is not None:
                    formatted_texts = [
                        prefix + response_so_far + ct
                        for ct in candidate_texts
                    ]
                else:
                    bos = self._llm_tok.bos_token
                    kwargs = {"tokenize": False, "add_generation_prompt": True}
                    if self._ENABLE_THINKING is not None:
                        kwargs["enable_thinking"] = self._ENABLE_THINKING
                    fallback_prefix = self._llm_tok.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        **kwargs,
                    )
                    if bos and fallback_prefix.startswith(bos):
                        fallback_prefix = fallback_prefix[len(bos):]
                    formatted_texts = [
                        fallback_prefix + response_so_far + ct
                        for ct in candidate_texts
                    ]
                t_format = time.perf_counter() if pf_on else 0.0

                # Cross-vocab: re-encode with RM tokenizer (add_special_tokens=False
                # aligns with official; avoids vllm server injecting BOS/EOS again), or send string
                # directly and let vllm tokenize with default settings (compatible)
                if self._USE_TOKEN_IDS and self._rm_tok is not None:
                    input_payload = [
                        self._rm_tok.encode(t, add_special_tokens=False)
                        for t in formatted_texts
                    ]
                else:
                    input_payload = formatted_texts
                t_tokenize = time.perf_counter() if pf_on else 0.0

            # ==== Step 3: HTTP call ====
            # **Key**: vllm 0.19 /classify schema field name is `use_activation` (not `activation`).
            # The previous wrong field name was silently ignored by Pydantic, defaulting
            # `None`->True => vllm-side sigmoid always on. This compressed RM scores to [0,1] range;
            # sigmoid(x)->1 for x>16 even saturates (float32 ULP limit), requiring EPS-clamp
            # inverse sigmoid to recover — but that recovery saturates entirely for |x|>13.82.
            # 2026-06-03 dual-VM validation: `use_activation=False` retrieves raw logits
            # directly (range ~[-15,+50]); sigmoid saturation issue disappears completely.
            resp = self._rm_session.post(
                f"{self._RM_URL}/classify",
                json={
                    "model": self._RM_MODEL,
                    "input": input_payload,
                    "use_activation": False,   # Key: disable sigmoid to get raw logit
                },
                timeout=120,
            )
            resp.raise_for_status()
            t_http = time.perf_counter() if pf_on else 0.0

            # ==== Step 4: Parse ====
            data = resp.json()["data"]
            # vLLM does not guarantee return order; sort by index
            data_sorted = sorted(data, key=lambda x: x.get("index", 0))
            # `probs` field is already the raw logit (no sigmoid) when use_activation=False;
            # no post-processing needed, use directly.
            scores = [d["probs"][0] for d in data_sorted]
            result = torch.tensor(scores, dtype=torch.float32)
            t_parse = time.perf_counter() if pf_on else 0.0

            # ==== Step 5 (optional): Dual-VM logging hook ====
            # When SIA_DUAL_VM_URL env var is set, the same token_ids payload (path A only)
            # is also forwarded to a standalone PyTorch official VM server, writing both sets
            # of scores side-by-side to SIA_DUAL_VM_LOG (JSONL). For offline dual-VM validation only;
            # does not affect SIA intervention (intervention uses the vllm scores).
            if self._DUAL_VM_URL and self._same_tokenizer and isinstance(input_payload, list) and input_payload and isinstance(input_payload[0], list):
                try:
                    self._dual_vm_call_count += 1
                    dual_resp = self._rm_session.post(
                        self._DUAL_VM_URL,
                        json={"input": input_payload,
                              "request_id": f"req{req_idx}_step{len(output_ids)}"},
                        timeout=120,
                    )
                    dual_data = dual_resp.json()
                    pt_scores = dual_data.get("scores", [])
                    rec = {
                        "ts": time.time(),
                        "req_idx": req_idx,
                        "step": len(output_ids),
                        "n_cand": len(candidate_token_ids),
                        "cand_ids": [int(c) for c in candidate_token_ids],
                        "vllm_scores": [float(s) for s in scores],
                        "pt_scores":   [float(s) for s in pt_scores],
                    }
                    import json
                    with open(self._DUAL_VM_LOG, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    if self._dual_vm_call_count <= 5:
                        print(f"[SIA-dual-vm] error #{self._dual_vm_call_count}: {type(e).__name__}: {e}", flush=True)

            if pf_on:
                self._pf_record("format_chat",     (t_format   - t0)         * 1000)
                self._pf_record("tokenize_client", (t_tokenize - t_format)   * 1000)
                self._pf_record("http_post",       (t_http     - t_tokenize) * 1000)
                self._pf_record("parse_response",  (t_parse    - t_http)     * 1000)
                self._pf_record("total",           (t_parse    - t0)         * 1000)
                self._pf_summary_if_due()
            return result

        # ----------------------------------------------------------------
        # response_so_far incremental decoding
        # ----------------------------------------------------------------
        def _get_response_so_far(self, req_idx: int, output_ids: list) -> str:
            """Decodes only the newly added tail each step; result is equivalent to full re-decode but O(1) per step.

            BPE boundary protection: if the tail ends with \\ufffd (unclosed multi-byte UTF-8),
            do not commit the cache this step; let subsequent tokens decode together with the
            unclosed bytes, ensuring the text sent to the RM is byte-equivalent to a one-shot decode.
            """
            last_n = self._decoded_token_count.get(req_idx, 0)
            cached = self._decoded_text.get(req_idx, "")
            n = len(output_ids)
            if n <= last_n:
                return cached
            tail_ids = output_ids[last_n:]
            try:
                tail_text = self._llm_tok.decode(tail_ids, skip_special_tokens=True)
            except (OverflowError, ValueError) as _dec_err:
                # Rare: vLLM may produce a token ID outside [0, vocab_size) under
                # certain CUDA graph / sampling edge cases at high concurrency.
                # Filter and log so we can identify the root cause without crashing.
                vocab_size = self._llm_tok.vocab_size
                bad = [t for t in tail_ids if not (isinstance(t, int) and 0 <= t < vocab_size)]
                print(
                    f"[SIA] WARNING req={req_idx}: token decode error ({_dec_err}); "
                    f"bad ids={bad[:5]} (first 5 of {len(bad)}), tail_len={len(tail_ids)}. "
                    f"Filtering and retrying.",
                    flush=True,
                )
                safe_ids = [int(t) for t in tail_ids if isinstance(t, int) and 0 <= t < vocab_size]
                tail_text = self._llm_tok.decode(safe_ids, skip_special_tokens=True)
            response = cached + tail_text
            if not tail_text.endswith("�"):
                self._decoded_text[req_idx] = response
                self._decoded_token_count[req_idx] = n
            return response

        # ----------------------------------------------------------------
        # apply: called by vllm once before each token is generated
        # ----------------------------------------------------------------
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            # === per-token profiling start ===
            pf_on = self._PROFILE_DETAIL
            t_apply_start = time.perf_counter() if pf_on else 0.0

            batch_size = logits.shape[0]

            # ==== Batch-wide topk + entropy (submit all GPU work at once) ====
            # The original implementation called .item() per batch item inside a for-loop,
            # triggering a GPU->CPU sync for each one and blocking the LLM forward pipeline.
            # Changed to compute everything outside the loop with a single sync, collapsing
            # batch_size syncs into 1.
            # Per-request effective topk (may differ per request if sia_topk was set).
            # Use the batch maximum so a single torch.topk covers all requests.
            if self._topk_per_req:
                effective_topks = [self._topk_per_req.get(i, self._TOPK) for i in range(batch_size)]
                max_topk = max(effective_topks)
            else:
                effective_topks = None
                max_topk = self._TOPK
            # Entropy gate always uses the global _TOPK window so the threshold
            # remains calibrated regardless of per-request sia_topk. A small
            # sia_topk (e.g. 1) would make softmax entropy near-zero over just
            # one element, wrongly suppressing all interventions.
            entropy_topk_result = torch.topk(logits, self._TOPK, dim=-1)
            log_probs = F.log_softmax(entropy_topk_result.values.float(), dim=-1)
            probs = log_probs.exp()
            entropies = -(probs * log_probs).sum(dim=-1)  # (batch,)
            # Scoring topk may differ from entropy topk when sia_topk is set.
            scoring_topk_result = (
                torch.topk(logits, max_topk, dim=-1)
                if max_topk != self._TOPK
                else entropy_topk_result
            )

            # Note: the above GPU work has not synced yet; actual sync happens below at .cpu().tolist().
            # Split timer: t_after_gpu_dispatch = topk+entropy "Python dispatch" complete
            # (but GPU still running); t_after_sync = GPU pipeline wait complete triggered by .cpu()
            t_after_gpu_dispatch = time.perf_counter() if pf_on else 0.0

            # Single GPU->CPU sync (replaces per-item .item() in the original code).
            # Merge entropy + logit gap into one sync when either dual-gating or DEBUG_HIST
            # needs the gap — avoids a second GPU→CPU round-trip.
            # logit gap = top1_logit - top2_logit (raw logit space; equivalent to log-prob ratio).
            # ===== SIA DEBUG HISTOGRAM START (gap_values also used by dual-gating below) =====
            _need_gap = (self._GAP_THRESHOLD is not None) or self._DEBUG_HIST
            if _need_gap:
                gap_tensor = entropy_topk_result.values[:, 0] - entropy_topk_result.values[:, 1]  # (batch,)
                # Sync entropy + gap together in one call; saves one round-trip vs separate .cpu() calls
                combined = torch.stack([entropies, gap_tensor], dim=1)  # (batch, 2)
                combined_cpu = combined.cpu().tolist()
                entropy_values = [m[0] for m in combined_cpu]
                gap_values = [m[1] for m in combined_cpu]
            else:
                entropy_values = entropies.cpu().tolist()
                gap_values = None  # type: ignore[assignment]
            # ===== SIA DEBUG HISTOGRAM END =====
            t_after_sync = time.perf_counter() if pf_on else 0.0

            # ===== SIA DEBUG HISTOGRAM START =====
            # Bucketing (done after entropy_values are synced to CPU; no new sync introduced)
            if self._DEBUG_HIST:
                for bi in range(batch_size):
                    # Advance <think> / </think> state by checking the last generated token
                    out_ids = self._output_ids.get(bi, [])
                    if out_ids:
                        last_tok = out_ids[-1]
                        if last_tok in self._think_open_ids:
                            self._dbg_in_think[bi] = True
                        elif last_tok in self._think_close_ids:
                            self._dbg_in_think[bi] = False
                    in_think = self._dbg_in_think.setdefault(bi, True)

                    e = entropy_values[bi]
                    g = gap_values[bi]
                    eb = min(int(e * 10), 15)         # 16 bins, step=0.1
                    gb = min(int(g * 2), 15)          # 16 bins, step=0.5
                    self._dbg_entropy_hist.setdefault(bi, [0]*16)[eb] += 1
                    self._dbg_gap_hist.setdefault(bi, [0]*16)[gb] += 1
                    if in_think:
                        self._dbg_entropy_hist_inthink.setdefault(bi, [0]*16)[eb] += 1
                    else:
                        self._dbg_entropy_hist_outthink.setdefault(bi, [0]*16)[eb] += 1
            # ===== SIA DEBUG HISTOGRAM END =====

            # Make SKIP/INTERVENE decisions on CPU (no further sync triggered).
            # Dual-gating: INTERVENE only when entropy gate passes AND gap gate passes.
            #   entropy gate: entropy >= _ENTROPY_THRESHOLD  (or no threshold → always pass)
            #   gap gate:     logit_gap < _GAP_THRESHOLD     (or no threshold → always pass)
            # Either gate can be disabled independently (set to None).
            _gap_thr = self._GAP_THRESHOLD
            if self._ENTROPY_THRESHOLD is not None:
                if _gap_thr is not None:
                    intervene_flags = [
                        e >= self._ENTROPY_THRESHOLD and g < _gap_thr
                        for e, g in zip(entropy_values, gap_values)
                    ]
                else:
                    intervene_flags = [
                        e >= self._ENTROPY_THRESHOLD for e in entropy_values
                    ]
            else:
                if _gap_thr is not None:
                    intervene_flags = [g < _gap_thr for g in gap_values]
                else:
                    intervene_flags = [True] * batch_size
            # Override per-request entropy threshold where set.
            # Per-request thr=None means "force intervene" — bypasses both gates.
            # Per-request thr=value applies entropy gate; gap gate still applies.
            if self._entropy_per_req:
                for i in range(batch_size):
                    if i in self._entropy_per_req:
                        thr = self._entropy_per_req[i]
                        if thr is None:
                            intervene_flags[i] = True  # force: bypass both gates
                        else:
                            entropy_ok = entropy_values[i] >= thr
                            gap_ok = (_gap_thr is None) or (gap_values[i] < _gap_thr)
                            intervene_flags[i] = entropy_ok and gap_ok

            # Only pull topk_indices to CPU if at least one request will INTERVENE; pure SKIP saves a sync.
            # A-2: topk_values_lists no longer needed — flip detection uses GPU argmax path;
            #      verbose log pre_top1/post_top1 only needs topk_indices_lists.
            verbose = self._LOG_LEVEL == "verbose"
            if any(intervene_flags):
                topk_indices_lists = scoring_topk_result.indices.cpu().tolist()
            else:
                topk_indices_lists = None

            # ==== b2 batch pre-scoring ==========================================
            # For the b2 backend only: instead of N sequential score_candidates()
            # calls (each a separate VM forward), collect all INTERVENE requests,
            # advance their sessions (pure Python), and issue one
            # score_candidates_batch() call (single VM forward for N×K prompts).
            #
            # Falls back silently to per-request sequential scoring if the batch
            # call raises — the per-item loop then calls _score_candidates_b2()
            # normally.  Because _prepare_b2_session() was already called for each
            # request, _score_candidates_b2()'s fix_a_token chain is a no-op and
            # the fallback is correct.
            #
            # Non-b2 backends are completely unaffected (b2_batch_scores stays {}).
            b2_batch_scores: dict = {}   # req_idx -> Tensor (K_i,)
            if self._RM_BACKEND == "b2" and topk_indices_lists is not None:
                _b2_reqs: list = []   # (req_idx, sid, candidates_list)
                for _i in range(batch_size):
                    if not intervene_flags[_i]:
                        continue
                    if self._weight_per_req.get(_i, self._WEIGHT) == 0.0:
                        continue
                    _out_ids = list(self._output_ids.get(_i, []))
                    _user    = self._prompt_user.get(_i, "")
                    _cands   = topk_indices_lists[_i]
                    if effective_topks is not None:
                        _eff_k = effective_topks[_i]
                        if _eff_k < max_topk:
                            _cands = _cands[:_eff_k]
                    if self._VM_TOPK is not None and self._VM_TOPK < len(_cands):
                        _cands = _cands[:self._VM_TOPK]
                    _t_prep = time.perf_counter() if pf_on else 0.0
                    try:
                        _sid = self._prepare_b2_session(_i, _user, _out_ids)
                    except Exception as _prep_err:
                        print(
                            f"[SIA] b2 session prep failed req={_i}: {_prep_err}",
                            flush=True,
                        )
                        continue
                    if pf_on:
                        self._pf_record("b2_prefix_adv",
                                        (time.perf_counter() - _t_prep) * 1000)
                    _b2_reqs.append((_i, _sid, _cands))

                if _b2_reqs:
                    _t_batch = time.perf_counter() if pf_on else 0.0
                    # vocab_lowrank head: one prefix forward per chunk → N forwards instead of N×K.
                    # Falls back to N×K inside score_with_vocab_head_batch for cross-tokenizer sessions.
                    _use_vocab_head = (self._VM_HEAD_TYPE == "vocab_lowrank")
                    # Chunk size: SIA_RM_BATCH_CHUNK sessions per call.
                    # Prevents RM KV-cache OOM when batch_size×topk is large
                    # (e.g. 128 sessions × 10 candidates = 1280 prompts).
                    # Each chunk submits at most _B2_BATCH_CHUNK×topk prompts,
                    # keeping peak RM block usage within the pre-allocated pool.
                    # Chunk=0 disables chunking (original unlimited behavior).
                    _chunk_sz = self._B2_BATCH_CHUNK
                    _use_chunks = _chunk_sz > 0 and len(_b2_reqs) > _chunk_sz
                    _chunks = (
                        [_b2_reqs[_s:_s + _chunk_sz]
                         for _s in range(0, len(_b2_reqs), _chunk_sz)]
                        if _use_chunks else [_b2_reqs]
                    )
                    try:
                        for _chunk in _chunks:
                            _payload = [(_sid, _cands) for (_, _sid, _cands) in _chunk]
                            if _use_vocab_head:
                                _chunk_scores = self._rm.score_with_vocab_head_batch(_payload)
                            else:
                                _chunk_scores = self._rm.score_candidates_batch(_payload)
                            for (_req_i, _, _), _s in zip(_chunk, _chunk_scores):
                                b2_batch_scores[_req_i] = _s
                        if pf_on:
                            _batch_wall_ms = (time.perf_counter() - _t_batch) * 1000
                            # Absolute batch wall (one record per INTERVENE step, regardless of N)
                            self._pf_record("b2_batch_wall_abs", _batch_wall_ms)
                            self._pf_record("intv_batch_size", float(len(_b2_reqs)))
                            _total_cands = sum(len(_c) for (_, _, _c) in _b2_reqs)
                            self._pf_record("vm_cands_per_step", float(_total_cands))
                            # Per-request amortized (divide by N so it's comparable to sequential path)
                            _amortized_ms = _batch_wall_ms / len(_b2_reqs)
                            for _ in _b2_reqs:
                                self._pf_record("b2_score_call", _amortized_ms)
                                self._pf_summary_if_due()
                    except Exception as _batch_err:
                        print(
                            f"[SIA] b2 batch scoring failed "
                            f"(falling back to sequential): {_batch_err}",
                            flush=True,
                        )
                        # b2_batch_scores stays empty; per-item loop uses
                        # _score_candidates_b2() normally for each request.
            # ==== end b2 batch pre-scoring =====================================

            # Predicted next token per intervening request (for eager VM prefill).
            # Populated inside the INTERVENE branch; consumed after the loop.
            _eager_intv_best: dict = {}   # req_idx -> post_top1 token

            # ==== Per-item loop: no more .item() / .tolist() syncs inside the loop body ====
            #
            # SKIP path Python cleanup (2026-06-02):
            # Moved output_ids list copy O(N) and user_content dict access into the INTERVENE branch
            # — the SKIP path does not read these values; doing it on every token was wasted work
            # (output_ids grows to 700+ tokens, list copy is not free).
            # _total_steps is still updated every token (needed for stats); req_step is only fetched
            # in verbose mode for logging, deferred to each SKIP/INTERVENE print branch.
            for i in range(batch_size):
                self._total_steps[i] = self._total_steps.get(i, 0) + 1

                entropy = entropy_values[i]

                if not intervene_flags[i]:
                    if verbose:
                        req_step = self._total_steps[i]
                        gap_val = gap_values[i] if gap_values is not None else None
                        if self._ENTROPY_THRESHOLD is not None and entropy < self._ENTROPY_THRESHOLD:
                            skip_reason = f"entropy={entropy:.3f} < {self._ENTROPY_THRESHOLD}"
                        elif gap_val is not None and self._GAP_THRESHOLD is not None and gap_val >= self._GAP_THRESHOLD:
                            skip_reason = f"gap={gap_val:.3f} >= {self._GAP_THRESHOLD}"
                        else:
                            skip_reason = f"entropy={entropy:.3f}"
                        print(
                            f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
                            f"step={req_step:3d} req={i} "
                            f"SKIP ({skip_reason})",
                            flush=True,
                        )
                    continue

                # === INTERVENE path timing start ===
                t_intv_start = time.perf_counter() if pf_on else 0.0

                # State only needed on the INTERVENE path: deferred until here; SKIP doesn't pay.
                output_ids = list(self._output_ids.get(i, []))
                user_content = self._prompt_user.get(i, "")
                req_step = self._total_steps[i]

                # Early exit when weight=0: no RM call, no logit modification,
                # not counted as intervened (noSIA semantics for this request).
                effective_weight = self._weight_per_req.get(i, self._WEIGHT)
                if effective_weight == 0.0:
                    continue

                topk_indices_i = topk_indices_lists[i]            # list[int] (CPU)
                topk_indices_gpu = scoring_topk_result.indices[i]  # GPU view, for indexing logits
                # Per-request topk override: slice both CPU and GPU views
                if effective_topks is not None:
                    eff_k = effective_topks[i]
                    if eff_k < max_topk:
                        topk_indices_i = topk_indices_i[:eff_k]
                        topk_indices_gpu = topk_indices_gpu[:eff_k]

                response_so_far = self._get_response_so_far(i, output_ids)
                t_intv_prepare_end = time.perf_counter() if pf_on else 0.0

                # Use pre-computed batch scores for b2 (fast path), or fall back
                # to per-request scoring for non-b2 backends / batch errors.
                if i in b2_batch_scores:
                    rm_scores = b2_batch_scores[i]
                else:
                    try:
                        rm_scores = self._score_candidates(
                            i, user_content, response_so_far, topk_indices_i,
                            output_ids,
                        )
                    except Exception as e:
                        # RM errors are always printed (rare but important; not affected by LOG_LEVEL)
                        print(
                            f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
                            f"step={req_step:3d} req={i} RM error: {e}",
                            flush=True,
                        )
                        continue

                # Fix #4 (2026-06-03): remove mean-norm; strictly match the official formula
                #   combined[top-k] = orig_logits + rm_scores * weight
                # Official src/sia.py:313 is `combined_scores = rewards * weight + orig_scores`,
                # with no normalization. Our previous `(rm_scores - mean) * weight`, while theoretically
                # shift-invariant under -inf mask + softmax, **changed the mixing scale of RM signal
                # relative to orig_logits** — when raw_rm magnitude is large RM dominates; after normalization
                # it is scaled to a uniform stdev, hurting cross-sample consistency.
                #
                # CRITICAL: cannot use `logits[i, idx].add_(...)` — advanced indexing
                #     (tensor index) returns a copy not a view; .add_ on the copy does not write back to logits,
                #     meaning SIA intervention silently has no effect.
                # === apply_logits timing start (after RM call) ===
                t_intv_apply_start = time.perf_counter() if pf_on else 0.0

                # B-1: rm_scores is GPU bfloat16 in inproc mode (score head output dtype);
                # CPU float32 in legacy mode (read_rewards already cast).
                # Uniformly cast to logits' device + dtype; otherwise subsequent ops raise
                # due to source/self dtype mismatch (e.g. self=float, source=bfloat16).
                if rm_scores.device != logits.device or rm_scores.dtype != logits.dtype:
                    rm_scores = rm_scores.to(logits.device, dtype=logits.dtype)
                rm_deltas = rm_scores * effective_weight

                # === 2026-06-02: strictly align with official / paper baseline ===
                # Official src/sia.py:286-313 semantics:
                #   rewards = -inf everywhere, only top-k positions = raw RM logits
                #   combined = rewards * weight + orig_logits
                #   -> non-top-k positions are always -inf; after softmax prob=0; sampling only picks from top-k
                # Fix: set all non-top-k positions to -inf, equivalent to official.
                topk_vals_i = (scoring_topk_result.values[i] if effective_topks is None
                               else scoring_topk_result.values[i, :effective_topks[i]])
                # vm_topk: rm_scores may cover fewer candidates than topk; trim GPU views to match
                # so index_copy_ and the logit fill are consistent.
                _n_vm = rm_scores.shape[0]
                if _n_vm < topk_vals_i.shape[0]:
                    topk_vals_i = topk_vals_i[:_n_vm]
                    topk_indices_gpu = topk_indices_gpu[:_n_vm]
                modified_topk = topk_vals_i + rm_deltas
                logits[i].fill_(float('-inf'))
                logits[i].index_copy_(0, topk_indices_gpu, modified_topk)

                self._intervened_steps[i] = self._intervened_steps.get(i, 0) + 1

                # top-1 flip detection (stats retained; unaffected by LOG_LEVEL):
                # pre-intervention top-1 = topk_indices_i[0] (already sorted by logit descending);
                # post-intervention top-1 = argmax within the topk (SIA weight magnitude is much smaller
                # than the logit gap within topk; argmax rarely jumps outside topk; intra-topk ordering
                # sufficiently represents actual selection change).
                # A-3: use GPU argmax path; 1 int sync replaces the previous 5-element .tolist() + Python max.
                modified_topk_vals = topk_vals_i + rm_deltas
                post_top1_local = int(modified_topk_vals.argmax().item())
                pre_top1 = topk_indices_i[0]
                post_top1 = topk_indices_i[post_top1_local]
                flipped = pre_top1 != post_top1
                if self._EAGER_VM_PREFILL:
                    _eager_intv_best[i] = post_top1
                if flipped:
                    self._flipped_steps[i] = self._flipped_steps.get(i, 0) + 1

                if verbose:
                    # rm_scores.min()/.max() on a GPU tensor is a reduction,
                    # triggering 2 syncs, but verbose defaults to quiet so this doesn't affect the hot path.
                    print(
                        f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
                        f"step={req_step:3d} req={i} INTERVENE "
                        f"entropy={entropy:.3f} "
                        f"gen_len={len(output_ids)} "
                        f"rm=[{rm_scores.min():.3f}, {rm_scores.max():.3f}] "
                        f"flip={'Y' if flipped else 'N'} "
                        f"pre_top1={pre_top1} post_top1={post_top1}",
                        flush=True,
                    )

                # === intv_apply_logits timing end ===
                if pf_on:
                    t_intv_apply_end = time.perf_counter()
                    self._pf_record("intv_prepare",
                                    (t_intv_prepare_end - t_intv_start) * 1000)
                    self._pf_record("intv_apply_logits",
                                    (t_intv_apply_end - t_intv_apply_start) * 1000)

            # ==== b2 eager VM prefill =====================================================
            # Pre-warm the VM KV cache with predicted next tokens for every active request.
            # Predicted token: post_top1 (highest logit+VM_score) for intervening requests,
            # top-1 raw logit for skip requests.  At the NEXT intervention the APC hits the
            # warmed prefix and only needs to compute the candidate tokens (no tail prefill).
            #
            # Runs at every apply() call (not just INTERVENE steps) so that even skip-step
            # tokens accumulate in the VM cache.  For pure-skip steps where topk_indices_lists
            # is None, one batch argmax is synced to CPU (cheap: 16 scalars vs 16×10 topk).
            #
            # Only warms sessions that are already initialized (_b2_sessions[i] != None);
            # first-intervention requests still pay full prefill cost.
            if self._RM_BACKEND == "b2" and self._EAGER_VM_PREFILL:
                _t_eager = time.perf_counter() if pf_on else 0.0
                # For pure skip steps, batch-sync argmax once (avoids 16 separate .item() calls).
                _argmax_all: list = (
                    logits.argmax(dim=-1).cpu().tolist()
                    if topk_indices_lists is None else []
                )
                _eager_reqs: list = []
                for _i in range(batch_size):
                    _sid = self._b2_sessions.get(_i)
                    if _sid is None:
                        continue  # session not yet initialized; first intervene will do full prefill
                    # Advance VM session with any tokens generated since the last fix_a_token call.
                    _out = self._output_ids.get(_i, [])
                    _n_vm = max(0, self._rm.session_length(_sid)
                                   - self._b2_chat_prefix_len.get(_i, 0))
                    for _tid in _out[_n_vm:]:
                        self._rm.fix_a_token(_sid, int(_tid))
                    # Predicted next token for this request.
                    if _i in _eager_intv_best:
                        _pred = _eager_intv_best[_i]      # intervening: post-intervention best
                    elif topk_indices_lists is not None:
                        _pred = topk_indices_lists[_i][0]  # skip at intervene step: top-1 raw logit
                    else:
                        _pred = _argmax_all[_i]            # pure skip step: batch argmax
                    _eager_reqs.append((_sid, _pred))
                if _eager_reqs:
                    self._rm.eager_prefill_batch(_eager_reqs)
                if pf_on:
                    self._pf_record("b2_eager_wall",
                                    (time.perf_counter() - _t_eager) * 1000)
            # ==== end b2 eager VM prefill ================================================

            # === apply() exit: record overall timing + distinguish SKIP/INTERVENE paths ===
            if pf_on:
                t_apply_end = time.perf_counter()
                _apply_ms = (t_apply_end - t_apply_start) * 1000
                self._pf_record("apply_total", _apply_ms)
                self._pf_record("apply_topk_ent",
                                (t_after_gpu_dispatch - t_apply_start) * 1000)
                self._pf_record("apply_cpu_sync",
                                (t_after_sync - t_after_gpu_dispatch) * 1000)
                _any_intv = any(intervene_flags)
                if _any_intv:
                    self._pf_record("apply_intv_step", _apply_ms)
                else:
                    self._pf_record("skip_step", _apply_ms)
                self._pf_apply_calls += 1
                self._pf_total_req_steps += batch_size
                # apply is called ~3x more often than INTERVENE (33% intervention rate);
                # just wait for the INTERVENE-driven summary trigger: by then apply has accumulated
                # 300+ calls and SKIP data has converged. New phases (apply_*, intv_*, skip_step)
                # are naturally included in the summary output.

            return logits

        def is_argmax_invariant(self) -> bool:
            return False

        # ----------------------------------------------------------------
        # update_state: called by vllm to inform the processor of batch changes
        # ----------------------------------------------------------------
        def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
            if batch_update is None:
                return

            for idx in batch_update.removed:
                total = self._total_steps.pop(idx, 0)
                intervened = self._intervened_steps.pop(idx, 0)
                flipped = self._flipped_steps.pop(idx, 0)
                ratio = intervened / total if total > 0 else 0.0
                flip_ratio = flipped / intervened if intervened > 0 else 0.0
                print(
                    f"[SIA] req={idx} DONE  "
                    f"intervened={intervened}/{total}  ratio={ratio:.1%}  "
                    f"top1_flip={flipped}/{intervened} ({flip_ratio:.1%})",
                    flush=True,
                )
                # ===== SIA DEBUG HISTOGRAM START =====
                if self._DEBUG_HIST:
                    eh = self._dbg_entropy_hist.pop(idx, None)
                    gh = self._dbg_gap_hist.pop(idx, None)
                    eh_in = self._dbg_entropy_hist_inthink.pop(idx, None)
                    eh_out = self._dbg_entropy_hist_outthink.pop(idx, None)
                    self._dbg_in_think.pop(idx, None)
                    if eh and total > 0:
                        # Compute fraction of steps with entropy >= threshold, for cross-checking against ratio
                        thr = self._ENTROPY_THRESHOLD or 1.0
                        thr_bin = min(int(thr * 10), 15)
                        ge_thr = sum(eh[thr_bin:])
                        ge_thr_pct = ge_thr / total * 100
                        in_total = sum(eh_in or [])
                        out_total = sum(eh_out or [])
                        print(
                            f"[SIA-debug] req={idx} ENTROPY_HIST [0..1.6 step=0.1] = {eh}  "
                            f"(ge_{thr:.1f}={ge_thr}/{total}={ge_thr_pct:.2f}%)\n"
                            f"[SIA-debug] req={idx} GAP_HIST     [0..8.0 step=0.5] = {gh}\n"
                            f"[SIA-debug] req={idx} ENTROPY_INTHINK  (n={in_total}): {eh_in}\n"
                            f"[SIA-debug] req={idx} ENTROPY_OUTTHINK (n={out_total}): {eh_out}",
                            flush=True,
                        )
                # ===== SIA DEBUG HISTOGRAM END =====
                self._output_ids.pop(idx, None)
                self._prompt_user.pop(idx, None)
                self._weight_per_req.pop(idx, None)
                self._topk_per_req.pop(idx, None)
                self._entropy_per_req.pop(idx, None)
                self._decoded_text.pop(idx, None)
                self._decoded_token_count.pop(idx, None)
                self._chat_prefix_per_req.pop(idx, None)
                self._chat_suffix_per_req.pop(idx, None)
                self._manual_prefix_ids_per_req.pop(idx, None)
                # b2: release RM session
                if self._rm is not None:
                    sid = self._b2_sessions.pop(idx, None)
                    self._b2_chat_prefix_len.pop(idx, None)
                    if sid is not None:
                        try:
                            self._rm.end_session(sid)
                        except Exception:
                            pass

            if batch_update.moved:
                old_out = dict(self._output_ids)
                old_prompt = dict(self._prompt_user)
                old_weight = dict(self._weight_per_req)
                old_topk = dict(self._topk_per_req)
                old_entropy = dict(self._entropy_per_req)
                old_total = dict(self._total_steps)
                old_intervened = dict(self._intervened_steps)
                old_flipped = dict(self._flipped_steps)
                old_decoded = dict(self._decoded_text)
                old_decoded_n = dict(self._decoded_token_count)
                old_chat_prefix = dict(self._chat_prefix_per_req)
                old_chat_suffix = dict(self._chat_suffix_per_req)
                old_manual_prefix = dict(self._manual_prefix_ids_per_req)
                old_b2_sessions = dict(self._b2_sessions)
                old_b2_prefix_len = dict(self._b2_chat_prefix_len)
                for i1, i2, directionality in batch_update.moved:
                    if directionality == MoveDirectionality.UNIDIRECTIONAL:
                        self._output_ids[i2] = old_out.get(i1, [])
                        self._prompt_user[i2] = old_prompt.get(i1, "")
                        if i1 in old_weight:
                            self._weight_per_req[i2] = old_weight[i1]
                        if i1 in old_topk:
                            self._topk_per_req[i2] = old_topk[i1]
                        else:
                            self._topk_per_req.pop(i2, None)
                        if i1 in old_entropy:
                            self._entropy_per_req[i2] = old_entropy[i1]
                        else:
                            self._entropy_per_req.pop(i2, None)
                        self._total_steps[i2] = old_total.get(i1, 0)
                        self._intervened_steps[i2] = old_intervened.get(i1, 0)
                        self._flipped_steps[i2] = old_flipped.get(i1, 0)
                        if i1 in old_decoded:
                            self._decoded_text[i2] = old_decoded[i1]
                            self._decoded_token_count[i2] = old_decoded_n.get(i1, 0)
                        if i1 in old_chat_prefix:
                            self._chat_prefix_per_req[i2] = old_chat_prefix[i1]
                            self._chat_suffix_per_req[i2] = old_chat_suffix.get(i1)
                        if i1 in old_manual_prefix:
                            self._manual_prefix_ids_per_req[i2] = old_manual_prefix[i1]
                        # b2 session: move i1 → i2
                        if i1 in old_b2_sessions:
                            self._b2_sessions[i2] = old_b2_sessions[i1]
                            self._b2_chat_prefix_len[i2] = old_b2_prefix_len.get(i1, 0)
                        # Always clear source slot — a new request arriving at i1
                        # must start a fresh session, not inherit the moved one.
                        self._b2_sessions.pop(i1, None)
                        self._b2_chat_prefix_len.pop(i1, None)
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
                        if i2 in old_topk:
                            self._topk_per_req[i1] = old_topk[i2]
                        else:
                            self._topk_per_req.pop(i1, None)
                        if i1 in old_topk:
                            self._topk_per_req[i2] = old_topk[i1]
                        else:
                            self._topk_per_req.pop(i2, None)
                        if i2 in old_entropy:
                            self._entropy_per_req[i1] = old_entropy[i2]
                        else:
                            self._entropy_per_req.pop(i1, None)
                        if i1 in old_entropy:
                            self._entropy_per_req[i2] = old_entropy[i1]
                        else:
                            self._entropy_per_req.pop(i2, None)
                        self._total_steps[i1] = old_total.get(i2, 0)
                        self._total_steps[i2] = old_total.get(i1, 0)
                        self._intervened_steps[i1] = old_intervened.get(i2, 0)
                        self._intervened_steps[i2] = old_intervened.get(i1, 0)
                        self._flipped_steps[i1] = old_flipped.get(i2, 0)
                        self._flipped_steps[i2] = old_flipped.get(i1, 0)
                        if i2 in old_decoded:
                            self._decoded_text[i1] = old_decoded[i2]
                            self._decoded_token_count[i1] = old_decoded_n.get(i2, 0)
                        else:
                            self._decoded_text.pop(i1, None)
                            self._decoded_token_count.pop(i1, None)
                        if i1 in old_decoded:
                            self._decoded_text[i2] = old_decoded[i1]
                            self._decoded_token_count[i2] = old_decoded_n.get(i1, 0)
                        else:
                            self._decoded_text.pop(i2, None)
                            self._decoded_token_count.pop(i2, None)
                        if i2 in old_chat_prefix:
                            self._chat_prefix_per_req[i1] = old_chat_prefix[i2]
                            self._chat_suffix_per_req[i1] = old_chat_suffix.get(i2)
                        else:
                            self._chat_prefix_per_req.pop(i1, None)
                            self._chat_suffix_per_req.pop(i1, None)
                        if i1 in old_chat_prefix:
                            self._chat_prefix_per_req[i2] = old_chat_prefix[i1]
                            self._chat_suffix_per_req[i2] = old_chat_suffix.get(i1)
                        else:
                            self._chat_prefix_per_req.pop(i2, None)
                            self._chat_suffix_per_req.pop(i2, None)
                        # manual prefix ids (C3): SWAP
                        if i2 in old_manual_prefix:
                            self._manual_prefix_ids_per_req[i1] = old_manual_prefix[i2]
                        else:
                            self._manual_prefix_ids_per_req.pop(i1, None)
                        if i1 in old_manual_prefix:
                            self._manual_prefix_ids_per_req[i2] = old_manual_prefix[i1]
                        else:
                            self._manual_prefix_ids_per_req.pop(i2, None)
                        # b2 sessions: SWAP
                        if i2 in old_b2_sessions:
                            self._b2_sessions[i1] = old_b2_sessions[i2]
                            self._b2_chat_prefix_len[i1] = old_b2_prefix_len.get(i2, 0)
                        else:
                            self._b2_sessions.pop(i1, None)
                            self._b2_chat_prefix_len.pop(i1, None)
                        if i1 in old_b2_sessions:
                            self._b2_sessions[i2] = old_b2_sessions[i1]
                            self._b2_chat_prefix_len[i2] = old_b2_prefix_len.get(i1, 0)
                        else:
                            self._b2_sessions.pop(i2, None)
                            self._b2_chat_prefix_len.pop(i2, None)

            for idx, params, prompt_ids, output_ids in batch_update.added:
                self._output_ids[idx] = output_ids
                self._prompt_user[idx] = self._extract_user_content(list(prompt_ids))
                extra = (params.extra_args or {}) if params is not None else {}
                if "sia_weight" in extra:
                    self._weight_per_req[idx] = float(extra["sia_weight"])
                if "sia_topk" in extra:
                    self._topk_per_req[idx] = int(extra["sia_topk"])
                if "sia_entropy_threshold" in extra:
                    self._entropy_per_req[idx] = extra["sia_entropy_threshold"]
                # ===== SIA DEBUG HISTOGRAM START =====
                # New request entering slot; clear any residual dbg state from the previous request
                # (safety net; normally cleared by removed handler)
                if self._DEBUG_HIST:
                    self._dbg_entropy_hist.pop(idx, None)
                    self._dbg_gap_hist.pop(idx, None)
                    self._dbg_entropy_hist_inthink.pop(idx, None)
                    self._dbg_entropy_hist_outthink.pop(idx, None)
                    self._dbg_in_think[idx] = True   # default: inside think block (chat template starts with <think>)
                # ===== SIA DEBUG HISTOGRAM END =====

    return SIALogitsProcessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SIA vllm intervention (HTTP RM Server version)")
    p.add_argument("--llm",       required=True,  help="LLM model path")
    p.add_argument("--rm_url",    default="http://localhost:8001",
                   help="RM server address (default http://localhost:8001)")
    p.add_argument("--rm_backend",
                   choices=["pytorch", "vllm", "b2"],
                   default="pytorch",
                   help="RM backend: pytorch=custom /score; vllm=vllm serve /classify; "
                        "b2=in-process RMClient (same process as LLM, no HTTP overhead)")
    p.add_argument("--rm_model",  default=None,
                   help="required when rm_backend in {vllm, b2}: RM model path")
    p.add_argument("--rm_b2_gpu_mem", type=float, default=0.3,
                   help="gpu_memory_utilization for the RM vLLM instance under b2 backend (default 0.3)")
    p.add_argument("--llm_gpu_mem", type=float, default=0.5,
                   help="vllm gpu_memory_utilization (default 0.5)")
    p.add_argument("--topk",      type=int,   default=10)
    p.add_argument("--vm_topk",   type=int,   default=None,
                   help="Limit candidates sent to VM (default: same as --topk). "
                        "Keeps --topk for entropy gate so intervention rate is unchanged; "
                        "sends only vm_topk candidates to VM, reducing VM batch size. "
                        "Used for FaRMA proxy benchmarking.")
    p.add_argument("--weight",    type=float, default=1.0)
    p.add_argument("--max_tokens", type=int,  default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--entropy_threshold", type=float, default=None)
    p.add_argument("--logit_gap_threshold", type=float, default=None,
                   help="Dual-gate: skip VM scoring when top1-top2 logit gap >= this value "
                        "(None=disabled; 0 would skip everything since gap is always>=0). "
                        "Reduces intervention rate. Requires AlpacaEval A/B validation before "
                        "production use.")
    p.add_argument("--eager_vm_prefill", action="store_true", default=False,
                   help="b2 backend: after each decode step pre-warm the VM KV cache with the "
                        "predicted next token for all active requests, so the next INTERVENE "
                        "finds a warm prefix and only needs to compute candidate tokens.")
    p.add_argument("--vm_head_type", default="scalar",
                   choices=["scalar", "vocab_lowrank"],
                   help="b2 backend: VM reward head type. "
                        "'scalar' (default): K RM forwards per INTERVENE step (original). "
                        "'vocab_lowrank': 1 RM forward per step + vocab indexing (FaRMA, ~K× speedup). "
                        "Requires a checkpoint converted with convert_rm_for_vllm.py --head_type vocab_lowrank.")
    p.add_argument("--vm_head_rank", type=int, default=64,
                   help="Rank for vocab_lowrank head (default 64, ignored for scalar head).")
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
          f"entropy_threshold={args.entropy_threshold}  "
          f"logit_gap_threshold={args.logit_gap_threshold}")
    print("=" * 60)

    SIAProcessor = make_sia_processor(
        rm_url=args.rm_url,
        topk=args.topk,
        weight=args.weight,
        entropy_threshold=args.entropy_threshold,
        logit_gap_threshold=args.logit_gap_threshold,
        rm_backend=args.rm_backend,
        rm_model=args.rm_model,
        rm_b2_gpu_mem=args.rm_b2_gpu_mem,
        eager_vm_prefill=args.eager_vm_prefill,
        vm_head_type=args.vm_head_type,
        vm_head_rank=args.vm_head_rank,
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

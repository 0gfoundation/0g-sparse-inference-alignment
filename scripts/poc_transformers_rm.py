#!/usr/bin/env python3
"""
P2 PoC: transformers + DynamicCache 替代 nested vllm.LLM() 的 VM scoring 路径

当前 b2 inproc 路径（vllm 嵌套）延迟分解（conc=16, n_sess=3.2, K=10）：
    GPU 计算:    ~1.8ms  ( 7%)
    vllm overhead: ~26.7ms (93%)  ← vllm scheduler / dispatch / IPC
    总计:        ~28.5ms

本 PoC 测试路径：
    transformers Qwen3ForSequenceClassification + DynamicCache
    直接调用 model.forward()，完全绕过 vllm scheduler

已修复的 Bug（v2）：
    Bug 1: score_candidates_with_kv 做 K 次顺序 forward（K×4ms≈40ms）→
           score_candidates_batch 一次 [K,1] batched forward（~4ms）
    Bug 2: DynamicCache 被顺序 forward 污染（_seen_tokens 漂移）→
           每次展开临时 expanded_kv，prefix_kv 始终不变

预期延迟（prefix 已缓存，bugs 修复后）：
    prefix KV 缓存命中 → 一次 [K, 1] batched forward
    K=10 并行 → ~4ms GPU + <1ms Python = ~4-5ms / session
    3.2 sessions × ~5ms = ~16ms（vs b2 inproc ~28.5ms）

为什么 vllm 0.18.0 不会有 CUDA graph 冲突：
    vllm 0.18.0 使用 AOT piecewise graphs（启动时一次性捕获，推理时只 replay）。
    推理期间只调用 cudaGraphLaunch，不进入 torch.cuda.graph() capture context，
    所以 transformers.forward() 可以安全并发。
    （vllm 0.19+ 改为 runtime capture，才会出现冲突。）

用法：
    # 基础测试（eager 模式，不需要 vllm）
    python scripts/poc_transformers_rm.py \\
        --vm_model /workspace/models/VM-Qwen3-4B-merged-for-vllm

    # 完整测试（含 prefix 增量更新模拟）
    python scripts/poc_transformers_rm.py \\
        --vm_model /workspace/models/VM-Qwen3-4B-merged-for-vllm \\
        --full
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="P2 PoC: transformers VM scoring vs b2 inproc baseline"
    )
    p.add_argument("--vm_model", required=True,
                   help="VM-Qwen3-4B-merged-for-vllm 路径")
    p.add_argument("--device", default="cuda:0",
                   help="GPU 设备（default: cuda:0）")
    p.add_argument("--prefix_len", type=int, default=480,
                   help="模拟前缀长度（default: 480）")
    p.add_argument("--topk", type=int, default=10,
                   help="候选数 K（default: 10）")
    p.add_argument("--n_warmup", type=int, default=10,
                   help="预热次数（default: 10）")
    p.add_argument("--n_trials", type=int, default=50,
                   help="计时次数（default: 50）")
    p.add_argument("--full", action="store_true",
                   help="额外运行增量前缀更新模拟（真实 SIA 场景）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _rand_ids(n, seed, vocab_size=32000):
    random.seed(seed)
    return [random.randint(1, vocab_size - 1) for _ in range(n)]


def _ids_tensor(ids, device):
    return torch.tensor(ids, dtype=torch.long, device=device)


def _expand_kv_for_batch(prefix_kv, K):
    """
    batch=1 KV cache 扩展为 batch=K，兼容多版本 transformers DynamicCache API。

    transformers 不同版本内部存储不一致：
    - 4.36-4.49: key_cache / value_cache 列表
    - 4.50+:     to_legacy_cache() + update() 是更稳定的公开 API
    """
    from transformers import DynamicCache

    # ── 读取各层 (k, v) ─────────────────────────────────────────────
    pairs = []

    # 优先：to_legacy_cache（官方 read API，适用多版本）
    if hasattr(prefix_kv, 'to_legacy_cache'):
        try:
            pairs = list(prefix_kv.to_legacy_cache())
        except Exception:
            pass

    # 次选：key_cache / value_cache（4.36-4.49 典型 API）
    if not pairs and hasattr(prefix_kv, 'key_cache'):
        pairs = list(zip(prefix_kv.key_cache, prefix_kv.value_cache))

    # 次选：__getitem__ + __len__（Cache 基类接口）
    if not pairs and hasattr(prefix_kv, '__len__') and hasattr(prefix_kv, '__getitem__'):
        pairs = [prefix_kv[i] for i in range(len(prefix_kv))]

    # fallback：已经是 tuple/list
    if not pairs and isinstance(prefix_kv, (tuple, list)):
        pairs = list(prefix_kv)

    if not pairs:
        attrs = [a for a in dir(prefix_kv) if not a.startswith('_')]
        raise TypeError(
            f"无法读取 KV cache，类型={type(prefix_kv).__name__}，"
            f"可用属性: {attrs}"
        )

    # ── 构建 batch=K 的新 DynamicCache ──────────────────────────────
    expanded = DynamicCache()
    for layer_idx, layer_kv in enumerate(pairs):
        k, v = layer_kv[0], layer_kv[1]
        k_exp = k.expand(K, -1, -1, -1).contiguous()
        v_exp = v.expand(K, -1, -1, -1).contiguous()
        if hasattr(expanded, 'update'):
            expanded.update(k_exp, v_exp, layer_idx)
        else:
            expanded.key_cache.append(k_exp)
            expanded.value_cache.append(v_exp)

    if hasattr(prefix_kv, '_seen_tokens') and hasattr(expanded, '_seen_tokens'):
        expanded._seen_tokens = prefix_kv._seen_tokens

    return expanded


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_vm(vm_model_path, device):
    from transformers import AutoModelForSequenceClassification

    print(f"加载 VM: {vm_model_path} → {device}")
    t0 = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        vm_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"加载完成 ({time.perf_counter()-t0:.1f}s)")
    return model


# ---------------------------------------------------------------------------
# 评分函数
# ---------------------------------------------------------------------------

def compute_prefix_kv(model, prefix_ids_tensor):
    """
    计算前缀 KV cache（cold pass）。
    返回 past_key_values（可安全复用于多个 candidate forward）。
    """
    with torch.inference_mode():
        out = model.model(
            input_ids=prefix_ids_tensor.unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    return out.past_key_values


def score_candidates_batch(model, prefix_kv, candidate_ids, device):
    """
    K 个候选 token 一次 batched forward（Bug 1 + Bug 2 修复版）。

    原 score_candidates_with_kv 做 K 次顺序 forward（K×4ms≈40ms）。
    本函数把 prefix_kv 扩展到 batch=K，一次 forward 得到所有 K 个分数（~4ms）。

    DynamicCache 不污染：expanded_kv 为临时对象，prefix_kv 始终不变。
    """
    K = len(candidate_ids)
    cand_input = torch.tensor(
        [[c] for c in candidate_ids], dtype=torch.long, device=device
    )  # [K, 1]
    expanded_kv = _expand_kv_for_batch(prefix_kv, K)
    with torch.inference_mode():
        out = model(
            input_ids=cand_input,
            past_key_values=expanded_kv,
            use_cache=False,
        )
    return out.logits.squeeze(-1).cpu().tolist()  # [K]


def _score_candidates_sequential(model, prefix_kv, candidate_ids, device):
    """
    原始顺序评分（仅用于 test_smoke 交叉验证，不用于 benchmark）。
    Bug: K 次顺序 forward + DynamicCache 污染。
    """
    scores = []
    with torch.inference_mode():
        for cand_id in candidate_ids:
            cand = torch.tensor([[cand_id]], dtype=torch.long, device=device)
            out = model(
                input_ids=cand,
                past_key_values=prefix_kv,
                use_cache=False,
            )
            scores.append(out.logits.item())
    return scores


def score_candidates_naive_batch(model, prefix_ids_tensor, candidate_ids, device):
    """
    批量全量 forward（所有 [prefix + cand_i] 一次 batched forward）。
    不用 KV cache，适合短 prefix 或 n_sess 较多的场景。
    """
    prefix = prefix_ids_tensor.to(device)
    K = len(candidate_ids)
    all_ids = torch.stack([
        torch.cat([prefix, torch.tensor([c], device=device)])
        for c in candidate_ids
    ])  # [K, prefix_len+1]
    with torch.inference_mode():
        out = model(input_ids=all_ids)
    return out.logits.squeeze(-1).cpu().tolist()


# ---------------------------------------------------------------------------
# 测试 1：冒烟测试
# ---------------------------------------------------------------------------

def test_smoke(model, device):
    print("\n[1] 冒烟测试")

    prefix = _ids_tensor(_rand_ids(20, seed=0), device)
    cands_a = [500, 501]

    kv = compute_prefix_kv(model, prefix)

    # 批量评分（新实现）
    scores_batch = score_candidates_batch(model, kv, cands_a, device)
    print(f"  batch 候选 {cands_a}: scores = {[f'{s:.4f}' for s in scores_batch]}")

    # 交叉验证：batch vs sequential（应基本一致，允许 bfloat16 误差）
    scores_seq = _score_candidates_sequential(model, kv, cands_a, device)
    diffs = [abs(b - s) for b, s in zip(scores_batch, scores_seq)]
    if max(diffs) < 5e-2:
        print(f"  ✅ batch vs sequential 一致（max_diff={max(diffs):.2e}）")
    else:
        print(f"  ⚠️  batch vs sequential 差异: {diffs}")

    # 确定性检查（两次 batch 调用，prefix_kv 不变）
    s1 = score_candidates_batch(model, kv, [500], device)[0]
    s2 = score_candidates_batch(model, kv, [500], device)[0]
    diff = abs(s1 - s2)
    if diff < 1e-5:
        print(f"  ✅ 确定性 OK（同 prompt 两次 batch 差值 {diff:.2e}）")
    else:
        print(f"  ⚠️  非确定性: 差值 {diff:.6f}（DynamicCache 污染未修复？）")

    # 不同前缀 → 分数应不同
    prefix2 = _ids_tensor(_rand_ids(20, seed=999), device)
    kv2 = compute_prefix_kv(model, prefix2)
    sa = score_candidates_batch(model, kv, [500], device)[0]
    sb = score_candidates_batch(model, kv2, [500], device)[0]
    print(f"  同候选 500，不同前缀: {sa:.4f} vs {sb:.4f}")

    # score head 区分度
    if abs(scores_batch[0] - scores_batch[1]) > 1e-4:
        print(f"  ✅ score head 有区分度（diff={abs(scores_batch[0]-scores_batch[1]):.4f}）")
    else:
        print(f"  ⚠️  两个候选分数完全相同，score head 可能未生效")

    return True


# ---------------------------------------------------------------------------
# 测试 2：prefix KV 缓存有效性（cold vs warm）
# ---------------------------------------------------------------------------

def test_prefix_cache(model, prefix_len, topk, device):
    print(f"\n[2] Prefix KV 缓存有效性（prefix_len={prefix_len}, topk={topk}）")

    cands = _rand_ids(topk, seed=42)

    # Cold: 每次用新前缀（需重新计算 prefix KV）
    cold_times = []
    for i in range(15):
        prefix = _ids_tensor(_rand_ids(prefix_len, seed=i * 7777), device)
        t0 = time.perf_counter()
        kv = compute_prefix_kv(model, prefix)
        score_candidates_batch(model, kv, cands, device)
        cold_times.append((time.perf_counter() - t0) * 1000)

    # Warm: 同一前缀（prefix KV 复用，不重新计算）
    prefix_warm = _ids_tensor(_rand_ids(prefix_len, seed=8888), device)
    kv_warm = compute_prefix_kv(model, prefix_warm)
    warm_times = []
    for _ in range(15):
        t0 = time.perf_counter()
        score_candidates_batch(model, kv_warm, cands, device)
        warm_times.append((time.perf_counter() - t0) * 1000)

    cold_p50 = float(np.percentile(cold_times, 50))
    warm_p50 = float(np.percentile(warm_times, 50))
    ratio = cold_p50 / warm_p50 if warm_p50 > 0 else 0

    print(f"  cold (prefix 未缓存): p50 = {cold_p50:6.1f}ms")
    print(f"  warm (prefix 已缓存): p50 = {warm_p50:6.1f}ms")
    print(f"  Speedup: {ratio:.1f}×", end="")
    if ratio >= 5:
        print("  ✅ prefix KV 缓存效果显著")
    elif ratio >= 2:
        print("  ⚠️  prefix KV 缓存有效果，但不如预期")
    else:
        print("  ❌ prefix KV 缓存无效")

    return warm_p50


# ---------------------------------------------------------------------------
# 测试 3：延迟 benchmark（对比 b2 inproc 基线）
# ---------------------------------------------------------------------------

def benchmark_latency(model, prefix_len, topk, n_warmup, n_trials, device):
    """
    模拟 SIA conc=16 下的 batch scoring：M sessions × K candidates。
    prefix KV 已提前缓存（warm state）。
    """
    # b2 inproc eager 基线（实测）
    b2_ref = {1: 9.0, 2: 14.0, 3: 18.0, 4: 21.0, 8: 28.5}

    print(f"\n[3] 延迟 benchmark（prefix_len={prefix_len}, topk={topk}）")
    print(f"  {'n_sess':>6}  {'batch':>5}  {'p50':>8}  {'p90':>8}  "
          f"{'amort':>8}  {'vs b2_inproc':>12}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")

    for n_sess in [1, 2, 3, 4, 8]:
        # 预计算 n_sess 个不同的 prefix KV（warm state）
        prefix_kvs = []
        for s in range(n_sess):
            prefix = _ids_tensor(_rand_ids(prefix_len, seed=s * 100), device)
            kv = compute_prefix_kv(model, prefix)
            prefix_kvs.append(kv)

        cands_per_sess = [
            _rand_ids(topk, seed=s * 999) for s in range(n_sess)
        ]

        # 预热
        for _ in range(n_warmup):
            for s in range(n_sess):
                score_candidates_batch(model, prefix_kvs[s], cands_per_sess[s], device)

        # 计时
        torch.cuda.synchronize(device)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            for s in range(n_sess):
                score_candidates_batch(model, prefix_kvs[s], cands_per_sess[s], device)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = float(np.percentile(times, 50))
        p90 = float(np.percentile(times, 90))
        amort = p50 / n_sess
        ref = b2_ref.get(n_sess)
        ref_str = f"{p50/ref:.2f}×" if ref else "—"

        print(f"  {n_sess:>6}  {n_sess*topk:>5}  "
              f"{p50:>7.1f}ms  {p90:>7.1f}ms  "
              f"{amort:>7.1f}ms  {ref_str:>12}")


# ---------------------------------------------------------------------------
# 测试 4（--full）：增量前缀更新模拟（真实 SIA 场景）
# ---------------------------------------------------------------------------

def test_incremental_prefix(model, prefix_len, topk, n_steps, device):
    """
    模拟真实 SIA decode 流程：
    每步生成 1 个 token → prefix 增长 1 → 重新对 K 个候选评分。

    使用 DynamicCache 增量追加（无需重新计算整个 prefix）。
    评分步骤用 _expand_kv_for_batch 隔离，避免 DynamicCache 污染。
    """
    from transformers import DynamicCache

    print(f"\n[4] 增量前缀更新模拟（{n_steps} 步, topk={topk}）")

    K = topk
    initial_prefix = _ids_tensor(_rand_ids(prefix_len, seed=1234), device)

    # 构建初始 DynamicCache
    cache = DynamicCache()
    with torch.inference_mode():
        model.model(
            input_ids=initial_prefix.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    vocab_size = 32000
    step_times = []

    for step in range(n_steps):
        new_token_id = random.randint(1, vocab_size - 1)
        cands = _rand_ids(K, seed=step * 333)

        t0 = time.perf_counter()

        # 1. 将新生成的 token 追加到 DynamicCache（增量，非全量重算）
        with torch.inference_mode():
            model.model(
                input_ids=torch.tensor([[new_token_id]], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

        # 2. 对 K 个候选评分（批量 forward，不污染 cache）
        expanded_kv = _expand_kv_for_batch(cache, K)
        cand_input = torch.tensor(
            [[c] for c in cands], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            model(
                input_ids=cand_input,
                past_key_values=expanded_kv,
                use_cache=False,
            )

        torch.cuda.synchronize(device)
        step_times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(step_times, 50))
    p90 = float(np.percentile(step_times, 90))
    print(f"  每步（prefix+1 + K 候选批量评分）p50={p50:.1f}ms  p90={p90:.1f}ms")
    print(f"  vs b2 inproc (~9ms @ n_sess=1): {p50/9:.2f}×")
    print(f"  cache 长度增长: {prefix_len} → {prefix_len + n_steps} tokens")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 65)
    print("P2 PoC: transformers + DynamicCache VM scoring (v2 batched)")
    print("=" * 65)
    print(f"  vm_model:   {args.vm_model}")
    print(f"  device:     {args.device}")
    print(f"  prefix_len: {args.prefix_len}  topk: {args.topk}")

    if not os.path.isdir(args.vm_model):
        print(f"\n❌ vm_model 路径不存在: {args.vm_model}")
        sys.exit(1)

    # CUDA graph 冲突检查（仅供参考）
    print("\n[0] CUDA graph 冲突检查")
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_current_stream_capturing"):
        if torch.cuda.is_current_stream_capturing():
            print("  ⚠️  当前在 CUDA graph capture context 内 — transformers forward 可能失败")
        else:
            print("  ✅ 不在 CUDA graph capture context 内 — 可安全运行")
    else:
        print("  INFO: vllm 0.18.0 使用 AOT graphs，推理时不进入 capture context，无冲突风险")

    # 加载模型
    model = load_vm(args.vm_model, args.device)

    # 测试
    test_smoke(model, args.device)
    warm_p50 = test_prefix_cache(model, args.prefix_len, args.topk, args.device)
    benchmark_latency(
        model, args.prefix_len, args.topk,
        args.n_warmup, args.n_trials, args.device
    )

    if args.full:
        test_incremental_prefix(
            model, args.prefix_len, args.topk,
            n_steps=50, device=args.device
        )

    print(f"\n{'='*65}")
    print("结论参考（修复 batch forward 后）")
    print(f"{'='*65}")
    print(f"  当前 b2 inproc (n_sess=3.2, K=10): ~28.5ms")
    print(f"  transformers batch warm p50 (n_sess=1): {warm_p50:.1f}ms")
    print(f"  预期 n_sess=3.2: {warm_p50*3.2:.1f}ms")
    print()
    print("  p50 < 5ms  → 强烈推荐迁移（>3× 提升）")
    print("  p50 < 10ms → 有改善，可以迁移（~2× 提升）")
    print("  p50 > 20ms → transformers 路径无优势，维持现状")


if __name__ == "__main__":
    main()

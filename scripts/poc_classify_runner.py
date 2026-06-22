#!/usr/bin/env python3
"""
PoC: vllm task='token_classify' 作为 b2 inproc VM 评分路径的替代方案

当前 b2 inproc 路径：
    LLM(hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]})
    + llm.generate(max_tokens=1) + inproc reward buffer 读写

本 PoC 测试的替代路径：
    LLM(task="token_classify")
    + llm.encode() → scores 直接返回（无 buffer hack）

VM-Qwen3-4B-merged-for-vllm 是 Qwen3ForSequenceClassification (num_labels=1)，
HTTP 后端已经用 --task token_classify 正常运行。本 PoC 验证能否用于 inproc。

验证目标：
    1. task='token_classify' 能否在 inproc 模式下正常加载
    2. APC 是否对 classify 路径有效（cold vs warm 延迟对比）
    3. 不同 batch size 的延迟（对比当前 b2 inproc eager 的 ~28.5ms）
    4. [--piecewise] piecewise CUDA graph 能否避免 batch_descriptor RuntimeError
       （这是最关键的问题：classify 路径的 batch_descriptor 是否更稳定？）

用法：
    # 基础测试（eager 模式）
    python scripts/poc_classify_runner.py \\
        --vm_model /path/to/VM-Qwen3-4B-merged-for-vllm \\
        --gpu_mem 0.20

    # 同时测试 piecewise CUDA graph
    python scripts/poc_classify_runner.py \\
        --vm_model /path/to/VM-Qwen3-4B-merged-for-vllm \\
        --gpu_mem 0.20 --piecewise
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="PoC: vllm classify runner for VM-Qwen3-4B inproc scoring"
    )
    p.add_argument("--vm_model", required=True,
                   help="VM-Qwen3-4B-merged-for-vllm 的路径")
    p.add_argument("--gpu_mem", type=float, default=0.20,
                   help="GPU 显存占比（default: 0.20）")
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="最大序列长度（default: 4096）")
    p.add_argument("--prefix_len", type=int, default=480,
                   help="模拟前缀长度（default: 480，近似真实 35B SIA 场景）")
    p.add_argument("--topk", type=int, default=10,
                   help="每个 session 的候选数（default: 10）")
    p.add_argument("--n_warmup", type=int, default=5,
                   help="延迟测量前的预热次数（default: 5）")
    p.add_argument("--n_trials", type=int, default=30,
                   help="延迟测量次数（default: 30）")
    p.add_argument("--piecewise", action="store_true",
                   help="额外测试 piecewise CUDA graph（关键问题：能否避免 batch_descriptor 错误？）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _make_session_prompts(prefix_len, topk, seed=42):
    """生成一个 session 的 K 个候选 prompt（prefix + [candidate]）。"""
    from vllm import TokensPrompt
    random.seed(seed)
    vocab_size = 32000  # 近似值，用于随机 token 生成
    prefix = [random.randint(1, vocab_size - 1) for _ in range(prefix_len)]
    return [
        TokensPrompt(prompt_token_ids=prefix + [random.randint(1, vocab_size - 1)])
        for _ in range(topk)
    ]


def _make_batch_prompts(prefix_len, topk, n_sessions):
    """生成 M sessions × K 候选的完整 batch。各 session 使用不同前缀。"""
    from vllm import TokensPrompt
    all_prompts = []
    for s in range(n_sessions):
        random.seed(s * 1000)
        vocab_size = 32000
        prefix = [random.randint(1, vocab_size - 1) for _ in range(prefix_len)]
        for _ in range(topk):
            cand = [random.randint(1, vocab_size - 1)]
            all_prompts.append(TokensPrompt(prompt_token_ids=prefix + cand))
    return all_prompts


def _extract_scores(outputs):
    """从 encode() 输出中提取标量分数列表。"""
    scores = []
    for out in outputs:
        data = out.outputs.data
        # num_labels=1: data 是长度为 1 的 list 或 0 维 tensor
        if hasattr(data, '__len__'):
            scores.append(float(data[0]))
        else:
            scores.append(float(data))
    return scores


# ---------------------------------------------------------------------------
# 加载模型
# ---------------------------------------------------------------------------

def load_classify_llm(vm_model, gpu_mem, max_model_len, enforce_eager=True):
    """以 runner='pooling' + convert='classify' 加载 VM（inproc 模式）。

    vllm 0.18.0 用 runner/convert 而非 task 参数（task= 是更新版本加的）。
    对应 CLI: vllm serve ... --runner pooling --convert classify
    """
    # inproc 模式：关闭多进程，所有计算在当前进程内
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM

    kwargs = dict(
        model=vm_model,
        runner="pooling",
        convert="classify",
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )
    if not enforce_eager:
        # piecewise 模式：不强制 eager，让 vllm 走默认 PIECEWISE CUDA graph
        kwargs.pop("enforce_eager")  # 使用 vllm 默认（FULL_AND_PIECEWISE）

    return LLM(**kwargs)


# ---------------------------------------------------------------------------
# 测试 1：基础加载 + 冒烟测试
# ---------------------------------------------------------------------------

def test_smoke(llm, topk, mode_tag):
    """验证 classify runner 能正常评分（不报错，分数合理）。"""
    from vllm import TokensPrompt


    # 3 条不同的短 prompt
    test_prompts = [
        TokensPrompt(prompt_token_ids=[1, 2, 3, 4, 5]),
        TokensPrompt(prompt_token_ids=[1, 2, 3, 4, 6]),     # 只有最后 1 token 不同
        TokensPrompt(prompt_token_ids=[100, 200, 300, 400, 500]),
    ]

    try:
        out1 = llm.encode(test_prompts, pooling_task="token_classify", use_tqdm=False)
        scores1 = _extract_scores(out1)
        print(f"  [{mode_tag}] 冒烟测试分数: {[f'{s:.4f}' for s in scores1]}")
    except Exception as e:
        print(f"  [{mode_tag}] ❌ 冒烟测试失败: {e}")
        return False

    # 检查分数是数字（非 NaN/Inf）
    for i, s in enumerate(scores1):
        if not (s == s) or abs(s) > 1e6:   # NaN check + sanity range
            print(f"  [{mode_tag}] ⚠️  prompt[{i}] score={s} 异常")
            return False

    # 确定性检查：同一 prompt 跑两次结果相同
    out2 = llm.encode(test_prompts[:1], pooling_task="token_classify", use_tqdm=False)
    scores2 = _extract_scores(out2)
    diff = abs(scores1[0] - scores2[0])
    if diff > 1e-3:
        print(f"  [{mode_tag}] ⚠️  非确定性: 同 prompt 两次得分差 {diff:.6f}")
    else:
        print(f"  [{mode_tag}] ✅ 确定性 OK（同 prompt 差值 {diff:.2e}）")

    # 相邻候选分数不应完全相同（验证 score head 有区分度）
    if abs(scores1[0] - scores1[1]) < 1e-6:
        print(f"  [{mode_tag}] ⚠️  prompt[0] 和 [1] 分数完全相同，可能 score head 未生效")
    else:
        print(f"  [{mode_tag}] ✅ 候选间有区分度（diff={abs(scores1[0]-scores1[1]):.4f}）")

    return True


# ---------------------------------------------------------------------------
# 测试 2：APC 有效性（cold vs warm 对比）
# ---------------------------------------------------------------------------

def test_apc(llm, prefix_len, topk):
    """通过 cold/warm 延迟对比验证 APC 是否对 classify 路径有效。"""

    # 同一批 prompt（APC warm = 前缀已缓存）
    warm_prompts = _make_session_prompts(prefix_len, topk, seed=7777)

    # 不同前缀的 prompt（APC cold = 每次 miss）
    def make_cold_prompts(trial_seed):
        return _make_session_prompts(prefix_len, topk, seed=trial_seed * 9999)

    # Warm：先跑一次把前缀推入 APC
    llm.encode(warm_prompts, pooling_task="token_classify", use_tqdm=False)

    # Cold 测量（每次新前缀）
    cold_times = []
    for i in range(10):
        prompts = make_cold_prompts(i)
        t0 = time.perf_counter()
        llm.encode(prompts, pooling_task="token_classify", use_tqdm=False)
        cold_times.append((time.perf_counter() - t0) * 1000)

    # Warm 测量（同一前缀，APC 命中）
    warm_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        llm.encode(warm_prompts, pooling_task="token_classify", use_tqdm=False)
        warm_times.append((time.perf_counter() - t0) * 1000)

    cold_p50 = float(np.percentile(cold_times, 50))
    warm_p50 = float(np.percentile(warm_times, 50))
    ratio = cold_p50 / warm_p50 if warm_p50 > 0 else 0

    print(f"  APC cold (新前缀)  p50 = {cold_p50:6.1f}ms")
    print(f"  APC warm (已缓存)  p50 = {warm_p50:6.1f}ms")
    print(f"  Speedup: {ratio:.2f}×", end="")
    if ratio >= 1.5:
        print("  ✅ APC 生效")
    elif ratio >= 1.1:
        print("  ⚠️  APC 有一定效果但不明显")
    else:
        print("  ❌ APC 无效（classify 路径可能未走 prefix caching）")


# ---------------------------------------------------------------------------
# 测试 3：延迟 benchmark（模拟 conc=16 的 batch scoring）
# ---------------------------------------------------------------------------

def benchmark_latency(llm, prefix_len, topk, n_warmup, n_trials, mode_tag):
    """测量不同 session 数下的 classify 延迟，与已知 b2 inproc 数字对比。"""

    print(f"\n  [{mode_tag}] 延迟 benchmark (prefix_len={prefix_len}, topk={topk})")
    print(f"  {'n_sess':>6}  {'batch':>5}  {'p50':>8}  {'p90':>8}  {'amort':>8}  {'vs b2 eager':>12}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")

    # b2 inproc eager 参考值（来自实测，batch call 总时间）
    b2_eager_ref = {1: 9.0, 2: 14.0, 3: 18.0, 4: 21.0, 8: 28.5, 16: 28.5}

    for n_sess in [1, 2, 3, 4, 8]:
        prompts = _make_batch_prompts(prefix_len, topk, n_sess)

        # 预热（第一次 cold prefill，之后 APC warm）
        for _ in range(n_warmup):
            llm.encode(prompts, pooling_task="token_classify", use_tqdm=False)

        # 计时（APC warm 状态，同前缀反复调用）
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            llm.encode(prompts, pooling_task="token_classify", use_tqdm=False)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = float(np.percentile(times, 50))
        p90 = float(np.percentile(times, 90))
        amort = p50 / n_sess

        ref = b2_eager_ref.get(n_sess, None)
        ref_str = f"{p50/ref:.2f}×" if ref else "—"

        print(f"  {n_sess:>6}  {n_sess*topk:>5}  "
              f"{p50:>7.1f}ms  {p90:>7.1f}ms  "
              f"{amort:>7.1f}ms  {ref_str:>12}")


# ---------------------------------------------------------------------------
# 测试 4：piecewise CUDA graph 压力测试
# ---------------------------------------------------------------------------

def test_piecewise_stress(llm, prefix_len, topk, n_stress=100):
    """
    用不同长度的前缀反复调用，看 piecewise CUDA graph 是否会触发 batch_descriptor
    RuntimeError（与 generate 路径的已知问题对比）。

    如果 0 错误 → classify 路径的 piecewise CUDA graph 可用 → 大幅加速机会！
    """
    from vllm import TokensPrompt

    print(f"\n  Piecewise 压力测试：{n_stress} 次变长前缀调用")
    vocab_size = 32000
    errors = []
    times = []

    for trial in range(n_stress):
        # 每次使用不同长度的前缀（模拟不同 session、不同生成步数）
        cur_prefix_len = random.randint(10, min(prefix_len, 500))
        random.seed(trial)
        prefix = [random.randint(1, vocab_size - 1) for _ in range(cur_prefix_len)]
        prompts = [
            TokensPrompt(prompt_token_ids=prefix + [random.randint(1, vocab_size - 1)])
            for _ in range(topk)
        ]

        try:
            t0 = time.perf_counter()
            llm.encode(prompts, pooling_task="token_classify", use_tqdm=False)
            times.append((time.perf_counter() - t0) * 1000)
        except RuntimeError as e:
            if "CUDA graph capturing" in str(e) or "batch_descriptor" in str(e).lower():
                errors.append((trial, str(e)[:120]))
            else:
                errors.append((trial, f"other RuntimeError: {str(e)[:120]}"))
        except Exception as e:
            errors.append((trial, f"{type(e).__name__}: {str(e)[:80]}"))

    if not errors:
        p50 = float(np.percentile(times, 50)) if times else 0
        p90 = float(np.percentile(times, 90)) if times else 0
        print(f"  ✅ 0 / {n_stress} 错误 — piecewise CUDA graph 对 classify 路径可用！")
        print(f"  延迟 p50={p50:.1f}ms  p90={p90:.1f}ms")
        print(f"  → 若 b2 inproc 改用 task='token_classify' + piecewise，")
        print(f"    可期望 VM batch call ~28ms → ~11-15ms（类似 VL-30B 效果）")
        return True
    else:
        print(f"  ❌ {len(errors)} / {n_stress} 次调用报错")
        for trial, msg in errors[:5]:
            print(f"     trial {trial}: {msg}")
        if len(errors) > 5:
            print(f"     ... 共 {len(errors)} 个错误")
        return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 65)
    print("PoC: vllm task='token_classify' for b2 inproc VM scoring")
    print("=" * 65)
    print(f"  vm_model:      {args.vm_model}")
    print(f"  gpu_mem:       {args.gpu_mem}")
    print(f"  max_model_len: {args.max_model_len}")
    print(f"  prefix_len:    {args.prefix_len}  topk: {args.topk}")

    if not os.path.isdir(args.vm_model):
        print(f"\n❌ vm_model 路径不存在: {args.vm_model}")
        sys.exit(1)

    # ---- 1. Eager 模式测试 ----
    print(f"\n{'='*65}")
    print("阶段 1：task='token_classify' + eager 模式")
    print(f"{'='*65}")

    t_load = time.perf_counter()
    try:
        llm_eager = load_classify_llm(
            args.vm_model, args.gpu_mem, args.max_model_len, enforce_eager=True
        )
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("  可能原因：")
        print("  - vllm 0.18.0 需要 runner='pooling' + convert='classify'（已修正）")
        print("  - 模型不是 Qwen3ForSequenceClassification 格式")
        print("  - 显存不足（先确认 nvidia-smi 显存已释放）")
        sys.exit(1)

    print(f"✅ 加载成功 ({time.perf_counter()-t_load:.1f}s)")

    print("\n[1] 冒烟测试")
    ok = test_smoke(llm_eager, args.topk, "eager")
    if not ok:
        print("❌ 冒烟测试失败，退出")
        sys.exit(1)

    print("\n[2] APC 有效性测试")
    test_apc(llm_eager, args.prefix_len, args.topk)

    print("\n[3] 延迟 benchmark")
    benchmark_latency(
        llm_eager, args.prefix_len, args.topk,
        args.n_warmup, args.n_trials, "eager"
    )

    # ---- 2. Piecewise 模式测试（可选，最关键）----
    if args.piecewise:
        print(f"\n{'='*65}")
        print("阶段 2：task='token_classify' + piecewise CUDA graph")
        print("  关键问题：classify 路径能否避免 batch_descriptor RuntimeError？")
        print(f"{'='*65}")

        # 释放 eager 模型显存
        del llm_eager
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print("\n加载 piecewise 模式...")
        t_load = time.perf_counter()
        try:
            llm_pw = load_classify_llm(
                args.vm_model, args.gpu_mem, args.max_model_len, enforce_eager=False
            )
        except Exception as e:
            print(f"❌ piecewise 加载失败: {e}")
            sys.exit(0)

        print(f"✅ 加载成功 ({time.perf_counter()-t_load:.1f}s)")

        print("\n[4] piecewise 压力测试")
        piecewise_ok = test_piecewise_stress(llm_pw, args.prefix_len, args.topk)

        if piecewise_ok:
            print("\n[5] piecewise 延迟 benchmark（压力测试 0 错误，继续）")
            benchmark_latency(
                llm_pw, args.prefix_len, args.topk,
                args.n_warmup, args.n_trials, "piecewise"
            )
        else:
            print("\n  跳过 piecewise latency benchmark（压力测试有错误）")

    print(f"\n{'='*65}")
    print("PoC 完成")
    print(f"{'='*65}")
    print()
    print("结论参考：")
    print("  - eager classify p50 < 25ms（vs 当前 b2 eager ~28ms）→ 小幅改善，可替换")
    print("  - piecewise 0 错误 + p50 < 15ms           → 大幅改善，强烈推荐迁移")
    print("  - piecewise 有 batch_descriptor 错误        → classify 路径与 generate 路径同样受限")


if __name__ == "__main__":
    main()

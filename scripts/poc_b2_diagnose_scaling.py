"""
诊断 M1b 的 7× scaling 问题 — 同一 prompt 分别从 vLLM 和 HF 取 hidden_state, 对比数值。

策略:
  1. 用 1 个 prompt 跑 vLLM patched 路径，hook 写出 hidden_state tensor 到 /tmp/b2_dump_vllm.pt
  2. 用同 prompt 跑 HF Qwen3Model.forward (没有 score head, 纯 backbone)，
     dump last token 的 hidden_state 到 /tmp/b2_dump_hf.pt
  3. 加载两份 dump，对比 norm/mean/std/前几个元素，以及 score(hidden) 的差
  4. 同时对比 input_ids 末尾 几个 token，确认采样位置一致

新增 patch (在 compute_logits) 写一份 hidden_state tensor 到 dump 文件。
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

DUMP_VLLM = "/tmp/b2_dump_vllm.pt"
DUMP_HF = "/tmp/b2_dump_hf.pt"
DUMP_IDS = "/tmp/b2_dump_ids.pt"

# 重要：vLLM 也必须用 VM 模型 (merged LoRA + score head),
# 否则 hidden_state 跟 HF 的 VM 不在同一个 backbone 上
MODEL_VLLM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"
MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"


def build_prompt(tok):
    convs = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": " 4"},
    ]
    text = tok.apply_chat_template(convs, tokenize=False)
    bos = tok.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def run_vllm(prompt):
    if os.path.exists(DUMP_VLLM):
        os.remove(DUMP_VLLM)
    if os.path.exists(DUMP_IDS):
        os.remove(DUMP_IDS)

    from vllm import LLM, SamplingParams
    print(f"[vLLM] Loading {MODEL_VLLM} ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_VLLM,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=False,
        disable_log_stats=True,
    )
    print(f"[vLLM] Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        ignore_eos=True)
    out = llm.generate([prompt], sp, use_tqdm=False)
    print(f"[vLLM] generated first token id = {out[0].outputs[0].token_ids}",
          flush=True)
    # 关键：dump vLLM 实际看到的 prompt_token_ids
    vllm_prompt_ids = list(out[0].prompt_token_ids)
    print(f"[vLLM] prompt_token_ids len={len(vllm_prompt_ids)} "
          f"last 10 = {vllm_prompt_ids[-10:]}", flush=True)
    torch.save(torch.tensor(vllm_prompt_ids), "/tmp/b2_dump_vllm_ids.pt")
    del llm
    torch.cuda.empty_cache()


def run_hf(prompt):
    print(f"[HF] Loading backbone (AutoModel) ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(
        MODEL_VM, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    print(f"[HF] backbone loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"[HF] input_ids shape = {ids.shape}, last 5 ids = {ids[0, -5:].tolist()}",
          flush=True)
    torch.save(ids.cpu(), DUMP_IDS)

    with torch.no_grad():
        out = backbone(ids)
    # last_hidden_state shape (1, seq_len, hidden)
    last_hidden = out.last_hidden_state[0, -1]  # (hidden,) — last token
    torch.save(last_hidden.cpu(), DUMP_HF)
    # 同时 dump 全序列 (1, 24, 2560) 用来找 vLLM 对应位置
    torch.save(out.last_hidden_state[0].cpu(), "/tmp/b2_dump_hf_full.pt")
    print(f"[HF] saved last-token hidden_state ({tuple(last_hidden.shape)}, "
          f"dtype={last_hidden.dtype})", flush=True)
    print(f"[HF] mean={last_hidden.float().mean():.4f} "
          f"std={last_hidden.float().std():.4f} "
          f"norm={last_hidden.float().norm():.4f}", flush=True)

    # 同时跑 SequenceClassification + score head 验证 reward
    del backbone
    torch.cuda.empty_cache()
    cls = AutoModelForSequenceClassification.from_pretrained(
        MODEL_VM, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    with torch.no_grad():
        cls_out = cls(ids)
    print(f"[HF] SequenceClassification reward = {cls_out.logits.float().item():.4f}",
          flush=True)
    # 拿 score head weight 出来
    sw = cls.score.weight.detach().cpu()
    torch.save(sw, "/tmp/b2_dump_score_w.pt")
    print(f"[HF] score head weight shape = {tuple(sw.shape)} "
          f"dtype={sw.dtype} norm={sw.float().norm():.4f}", flush=True)
    del cls
    torch.cuda.empty_cache()


def compare():
    print("\n" + "=" * 60)
    print("对比 vLLM dump vs HF dump")
    print("=" * 60)

    if not os.path.exists(DUMP_VLLM):
        print(f"❌ vLLM dump 不存在 {DUMP_VLLM} — patch 未生效")
        return
    # 先对比 input_ids
    if os.path.exists("/tmp/b2_dump_vllm_ids.pt") and os.path.exists(DUMP_IDS):
        vllm_ids = torch.load("/tmp/b2_dump_vllm_ids.pt").tolist()
        hf_ids = torch.load(DUMP_IDS).squeeze(0).tolist()
        print(f"vLLM ids len={len(vllm_ids)} HF ids len={len(hf_ids)}")
        print(f"  vLLM last 10: {vllm_ids[-10:]}")
        print(f"  HF   last 10: {hf_ids[-10:]}")
        if vllm_ids == hf_ids:
            print(f"  ✅ token ids 完全相同")
        else:
            print(f"  ❌ token ids 不同！这就是 hidden_state 差异的根因")
    v = torch.load(DUMP_VLLM)
    h = torch.load(DUMP_HF)
    print(f"vLLM  : shape={tuple(v.shape)} dtype={v.dtype} "
          f"mean={v.float().mean():.4f} std={v.float().std():.4f} "
          f"norm={v.float().norm():.4f}")
    print(f"HF    : shape={tuple(h.shape)} dtype={h.dtype} "
          f"mean={h.float().mean():.4f} std={h.float().std():.4f} "
          f"norm={h.float().norm():.4f}")

    # 维度对齐: vLLM 可能 (n_samples, hidden), HF 是 (hidden,)
    if v.dim() == 2:
        v_last = v[-1]
    else:
        v_last = v
    print(f"\n前 8 元素:")
    print(f"  vLLM: {v_last.float()[:8].tolist()}")
    print(f"  HF  : {h.float()[:8].tolist()}")

    # cos 相似度
    cos = torch.nn.functional.cosine_similarity(
        v_last.float().unsqueeze(0), h.float().unsqueeze(0)
    ).item()
    print(f"\ncos similarity (vLLM_last, HF_last) = {cos:.6f}")

    # 找 vLLM_last 对应 HF 序列中哪个位置 cos 最大
    if os.path.exists("/tmp/b2_dump_hf_full.pt"):
        hf_full = torch.load("/tmp/b2_dump_hf_full.pt").float()  # (seq, hidden)
        v_norm = v_last.float() / v_last.float().norm()
        h_norm = hf_full / hf_full.norm(dim=-1, keepdim=True)
        sims = (h_norm @ v_norm).tolist()
        print(f"\nvLLM_last vs HF[i] for each position i:")
        for i, s in enumerate(sims):
            mark = " ←最高" if s == max(sims) else ""
            print(f"  pos {i:2d}: cos={s:+.4f}{mark}")
    diff_norm = (v_last.float() - h.float()).norm().item()
    print(f"||vLLM - HF|| = {diff_norm:.4f}")
    print(f"||vLLM|| / ||HF|| = {v_last.float().norm() / h.float().norm():.4f}")

    # 用 HF score head 各自打 score
    if os.path.exists("/tmp/b2_dump_score_w.pt"):
        sw = torch.load("/tmp/b2_dump_score_w.pt")
        sv = (v_last.float() @ sw.float().T).item()
        sh = (h.float() @ sw.float().T).item()
        print(f"\nscore(vLLM_hidden) = {sv:.4f}")
        print(f"score(HF_hidden)   = {sh:.4f}")
        print(f"score ratio HF/vLLM = {sh / sv if abs(sv) > 1e-6 else 'inf':.4f}")


def main():
    print("=" * 60)
    print("M1b 诊断: vLLM 与 HF 路径 hidden_state 差异")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)
    prompt = build_prompt(tok)
    print(f"\nPrompt ({len(prompt)} chars):")
    print(prompt)
    print(f"\ntokenized last 10 ids: "
          f"{tok(prompt).input_ids[-10:]}")

    # 1) vLLM 路径
    print(f"\n[1/3] 跑 vLLM (会触发 patched compute_logits + dump)")
    run_vllm(prompt)

    # 2) HF backbone + score head
    print(f"\n[2/3] 跑 HF backbone + SequenceClassification")
    run_hf(prompt)

    # 3) compare
    print(f"\n[3/3] 对比")
    compare()


if __name__ == "__main__":
    main()

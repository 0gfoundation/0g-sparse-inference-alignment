"""
将 VM-Qwen3-4B-merged-for-vllm 量化为 FP8_DYNAMIC，保留 score head 为 BF16。

ignore list 基于 Step 1.0 实测出的真实模块名：
  classifier head:  score              -> Linear(2560, 1)
  embedding:        model.embed_tokens -> Embedding(151936, 2560)

用法:
    python scripts/quantize_rm_fp8_dynamic.py
        [--src /path/to/source] [--dst /path/to/output] [--n_samples 50]

详见 doc/rm-fp8-dynamic-quantization-plan.md。
"""
import argparse
import random
from datasets import Dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_calibration_dataset(tok, n_samples: int = 50, max_len: int = 1024):
    """构造 calibration 样本。FP8_DYNAMIC 不依赖分布，但 oneshot 仍需要样本走模型图。

    样本格式必须与部署一致：chat-formatted user+assistant turn。
    覆盖多种主题（技术 / 科学 / 人文 / 数学 / 闲聊 / 多语言 / 长短文），
    以更好覆盖 Linear 层的所有激活路径。
    """
    sample_pairs = [
        # 数学 / 算术
        ("What is 2+2?", "The answer is 4."),
        ("Calculate 15% of 200.", "30."),
        ("Solve x^2 = 25.", "x = 5 or x = -5."),
        # 科学 / 自然
        ("Explain photosynthesis briefly.",
         "Plants convert sunlight to energy via chlorophyll."),
        ("What is the boiling point of water?",
         "100 degrees Celsius at sea level."),
        ("Why is the sky blue?",
         "Rayleigh scattering of sunlight by air molecules."),
        ("Name 3 noble gases.", "Helium, neon, argon."),
        # 人文 / 历史 / 文学
        ("Who wrote Hamlet?", "William Shakespeare."),
        ("When did World War II end?", "September 2, 1945."),
        ("Capital of France?", "Paris."),
        ("Translate 'hello' to Spanish.", "Hola."),
        # 编程 / 技术
        ("What is recursion in programming?",
         "A function calling itself with smaller inputs."),
        ("Difference between TCP and UDP?",
         "TCP is connection-oriented and reliable; UDP is connectionless."),
        ("What does HTTP stand for?", "HyperText Transfer Protocol."),
        # 日常 / 列表
        ("List 3 colors.", "Red, blue, green."),
        ("Suggest a quick breakfast.", "Toast with peanut butter and a banana."),
        ("Name 2 musical instruments.", "Piano and guitar."),
        # 长一些的指令
        ("Write a short poem about autumn.",
         "Leaves of amber dance in fading light, "
         "the crisp air whispers summer's end."),
        ("Explain the theory of relativity in one sentence.",
         "Space and time are interwoven, and both bend under gravity and motion."),
        ("Summarize the plot of Romeo and Juliet.",
         "Two young lovers from feuding families secretly marry, "
         "then die by misunderstanding."),
        # 推理 / 多步
        ("If a train travels 60 mph for 2 hours, how far did it go?",
         "120 miles."),
        ("Is 17 prime?", "Yes, 17 has only 1 and 17 as divisors."),
        # 否定 / 偏见检查
        ("What's the capital of the moon?",
         "The moon has no capital city; it's not a country."),
        # 中英混合 / 多语言
        ("北京的首都是哪？", "北京就是中国的首都。"),
        ("用英文说'谢谢'。", "Thank you."),
        # 代码
        ("Python print 'hello world'.", "print('hello world')"),
        ("What does this Python do: x = [1,2,3]; print(sum(x))?",
         "It prints 6, the sum of the list."),
        # 哲学 / 开放
        ("Is free will real?",
         "Philosophers debate this; both compatibilist and "
         "libertarian views exist."),
        ("Define consciousness.",
         "Subjective awareness of one's surroundings and inner experience."),
        # 长输入
        ("List 10 fruits.",
         "Apple, banana, orange, grape, mango, strawberry, watermelon, "
         "pineapple, kiwi, peach."),
    ]
    random.seed(42)
    random.shuffle(sample_pairs)
    samples = (sample_pairs * (n_samples // len(sample_pairs) + 1))[:n_samples]

    texts = []
    for user, assistant in samples:
        convs = [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tok.apply_chat_template(convs, tokenize=False)
        bos = tok.bos_token
        if bos and text.startswith(bos):
            text = text[len(bos):]
        texts.append(text)

    encodings = tok(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
        return_attention_mask=True,
    )
    return Dataset.from_dict({
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm")
    p.add_argument("--dst", default="/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic")
    p.add_argument("--n_samples", type=int, default=50)
    args = p.parse_args()

    print(f"[quantize] Loading model from {args.src}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.src,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    print(f"[quantize] Building calibration dataset ({args.n_samples} samples)",
          flush=True)
    ds = build_calibration_dataset(tok, n_samples=args.n_samples)

    # Step 1.0 实测确认的真实模块名
    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme="FP8_DYNAMIC",
        ignore=["score", "model.embed_tokens", "lm_head"],
    )

    print(f"[quantize] Running oneshot → {args.dst}", flush=True)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=args.dst,
    )
    tok.save_pretrained(args.dst)
    print("[quantize] Done", flush=True)


if __name__ == "__main__":
    main()

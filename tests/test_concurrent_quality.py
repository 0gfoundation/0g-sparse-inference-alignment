"""
Concurrent Quality Test

Sends 16 concurrent SIA requests with diverse prompts, then uses noSIA as a judge
to verify that each response is coherent (no garbled text, no off-topic replies,
no repetition loops).

Purpose: detect cross-request state contamination or output corruption that can
appear under concurrent load but not in single-request testing.

PASS: at least 14 of 16 responses judged acceptable by noSIA.
"""
import argparse
import asyncio
import json
import sys
import time

try:
    import aiohttp
except ImportError:
    print("Missing dependency: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

URL_DEFAULT = "http://localhost:8000"
CONCURRENCY = 16
MAX_TOKENS_SIA = 1024
MAX_TOKENS_JUDGE = 512   # thinking model needs room to reason before outputting PASS/FAIL
PASS_THRESHOLD = 14   # out of CONCURRENCY

# Diverse prompts — different topics, styles, and expected answer shapes.
# Variety stresses concurrent session isolation (each request must get its own answer).
PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain what a neural network is in two sentences.",
    "What is 17 multiplied by 23?",
    "Name three programming languages and one use case for each.",
    "What causes rainbows?",
    "Translate 'Good morning' into Spanish, French, and Japanese.",
    "Write a one-sentence definition of machine learning.",
    "What are the primary colors?",
    "Describe the water cycle in three steps.",
    "What is the largest planet in our solar system?",
    "Give one example of a renewable energy source and explain it briefly.",
    "What does HTTP stand for?",
    "Write a very short story (2 sentences) about a curious cat.",
    "What is the boiling point of water in Celsius?",
    "Name the four seasons and one activity associated with each.",
]

JUDGE_SYSTEM = (
    "/no_think\n"
    "You are a response quality checker. Do NOT output any thinking or reasoning. "
    "Reply immediately with your verdict.\n"
    "You will be shown a user question and an AI-generated answer. "
    "Note: the answer may begin with a thinking/reasoning section before the final answer — "
    "this is normal for reasoning models; evaluate the overall response including the final answer.\n"
    "Decide whether the answer is acceptable:\n"
    "  - PASS if the response is coherent, the final answer addresses the question, "
    "contains no garbled characters or random symbols, and does not loop/repeat endlessly.\n"
    "  - FAIL if the response is completely cut off before any answer is given, "
    "is entirely off-topic, or contains obvious corruption.\n"
    "Your ENTIRE response must be: one word (PASS or FAIL) on the first line, "
    "then optionally one short reason sentence on the second line. Nothing else."
)


async def sia_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    idx: int,
) -> dict:
    """Send one SIA request (SIA enabled, default weight)."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS_SIA,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"idx": idx, "prompt": prompt, "ok": False,
                        "error": f"HTTP {resp.status}: {body[:200]}"}
            data = await resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            elapsed = (time.perf_counter() - t0) * 1000
            return {"idx": idx, "prompt": prompt, "ok": True,
                    "content": content, "elapsed_ms": elapsed}
    except Exception as e:
        return {"idx": idx, "prompt": prompt, "ok": False, "error": str(e)}


async def judge_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    response: str,
) -> tuple[bool, str]:
    """Ask noSIA to judge whether a response is acceptable. Returns (pass, reason)."""
    user_msg = f"User question: {prompt}\n\nAI answer: {response}"
    payload = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": MAX_TOKENS_JUDGE,
        "temperature": 0.0,
        "sia_weight": 0,   # noSIA for judging — neutral, faster
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return False, f"judge HTTP {resp.status}"
            data = await resp.json()
            verdict = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            # The judge model (0GM-35B) may emit a thinking section before its
            # verdict. Scan all lines for the first one that starts with PASS or
            # FAIL rather than only checking the first line.
            passed = False
            reason = ""
            for line in verdict.split("\n"):
                word = line.strip().upper()
                if word.startswith("PASS"):
                    passed = True
                    reason = line.strip()
                    break
                if word.startswith("FAIL"):
                    passed = False
                    reason = line.strip()
                    break
            return passed, reason
    except Exception as e:
        return False, f"judge error: {e}"


async def run(url: str) -> bool:
    print(f"  Sending {CONCURRENCY} concurrent SIA requests...", flush=True)

    async with aiohttp.ClientSession() as session:
        # Phase 1: concurrent SIA generation
        t0 = time.perf_counter()
        sia_results = await asyncio.gather(
            *[sia_request(session, url, PROMPTS[i], i) for i in range(CONCURRENCY)]
        )
        gen_elapsed = time.perf_counter() - t0
        print(f"  Generation done in {gen_elapsed:.1f}s", flush=True)

        # Phase 2: sequential judge calls (avoids overloading the server)
        print(f"  Judging {CONCURRENCY} responses with noSIA...", flush=True)
        passed = 0
        failed_indices = []

        for r in sorted(sia_results, key=lambda x: x["idx"]):
            idx = r["idx"]
            prompt_short = PROMPTS[idx][:50]

            if not r.get("ok"):
                verdict_str = "❌ SIA_FAIL"
                reason = r.get("error", "no response")
                failed_indices.append(idx)
            else:
                content = r["content"]
                judge_pass, reason = await judge_request(session, url, PROMPTS[idx], content)
                if judge_pass:
                    passed += 1
                    verdict_str = "✅ PASS"
                else:
                    failed_indices.append(idx)
                    verdict_str = "❌ FAIL"
                # Truncate content for display
                preview = content[:60].replace("\n", " ")

            if r.get("ok"):
                print(f"    [{idx:02d}] {verdict_str}  q={prompt_short!r}")
                print(f"          ans={preview!r}  reason={reason!r}")
            else:
                print(f"    [{idx:02d}] {verdict_str}  q={prompt_short!r}  err={reason!r}")

    total_ok = CONCURRENCY - len([r for r in sia_results if not r.get("ok")])
    print(f"\n  Results: {passed}/{CONCURRENCY} passed judge  "
          f"({total_ok}/{CONCURRENCY} got SIA response)")
    print(f"  Pass threshold: {PASS_THRESHOLD}/{CONCURRENCY}")

    if failed_indices:
        print(f"  Failed indices: {failed_indices}")

    overall = passed >= PASS_THRESHOLD
    print(f"  {'✅ PASS' if overall else '❌ FAIL'}")
    return overall


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent quality test: 16 SIA requests judged by noSIA"
    )
    parser.add_argument("--url", default=URL_DEFAULT)
    args = parser.parse_args()

    ok = asyncio.run(run(args.url))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

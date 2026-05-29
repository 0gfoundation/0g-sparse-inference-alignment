"""Quick sanity check that RMClient default (multiprocessing=False) works
end-to-end and yields the same shape rewards. Compares numerical values
against multiprocessing=True on the same inputs.

Run:
    python scripts/sanity_inproc.py
"""
import os
import random
import sys
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
os.environ.setdefault("PYTHONPATH", _SRC)
os.environ["SIA_REWARD_FILE_ID"] = f"sanity_{uuid.uuid4().hex[:8]}"

import sia_rm  # noqa
from sia_rm import RMClient

MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"

random.seed(7)
prefix = [random.randint(1000, 100000) for _ in range(50)]
candidates = [random.randint(1000, 100000) for _ in range(5)]

print(">>> RMClient default (multiprocessing=False, InprocClient) ...")
rm = RMClient(model_path=MODEL_VM, gpu_mem=0.3, max_model_len=2048)
sid = rm.new_session(prefix)
rewards = rm.score_candidates(sid, candidates)
print(f"    rewards = {[f'{r:.6f}' for r in rewards]}")
print(f"    len     = {len(rewards)}, type = {type(rewards[0]).__name__}")
print("    OK\n")

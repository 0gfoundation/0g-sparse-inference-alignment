"""
vLLM general plugin entry point.

vLLM calls this when any of its processes (LLM main, EngineCore subprocess,
or the model-inspection subprocess) finishes startup and runs
`load_general_plugins()`. We use it to trigger ModelRegistry registration
for our custom architectures so they're visible inside subprocesses.

Without this, the subprocess sees only vLLM's built-in architectures and
fails with "Model architectures ['Qwen3WithScoreForCausalLM'] are not
supported".

Wire-up is in pyproject.toml under [project.entry-points."vllm.general_plugins"].
Installation: `pip install -e .` from repo root.
"""


def register_with_vllm():
    # Import has the side effect of calling ModelRegistry.register_model()
    # at module top level.
    from . import qwen3_with_score  # noqa: F401

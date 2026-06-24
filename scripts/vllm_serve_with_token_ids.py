"""
Patches Pydantic when starting a vLLM RM server, so that ClassificationRequest.input
accepts token IDs including list[int] / list[list[int]], avoiding server-side re-tokenize.

vLLM 0.10.1.1 original schema:
    input: Union[list[str], str]
After patch:
    input: Union[list[int], list[list[int]], str, list[str]]

Downstream _preprocess_completion already supports token_ids (the embedding endpoint uses the same logic);
the only barrier is Pydantic's request model validation. This script only modifies the schema;
runtime behavior is unchanged.

Usage (replace the original `vllm serve ...` command):
    python scripts/vllm_serve_with_token_ids.py serve \\
        /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \\
        --runner pooling --convert classify \\
        --enable-prefix-caching --no-enable-chunked-prefill \\
        --gpu-memory-utilization 0.3 \\
        --max-model-len 2048 --port 8001

On the client side, just use the `--use_token_ids` startup option in src/sia_vllm_server.py.
"""
import sys


def patch_classification_request() -> None:
    """Modify the type annotation of ClassificationRequest.input before the vLLM API endpoint is registered."""
    from typing import Union, List
    from vllm.entrypoints.openai.protocol import ClassificationRequest

    new_type = Union[List[int], List[List[int]], str, List[str]]
    ClassificationRequest.model_fields["input"].annotation = new_type
    ClassificationRequest.model_rebuild(force=True)
    print(
        "[vllm-rm-patch] ClassificationRequest.input now accepts "
        "Union[List[int], List[List[int]], str, List[str]]",
        flush=True,
    )


def main() -> None:
    patch_classification_request()

    # Pass remaining argv through to vLLM's CLI main entry point
    # (`/path/to/venv/bin/vllm` actually dispatches to vllm.entrypoints.cli.main:main)
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv[0] = "vllm"
    sys.exit(vllm_main())


if __name__ == "__main__":
    main()

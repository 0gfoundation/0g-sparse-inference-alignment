"""
启动 vLLM RM server 时打 Pydantic 补丁，让 ClassificationRequest.input
接受 list[int] / list[list[int]] 在内的 token IDs，省服务端 re-tokenize。

vLLM 0.10.1.1 原 schema：
    input: Union[list[str], str]
打补丁后：
    input: Union[list[int], list[list[int]], str, list[str]]

下游 _preprocess_completion 早就支持 token_ids（embedding endpoint 走的是同一逻辑），
唯一阻拦是 Pydantic 的请求模型校验。本脚本只动 schema，运行时行为不变。

用法（替换原本的 `vllm serve ...` 命令）：
    python scripts/vllm_serve_with_token_ids.py serve \\
        /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \\
        --runner pooling --convert classify \\
        --enable-prefix-caching --no-enable-chunked-prefill \\
        --gpu-memory-utilization 0.3 \\
        --max-model-len 2048 --port 8001

客户端配合 src/sia_vllm_server.py 的 `--use_token_ids` 启动选项即可。
"""
import sys


def patch_classification_request() -> None:
    """在 vLLM API endpoint 注册前修改 ClassificationRequest.input 的类型注解。"""
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

    # 把后续 argv 透传给 vLLM 的 CLI 主入口
    # （`/path/to/venv/bin/vllm` 实际 dispatch 到 vllm.entrypoints.cli.main:main）
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv[0] = "vllm"
    sys.exit(vllm_main())


if __name__ == "__main__":
    main()

# Dockerfile for VL-30B SIA b2 inproc (vllm 0.17.1 sweet-spot path).
#
# This image bakes in all heavy dependencies (vllm 0.17.1, torch 2.10.0,
# transformers 4.57.6, etc.) so they're cached across runs. The actual SIA
# source code and model weights are NOT baked in — they're expected to be
# bind-mounted at container runtime (see doc/docker-install-vl30b-20260606.md).
#
# Assumed runtime layout (matches `docker run -v /dstack/persistent/SIA:/workspace`):
#   /workspace/models/                                ← model weights mount
#       Qwen3-VL-30B-A3B-Instruct/
#       VM-Qwen3-4B-merged-for-vllm/
#   /workspace/sia-repo/0g-sparse-inference-alignment/  ← repo mount
#       src/sia_vllm_server.py
#       pyproject.toml
#       ...
#
# Build:
#   docker build -t sia-vl30b:0.17.1 .
#
# Run:
#   docker run -it --rm --gpus all --shm-size=16g --ipc=host \
#     -p 8000:8000 \
#     -v /dstack/persistent/SIA:/workspace \
#     sia-vl30b:0.17.1 bash

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

# ---- System dependencies -----------------------------------------------
# - python3.12-venv : pytorch image's system Python lacks ensurepip support
# - curl            : smoke-test the HTTP endpoint after launch
# - ca-certificates : pip / huggingface_hub use HTTPS
# (git deliberately NOT installed — the repo is bind-mounted, not cloned)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# ---- Create venv with all heavy Python deps ----------------------------
# Isolated from the docker's pre-installed conda torch 2.11 — vllm 0.17.1
# pulls its own torch 2.10.0 inside the venv.
ENV VENV_PATH=/opt/venv-vl30b
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH=${VENV_PATH}/bin:${PATH}

# Copy ONLY the requirements files (not the whole repo); the source code
# itself is bind-mounted at runtime so source edits take effect without
# rebuilding the image.
COPY requirements/ /tmp/requirements/

RUN python3 -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/pip install --upgrade --no-cache-dir pip wheel && \
    ${VENV_PATH}/bin/pip install --no-cache-dir -r /tmp/requirements/vl30b-b2-inproc.txt && \
    rm -rf /tmp/requirements

# ---- Entrypoint: register sia_rm vllm plugin against the mounted repo --
# pyproject.toml's [project.entry-points."vllm.general_plugins"] sia_rm
# entry must be installed (pip install -e .) for vllm subprocess to find
# Qwen3WithScoreForCausalLM. Doing this at runtime lets the user mount any
# version of the repo without rebuilding the image.
COPY scripts/docker_entrypoint_vl30b.sh /usr/local/bin/sia-entrypoint
RUN chmod +x /usr/local/bin/sia-entrypoint

# Default working directory matches the expected bind-mount location.
WORKDIR /workspace/sia-repo/0g-sparse-inference-alignment

ENTRYPOINT ["/usr/local/bin/sia-entrypoint"]
CMD ["bash"]

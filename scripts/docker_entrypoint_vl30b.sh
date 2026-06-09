#!/usr/bin/env bash
# Entrypoint for the sia-vl30b docker image (see Dockerfile at repo root).
#
# Heavy deps (vllm 0.17.1 + torch 2.10.0 + transformers ...) are already
# baked into /opt/venv-vl30b inside the image. The only thing left to do
# at container startup is register the sia_rm vllm plugin entry point
# against the bind-mounted repo path. That requires `pip install -e
# <repo>` so vllm subprocesses discover Qwen3WithScoreForCausalLM via the
# pyproject.toml [project.entry-points."vllm.general_plugins"] section.
#
# pip install -e is fast (~1-2 sec, no actual file copies for editable
# install) and idempotent.

set -euo pipefail

REPO=/workspace/sia-repo/0g-sparse-inference-alignment
VENV_PIP=/opt/venv-vl30b/bin/pip

if [[ -f "$REPO/pyproject.toml" ]]; then
  echo "[entrypoint] registering sia_rm vllm plugin (pip install -e $REPO)"
  "$VENV_PIP" install --quiet --no-deps -e "$REPO"
else
  echo "[entrypoint] WARN: $REPO/pyproject.toml not found"
  echo "[entrypoint]       did you docker run with -v /dstack/persistent/SIA:/workspace ?"
  echo "[entrypoint] continuing without sia_rm plugin — SIA b2 backend will not work"
fi

exec "$@"

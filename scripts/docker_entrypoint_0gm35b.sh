#!/usr/bin/env bash
# Entrypoint for the sia-0gm35b docker image (see Dockerfile.0gm35b at repo root).
#
# Heavy deps (vllm 0.18.0 + torch 2.10.0 + transformers ...) AND the SIA
# source code itself are already baked into the image (see Dockerfile.0gm35b's
# COPY . step) — the repo path below is no longer bind-mounted. The only
# thing left to do at container startup is register the sia_rm vllm plugin
# entry point against that baked-in copy. That requires `pip install -e
# <repo>` so vllm subprocesses discover Qwen3WithScoreForCausalLM via the
# pyproject.toml [project.entry-points."vllm.general_plugins"] section.
#
# pip install -e is fast (~1-2 sec, no actual file copies for editable
# install) and idempotent.

set -euo pipefail

REPO=/workspace/sia-repo/0g-sparse-inference-alignment
VENV_PIP=/opt/venv-0gm35b/bin/pip

if [[ -f "$REPO/pyproject.toml" ]]; then
  echo "[entrypoint] registering sia_rm vllm plugin (pip install -e $REPO)"
  "$VENV_PIP" install --quiet --no-deps -e "$REPO"
else
  echo "[entrypoint] WARN: $REPO/pyproject.toml not found"
  echo "[entrypoint]       this is baked into the image at build time now, not"
  echo "[entrypoint]       bind-mounted — a missing file here means the image"
  echo "[entrypoint]       itself was built wrong, not a missing docker run -v"
  echo "[entrypoint] continuing without sia_rm plugin — SIA b2 backend will not work"
fi

exec "$@"

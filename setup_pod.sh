#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
# LoRA Forge: bootstrap a rented GPU box (any Linux machine with an NVIDIA GPU).
# Idempotent: checks before installing and logs how long each step takes.
# No hosts, ids or keys live here: every credential comes from the environment.
#
#   export HF_TOKEN=...                                    # gated base models (e.g. FLUX.1 dev)
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... # optional S3-compatible bucket for outputs
#   bash setup_pod.sh
set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }
timed() { local s=$(date +%s); "$@"; log "  ^ took $(( $(date +%s) - s ))s"; }

WORKSPACE="${WORKSPACE:-/workspace}"
FORGE_DIR="${FORGE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
AI_TOOLKIT_PATH="${AI_TOOLKIT_PATH:-$WORKSPACE/ai-toolkit}"
export FORGE_ON_POD=1

log "== Preflight =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || log "[!] no nvidia-smi"
df -h "$WORKSPACE" | tail -1

# uv for Python (fast and reproducible)
if ! command -v uv >/dev/null; then
  log "== Installing uv =="
  timed sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
  export PATH="$HOME/.local/bin:$PATH"
fi

# ai-toolkit (the training backend). Idempotent.
if [ ! -f "$AI_TOOLKIT_PATH/run.py" ]; then
  log "== Cloning ai-toolkit =="
  timed git clone https://github.com/ostris/ai-toolkit "$AI_TOOLKIT_PATH"
  (cd "$AI_TOOLKIT_PATH" && git submodule update --init --recursive) || true
else
  log "ai-toolkit already in $AI_TOOLKIT_PATH (not cloning again)"
fi

log "== Installing forge_core =="
cd "$FORGE_DIR"
[ -d .venv ] || timed uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
timed uv pip install -e ".[pod,web]"

# Expected credentials (warn, do not fail).
[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] || log "[!] HF_TOKEN missing (needed for gated base models)."
[ -n "${AWS_ACCESS_KEY_ID:-}" ] || log "[!] No object storage configured: outputs only live on the box's disk."

export AI_TOOLKIT_PATH
log "== forge env =="
forge env || python -m forge_cli.main env

log "== Ready. Start the web app with: python forge_web/api.py (listens on 127.0.0.1:8800)"
log "   and open it from your machine through an SSH tunnel: ssh -L 8800:127.0.0.1:8800 <user>@<gpu-box>"
log "Remember to stop the GPU box when it is idle: it is billed by the hour."

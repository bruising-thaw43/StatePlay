#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${STATEPLAY_BASE_MODEL:-${ROOT}/base_model}"
CHECKPOINT="${STATEPLAY_CHECKPOINT:-${ROOT}/examples/checkpoint/StatePlay.safetensors}"
cd "${ROOT}"
exec "${PYTHON_BIN}" -m stateplay.cli --base-model "${BASE_MODEL}" --checkpoint "${CHECKPOINT}" "$@"

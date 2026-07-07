#!/usr/bin/env bash
set -euo pipefail

export ENV_PROFILE_NAME="${ENV_PROFILE_NAME:-training_gpu}"
export TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_setup_linux_common.sh"

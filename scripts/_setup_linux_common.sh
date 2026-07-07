#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
ENV_PROFILE_NAME="${ENV_PROFILE_NAME:?ENV_PROFILE_NAME must be set}"
TORCH_VERSION="${TORCH_VERSION:?TORCH_VERSION must be set}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:?PYTORCH_INDEX_URL must be set}"

echo "[setup] project root: $ROOT"
echo "[setup] python: $PYTHON_BIN"
echo "[setup] venv: $VENV_DIR"
echo "[setup] profile: $ENV_PROFILE_NAME"
echo "[setup] torch: $TORCH_VERSION"
echo "[setup] pytorch index: $PYTORCH_INDEX_URL"

if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
  cat <<EOF
[setup] failed to create virtual environment with: $PYTHON_BIN -m venv $VENV_DIR
[setup] This usually means your Linux system is missing the stdlib venv/ensurepip package.
[setup] On Debian/Ubuntu, run:
  sudo apt update
  sudo apt install -y python3-venv
[setup] Then rerun the chosen setup script.
EOF
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[setup] virtual environment was created incompletely: $VENV_DIR/bin/python is missing" >&2
  echo "[setup] Remove $VENV_DIR and rerun after installing python3-venv." >&2
  exit 1
fi

cat >"$VENV_DIR/.sigma7_env_profile" <<EOF
ENV_PROFILE_NAME=$ENV_PROFILE_NAME
TORCH_VERSION=$TORCH_VERSION
PYTORCH_INDEX_URL=$PYTORCH_INDEX_URL
EOF

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install "torch==$TORCH_VERSION" --index-url "$PYTORCH_INDEX_URL"
"$VENV_DIR/bin/pip" install -e "$ROOT"

if [[ -n "${SIGMA7_SDK_ROOT:-}" ]]; then
  echo "[setup] building sigma7_pose_udp_sender with SIGMA7_SDK_ROOT=$SIGMA7_SDK_ROOT"
  cmake -S "$ROOT/tools/sigma7_pose_udp_sender" -B "$ROOT/tools/build"
  cmake --build "$ROOT/tools/build" -j
else
  echo "[setup] SIGMA7_SDK_ROOT not set; skipping Sigma7 sender build"
fi

"$VENV_DIR/bin/python" "$ROOT/scripts/doctor_linux.py"

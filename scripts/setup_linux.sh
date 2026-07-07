#!/usr/bin/env bash
set -euo pipefail

echo "[setup] setup_linux.sh defaults to the CPU runtime environment."
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_linux_cpu.sh"

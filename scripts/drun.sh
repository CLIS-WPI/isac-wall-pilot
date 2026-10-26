#!/usr/bin/env bash
# Usage: scripts/drun.sh <command...>
# ISAC_GPU   : GPU selector, e.g. "device=0" or "device=GPU-xxxxxxxx" (default device=0)
# ISAC_DEV=1 : bind-mount the repo so code edits are live without rebuilding
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
GPU="${ISAC_GPU:-device=0}"
EXTRA=()
if [[ "${ISAC_DEV:-0}" == "1" ]]; then
  EXTRA+=(--mount "type=bind,src=$PWD,dst=/workspace")
else
  EXTRA+=(--mount "type=bind,src=$PWD/results,dst=/workspace/results")
fi
exec docker run --rm --gpus "$GPU" --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e MPLBACKEND=Agg -e ISAC_GPU="$GPU" \
  "${EXTRA[@]}" isac-wall-pilot:2.2.0 "$@"

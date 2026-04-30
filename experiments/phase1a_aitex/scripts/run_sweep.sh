#!/usr/bin/env bash
# run_sweep.sh
# Tiny wrapper to sweep multiple seeds for a given config.
# Usage:
#   bash scripts/run_sweep.sh scripts/config/exp03_arch.yaml exp03_resnet18 resnet18
#
# Args:
#   $1 — config YAML
#   $2 — tag prefix (run dir will be <ts>_<tag>_seed<N>)
#   $3 — model name (override; pass "" to leave config default)
#   $4 — loss type (override; optional)
#
# Seeds: 42 43 44. Edit SEEDS env var to change.

set -euo pipefail

CONFIG="${1:?config yaml required}"
TAG="${2:?tag required}"
MODEL_NAME="${3:-}"
LOSS_TYPE="${4:-}"

SEEDS="${SEEDS:-42 43 44}"

EXTRA=()
[[ -n "$MODEL_NAME" ]] && EXTRA+=(--model-name "$MODEL_NAME")
[[ -n "$LOSS_TYPE" ]]  && EXTRA+=(--loss-type "$LOSS_TYPE")

for s in $SEEDS; do
  echo "=== seed $s ==="
  python -m scripts.classification.train --config "$CONFIG" --tag "$TAG" --seed "$s" "${EXTRA[@]}"
done

python scripts/aggregate_results.py

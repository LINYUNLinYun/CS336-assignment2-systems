#!/usr/bin/env bash
set -euo pipefail

SCRIPT="./benchmark_optimizer_sharding.py"
DDP_IMPL="assignment"

run() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
  uv run python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  "$SCRIPT" \
  "$@" \
  --ddp "$DDP_IMPL" \
  --model-size xl
}

run --task memory --optimizer baseline
run --task memory --optimizer sharded
run --task timing --optimizer baseline
run --task timing --optimizer sharded

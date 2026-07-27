#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON_FILE="${PYTHON_FILE:-benchmark_attention.py}"
OUTPUT_FILE="${OUTPUT_FILE:-results/attention/}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float32}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASUREMENT_STEPS="${MEASUREMENT_STEPS:-1}"

uv run python "${PYTHON_FILE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measurement-steps "${MEASUREMENT_STEPS}" \
  --output "${OUTPUT_FILE}"\
  --implementation compiled
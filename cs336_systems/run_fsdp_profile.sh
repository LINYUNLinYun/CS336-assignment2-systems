#!/usr/bin/env bash
set -euo pipefail

# Run from the CS336 assignment repository root after copying
# benchmark_fsdp_accounting.py into cs336_systems/.

OUT_DIR="${OUT_DIR:-results/fsdp_accounting}"
DTYPE="${DTYPE:-fp16}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
REPORT="${OUT_DIR}/fsdp_xl_${DTYPE}"

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
# torch.cuda.nvtx uses dynamic strings; let Nsight match PROFILE_STEP.
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

uv run nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn,nccl \
  --sample=none \
  --wait=all \
  --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=PROFILE_STEP \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${REPORT}" \
  -- torchrun --standalone --nproc_per_node=2 \
    cs336_systems/benchmark_fsdp_accounting.py \
    --compute-dtype "${DTYPE}" \
    --global-batch-size 4 \
    --context-length 512 \
    --warmup-steps 3 \
    --measurement-steps 10 \
    --output-dir "${OUT_DIR}"

echo
echo "Nsight report: ${REPORT}.nsys-rep"
echo "Useful text summaries:"
echo "  uv run nsys stats --report nvtx_sum,nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum ${REPORT}.nsys-rep"
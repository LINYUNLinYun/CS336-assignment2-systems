#!/usr/bin/env bash
set +e

mkdir -p logs

# 你可以在这里改模型规模
MODEL_SIZES=("small" "medium")

# 你可以在这里改 context length
# 作业要求：大于 128，并且是 2 的幂
CONTEXT_LENGTHS=(256 512 1024)

MODE="full_step"
WARMUP_STEPS=5
MEASUREMENT_STEPS=10

TRACE_FLAGS="cuda,nvtx,cudnn,cublas,osrt"
PYTORCH_FLAGS="functions-trace,autograd-shapes-nvtx"

echo "Nsight Systems profiling started."
echo "Models: ${MODEL_SIZES[*]}"
echo "Context lengths: ${CONTEXT_LENGTHS[*]}"
echo

# 清空旧的成功/失败记录
: > logs/success.log
: > logs/failed.log

for model in "${MODEL_SIZES[@]}"; do
  for ctx in "${CONTEXT_LENGTHS[@]}"; do
    name="${model}_ctx${ctx}_${MODE}"

    echo "========================================"
    echo "Running ${name}"
    echo "========================================"

    nsys profile \
      --trace="${TRACE_FLAGS}" \
      --pytorch="${PYTORCH_FLAGS}" \
      --sample=none \
      --cpuctxsw=none \
      --output="reports/${name}" \
      --force-overwrite=true \
      -- python benchmark_nvtx.py \
      --model-size "${model}" \
      --context-length "${ctx}" \
      --mode "${MODE}" \
      --warmup-steps "${WARMUP_STEPS}" \
      --measurement-steps "${MEASUREMENT_STEPS}" \
      2>&1 | tee "logs/${name}.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then
      echo "SUCCESS ${name}" | tee -a logs/success.log
    else
      echo "FAILED ${name}, status=${status}" | tee -a logs/failed.log
    fi

    echo
  done
done

echo "========================================"
echo "All profiling jobs attempted."
echo "Successes:"
cat logs/success.log
echo
echo "Failures:"
cat logs/failed.log
echo "========================================"
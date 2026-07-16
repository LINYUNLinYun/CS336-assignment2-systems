#!/usr/bin/env bash

set -uo pipefail

SCRIPT="benchmark_nvtx.py"
OUTPUT_DIR="results/memory_profiles"
LOG_DIR="${OUTPUT_DIR}/logs"

BATCH_SIZE="${BATCH_SIZE:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
DEVICE="${DEVICE:-cuda:0}"

# CONTEXT_LENGTHS=(128 2048)
CONTEXT_LENGTHS=(128)
MODES=(forward full_step)
PRECISIONS=(none)
# PRECISIONS=(none bf16)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

total=0
succeeded=0
failed=0

echo "========================================"
echo "CS336 Assignment 2 Memory Profiling"
echo "========================================"
echo "Script:       ${SCRIPT}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Warm-up:      ${WARMUP_STEPS}"
echo "Device:       ${DEVICE}"
echo "Output dir:   ${OUTPUT_DIR}"
echo

for context_length in "${CONTEXT_LENGTHS[@]}"; do
    for mode in "${MODES[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            total=$((total + 1))

            experiment="xl_ctx${context_length}_${mode}_${precision}"
            snapshot="${OUTPUT_DIR}/${experiment}.pickle"
            log_file="${LOG_DIR}/${experiment}.log"

            echo "----------------------------------------"
            echo "!!!!!!! Running ${experiment}"
            echo "Snapshot: ${snapshot}"
            echo "Log:      ${log_file}"
            echo "----------------------------------------"

            command=(
                uv run python "${SCRIPT}"
                --model-size xl
                --context-length "${context_length}"
                --batch-size "${BATCH_SIZE}"
                --mode "${mode}"
                --warmup-steps "${WARMUP_STEPS}"
                --measurement-steps 1
                --device "${DEVICE}"
                --mixed-precision "${precision}"
                --profile-memory
                --memory-snapshot "${snapshot}"
            )

            "${command[@]}" 2>&1 | tee "${log_file}"
            status=${PIPESTATUS[0]}

            if [[ ${status} -eq 0 && -f "${snapshot}" ]]; then
                echo "[SUCCESS] ${experiment}"
                succeeded=$((succeeded + 1))
            else
                echo "[FAILED] ${experiment}, exit code: ${status}"
                echo "See log: ${log_file}"
                failed=$((failed + 1))
            fi

            echo
        done
    done
done

echo "========================================"
echo "All experiments finished"
echo "========================================"
echo "Total:     ${total}"
echo "Succeeded: ${succeeded}"
echo "Failed:    ${failed}"
echo "Snapshots: ${OUTPUT_DIR}"
echo "Logs:      ${LOG_DIR}"

if [[ ${failed} -gt 0 ]]; then
    exit 1
fi
#!/bin/bash

models=("small" "medium" "large")
modes=("sync" "flat" "asyn")

for model in "${models[@]}"
do
    for mode in "${modes[@]}"
    do
        echo "=================================================="
        echo "Running model=${model}, mode=${mode}"
        echo "=================================================="

        uv run python benchmark_ddp.py \
            --world-size 2 \
            --model-size ${model} \
            --mode ${mode} \
            --dtype bf16 \
            --batch-size 4 \
            --context-length 512 \
            --warmup-steps 10 \
            --measurement-steps 50 \
            --output-dir results/ddp

        if [ $? -ne 0 ]; then
            echo "FAILED: ${model}-${mode}"
        else
            echo "DONE: ${model}-${mode}"
        fi

        echo ""
    done
done

echo "All benchmarks finished."
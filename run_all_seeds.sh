#!/bin/bash

# Run all models for seeds 1-5 (mirrors run_models_seed1.sh style)
# Usage: bash run_all_seeds.sh

set -e  # Exit on error

MODELS=("tcn" "rnn" "stgnn" "attn_longterm")
DATASET="temperature"
SEEDS=(1 2 3 4 5)

echo "================================================================================"
echo "Running models ${MODELS[@]} for seeds: ${SEEDS[@]}"
echo "Dataset: ${DATASET}"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "================================================================================="
    echo "Running seed: ${SEED}"
    echo "================================================================================="
    echo ""

    for MODEL in "${MODELS[@]}"; do
        echo "---------------------------------------------------------------------------------"
        echo "Model: ${MODEL} | Seed: ${SEED}"
        echo "---------------------------------------------------------------------------------"
        echo ""

        python -u -m experiments.run_temperature_prediction \
            dataset=${DATASET} \
            model=${MODEL} \
            seed=${SEED} \

        echo ""
        echo "✓ ${MODEL} (seed ${SEED}) completed"
        echo ""
    done

    echo "✓ Seed ${SEED} finished"
    echo ""
done

echo "================================================================================"
echo "All seeds completed!"
echo "End time: $(date)"
echo "================================================================================"

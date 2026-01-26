#!/bin/bash
#
# Parallel Execution Script for Deception Detection Experiment
#
# This script runs one scenario per pod for parallel execution on RunPod.
#
# Usage:
#   Run on 3 separate RunPod instances:
#     Pod 1: ./run_parallel.sh ultimatum_bluff
#     Pod 2: ./run_parallel.sh hidden_value
#     Pod 3: ./run_parallel.sh alliance_betrayal
#
# After all pods complete, merge results locally.

set -e

# Configuration
SCENARIO=${1:-"ultimatum_bluff"}
MODEL="google/gemma-2-2b-it"
TRIALS=40
MAX_ROUNDS=3
MAX_TOKENS=128
OUTPUT_DIR="./outputs/${SCENARIO}"

echo "=============================================="
echo "DECEPTION DETECTION EXPERIMENT - PARALLEL MODE"
echo "=============================================="
echo "Scenario:   ${SCENARIO}"
echo "Model:      ${MODEL}"
echo "Trials:     ${TRIALS} per condition"
echo "Max rounds: ${MAX_ROUNDS}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Output:     ${OUTPUT_DIR}"
echo "=============================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the experiment for this specific scenario
python run_deception_experiment.py \
    --mode emergent \
    --scenario-name ${SCENARIO} \
    --model ${MODEL} \
    --trials ${TRIALS} \
    --max-rounds ${MAX_ROUNDS} \
    --max-tokens ${MAX_TOKENS} \
    --device cuda \
    --dtype bfloat16 \
    --output ${OUTPUT_DIR}

echo ""
echo "=============================================="
echo "SCENARIO COMPLETE: ${SCENARIO}"
echo "=============================================="
echo "Output saved to: ${OUTPUT_DIR}"
echo ""
echo "After all pods complete, merge results with:"
echo "  python merge_results.py outputs/"

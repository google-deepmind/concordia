#!/bin/bash

AGENT_NAME="orp_base"
API_NAME="together_ai"
API_KEY="8114746e02d5a8b5c2133ab42c8ff5c377226f0307df6bd3d8250f833f628f31"
ENDPOINT_ID="3292325941165948928"
MODEL_NAME="google/gemma-2-9b-it"
EMBEDDER_NAME="all-mpnet-base-v2"
NUM_REPETITIONS_PER_SCENARIO=1

PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/launch_concordia_challenge_evaluation.py \
  --agent=$AGENT_NAME \
  --endpoint_id=$ENDPOINT_ID \
  --model=$MODEL_NAME \
  --embedder=$EMBEDDER_NAME \
  --num_repetitions_per_scenario=$NUM_REPETITIONS_PER_SCENARIO

#--api_type=$API_NAME \
#--api_key=$API_KEY \

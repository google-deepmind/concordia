# Reproducing the NeurIPS 2024 Concordia Contest

This document explains how to evaluate agent submissions to the NeurIPS 2024 Concordia Contest using the upgraded (v2.4.0) API.

## Background
The scripts used to evaluate agents during the NeurIPS 2024 contest were initially located in `examples/modular/`. Due to changes in the game engine architecture from "factories" to the modern "prefabs/entities" API in v2.0.0, the old modular framework was deprecated and deleted (commit `e253beta...`).

To maintain the ability to reproduce the contest results and evaluate new agents, we provide modern, lightweight replacement scripts in `examples/games/` that invoke the v2.4.0 game engine while supporting the original contest metrics.

## Setup Steps
1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your preferred LLM API keys. See the main documentation on `concordia.contrib.language_models` regarding options (e.g., `OPENAI_API_KEY`).

## Commands to Run

### 1. Evaluating an Agent
To evaluate a focal agent prefab (e.g., `rational__Entity` or your custom submission) against all standard and held-out NeurIPS 2024 contest scenarios:
```bash
PYTHONPATH=. python examples/games/evaluate_contest.py \
  --agent rational__Entity \
  --api_type openai \
  --model gpt-4o
```
You can pass `--disable_language_model` to test syntax instantly without any API costs.
Output logs and scores will be emitted dynamically, and the final average results per scenario will be written into a JSON file in the `evaluations/` folder.

### 2. Calculating Elo Ratings
Contest evaluation is based on computing relative Elo metrics between submissions according to scoring formulas. After evaluating multiple agents (which will populate the `evaluations/` directory with `*_out.json` files), you can calculate the Elo rankings:
```bash
PYTHONPATH=. python examples/games/calculate_ratings.py --eval_dir evaluations
```

## Known Limitations
* **Model Versions**: LLM outputs drift over time. Since NeurIPS 2024, the exact snapshots of GPT-4 or Gemini used might be deprecated. It is normal if agents receive slightly different scores in exact reproducible metrics.
* **Component Changes**: Under-the-hood engine changes in Concordia since the contest might cause marginal behavior deviations in background puppet/rational agents.
* **Cost**: Running the full gauntlet of contest scenarios requires substantial language model API tokens.

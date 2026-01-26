# RunPod Setup Guide: Emergent Deception Experiment

Complete setup steps for running the deception detection experiment on RunPod.

## Configuration Summary

| Setting | Value |
|---------|-------|
| **Model** | Gemma 2B (`google/gemma-2-2b-it`) |
| **GPU** | RTX 4090 (24GB) |
| **Scenarios** | 3 (ultimatum_bluff, hidden_value, alliance_betrayal) |
| **Trials** | 40 per condition (80 per scenario) |
| **Rounds** | 3 per trial |
| **Max Tokens** | 128 |
| **Estimated Time** | ~5-6 hours (single pod) or ~2 hours (3 parallel pods) |
| **Estimated Cost** | ~$5-8 |

---

## Prerequisites

- RunPod account
- HuggingFace account with access to Gemma 2

### HuggingFace Setup (do this first)

1. Go to [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
2. Click "Accept License" to get access
3. Go to [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
4. Create a new token (read access is sufficient)

---

## Option A: Single Pod Execution

Run all 3 scenarios on one RTX 4090 pod.

### Quick Setup (Copy-Paste All)

```bash
# 1. Clone and checkout
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia
git checkout emergent-deception-v2

# 2. Install Concordia base
pip install -e .

# 3. Install evaluation dependencies
cd concordia/prefabs/entity/negotiation/evaluation
pip install -r requirements.in

# 4. Fix transformer-lens compatibility (IMPORTANT)
pip install transformers==4.44.0 accelerate==0.33.0

# 5. Login to HuggingFace (paste token when prompted)
huggingface-cli login

# 6. Run experiment
python run_deception_experiment.py --mode emergent --device cuda --dtype bfloat16
```

### Step-by-Step Verification

After step 4, verify setup works:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformer_lens; print('TransformerLens: OK')"
python -c "from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner; print('Imports: OK')"
```

---

## Option B: Parallel Pod Execution (Recommended)

Run each scenario on a separate RTX 4090 pod for ~3x speedup.

### Setup Script (same for all pods)

Copy this to each pod first:
```bash
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia
git checkout emergent-deception-v2
pip install -e .
cd concordia/prefabs/entity/negotiation/evaluation
pip install -r requirements.in
pip install transformers==4.44.0 accelerate==0.33.0
huggingface-cli login
```

### Pod 1: ultimatum_bluff

After setup, run:
```bash
python run_deception_experiment.py --scenario-name ultimatum_bluff --device cuda --dtype bfloat16 --output ./outputs/ultimatum_bluff
```

### Pod 2: hidden_value

After setup, run:
```bash
python run_deception_experiment.py --scenario-name hidden_value --device cuda --dtype bfloat16 --output ./outputs/hidden_value
```

### Pod 3: alliance_betrayal

After setup, run:
```bash
python run_deception_experiment.py --scenario-name alliance_betrayal --device cuda --dtype bfloat16 --output ./outputs/alliance_betrayal
```

### Merge Results

After all pods complete, download the outputs and merge locally:

```bash
python merge_results.py outputs/ --train-probes
```

---

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | emergent | `emergent` (incentive-based) or `instructed` (explicit) |
| `--scenario-name` | None | Run specific scenario (for parallel execution) |
| `--scenarios` | 3 | Number of scenarios (max 6) |
| `--trials` | 40 | Trials per scenario per condition |
| `--max-rounds` | 3 | Max negotiation rounds per trial |
| `--max-tokens` | 128 | Max tokens per LLM response |
| `--model` | google/gemma-2-2b-it | HuggingFace model name |
| `--device` | cuda | `cuda` or `cpu` |
| `--dtype` | bfloat16 | `bfloat16`, `float16`, or `float32` |
| `--output` | ./experiment_output | Output directory |

---

## Expected Output

### With Defaults (3 scenarios)

- 3 scenarios × 2 conditions × 40 trials = **240 total trials**
- Estimated time: **~5-6 hours** (single pod) or **~2 hours** (3 parallel pods)

### Runtime Estimates (RTX 4090, Gemma 2B)

| Setup | Trials | Estimated Time | Cost |
|-------|--------|----------------|------|
| Single pod | 240 | ~5-6 hrs | ~$2-3 |
| 3 parallel pods | 240 | ~1.5-2 hrs | ~$3-4 |

---

## Output Files

After completion, find results in output directory:

```
outputs/
├── ultimatum_bluff/
│   ├── activations_emergent_*.pt
│   └── probe_results.json
├── hidden_value/
│   └── ...
├── alliance_betrayal/
│   └── ...
└── merged_activations.pt  # After running merge_results.py
```

---

## Troubleshooting

### "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "ModuleNotFoundError: Could not import module 'BertForPreTraining'"
```bash
pip install transformers==4.44.0 accelerate==0.33.0
```

### "Access denied" for Gemma model
1. Accept license at https://huggingface.co/google/gemma-2-2b-it
2. Run `huggingface-cli login` with valid token

### Check GPU usage
```bash
nvidia-smi
```

### Check if experiment is running
```bash
ps aux | grep python
```

---

## Quick Reference

```bash
# Single pod (all 3 scenarios)
python run_deception_experiment.py --mode emergent --device cuda

# Parallel pod (specific scenario)
python run_deception_experiment.py --scenario-name ultimatum_bluff --device cuda

# Monitor
tail -f experiment.log

# Kill experiment
pkill -f run_deception_experiment

# Merge parallel results
python merge_results.py outputs/ --train-probes
```

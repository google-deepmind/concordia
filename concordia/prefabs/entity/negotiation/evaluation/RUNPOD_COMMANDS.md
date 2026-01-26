# RunPod Commands - Copy & Paste Ready

## Quick Setup (Copy entire block)

```bash
cd /workspace && \
git clone https://github.com/tesims/concordia.git && \
cd concordia && \
git checkout hybrid-sae-experiment && \
pip install -e . && \
pip install -r concordia/prefabs/entity/negotiation/evaluation/requirements.in && \
pip install transformers==4.44.0 accelerate==0.33.0 && \
huggingface-cli login
```

## Validate Setup

```bash
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f"HuggingFace: {user['name']}")

from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner, EMERGENT_SCENARIOS
print(f"Scenarios: {list(EMERGENT_SCENARIOS.keys())}")
print("✅ All imports OK")
EOF
```

## Quick Test (1 trial)

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/test_output && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name ultimatum_bluff \
    --trials 1 \
    --max-rounds 1 \
    --hybrid \
    --fast \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/test_output
```

## Full Experiment (All Scenarios)

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/outputs && \
for scenario in ultimatum_bluff hidden_value alliance_betrayal info_withholding promise_break; do
    echo "========================================"
    echo "Starting: $scenario at $(date)"
    echo "========================================"
    python -u run_deception_experiment.py \
        --mode emergent \
        --scenario-name $scenario \
        --trials 25 \
        --max-rounds 3 \
        --hybrid \
        --sae \
        --fast \
        --device cuda \
        --dtype bfloat16 \
        --output /workspace/persistent/outputs/$scenario
done && \
echo "✅ All scenarios complete!"
```

## Single Scenario (for parallel pods)

### Pod 1:
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name ultimatum_bluff \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/ultimatum_bluff
```

### Pod 2:
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name hidden_value \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/hidden_value
```

### Pod 3:
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name alliance_betrayal \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/alliance_betrayal
```

## Train Probes on Existing Data

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --train-only \
    --data /workspace/persistent/outputs/activations_emergent_*.pt \
    --output /workspace/persistent/outputs
```

## Merge Activations from Multiple Runs

```bash
python << 'EOF'
import torch
from pathlib import Path
import glob

output_base = Path("/workspace/persistent/outputs")
all_activations = {}
all_labels = {"gm_labels": [], "agent_labels": [], "scenario": []}

for pt_file in glob.glob(str(output_base / "*/activations_*.pt")):
    print(f"Loading: {pt_file}")
    data = torch.load(pt_file, weights_only=False)

    for layer, acts in data.get("activations", {}).items():
        if layer not in all_activations:
            all_activations[layer] = []
        all_activations[layer].append(acts)

    labels = data.get("labels", {})
    all_labels["gm_labels"].extend(labels.get("gm_labels", []))
    all_labels["agent_labels"].extend(labels.get("agent_labels", []))
    all_labels["scenario"].extend(labels.get("scenario", []))

for layer in all_activations:
    all_activations[layer] = torch.cat(all_activations[layer], dim=0)

merged = {"activations": all_activations, "labels": all_labels}
torch.save(merged, output_base / "merged_activations.pt")
print(f"✅ Merged {len(all_labels['gm_labels'])} samples")
EOF
```

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `emergent` | `emergent` (incentive-based) or `instructed` (explicit) |
| `--model` | `google/gemma-2-9b-it` | HuggingFace model name |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dtype` | `bfloat16` | `float32`, `float16`, or `bfloat16` |
| `--scenario-name` | None | Single scenario (for parallel runs) |
| `--scenarios` | `3` | Number of scenarios (1-6) |
| `--trials` | `40` | Trials per scenario per condition |
| `--max-rounds` | `3` | Max negotiation rounds per trial |
| `--max-tokens` | `128` | Max tokens per LLM response |
| `--hybrid` | False | Use HuggingFace + TransformerLens (20x faster) |
| `--sae` | False | Enable SAE feature extraction |
| `--sae-layer` | `21` | Layer for SAE (middle layer for 9B) |
| `--fast` | False | Disable ToM module (~3x speedup) |
| `--ultrafast` | False | Minimal agents (~5x additional speedup) |
| `--output` | `./experiment_output` | Output directory |
| `--checkpoint-dir` | None | Checkpoint directory (for crash recovery) |
| `--train-only` | False | Only train probes on existing data |
| `--data` | None | Path to activations.pt for `--train-only` |
| `--causal` | False | Run causal validation (activation patching, ablation tests) |
| `--causal-samples` | `20` | Number of samples for causal validation tests |

## Troubleshooting

### CUDA out of memory
```bash
# Use smaller dtype
--dtype bfloat16

# Or reduce max tokens
--max-tokens 64
```

### TransformerLens errors
```bash
# Reinstall exact versions
pip install transformers==4.44.0 accelerate==0.33.0
```

### HuggingFace access denied
```bash
# Login and accept Gemma license
huggingface-cli login
# Then visit: https://huggingface.co/google/gemma-2-9b-it and accept
```

### Import errors
```bash
# Reinstall Concordia
cd /workspace/concordia
pip install -e .
```

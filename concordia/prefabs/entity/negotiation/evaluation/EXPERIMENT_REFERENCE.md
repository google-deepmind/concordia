# Deception Detection Experiment - Complete Reference

## Quick Start (RunPod RTX 4090)

```bash
# Setup
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia
git checkout emergent-deception-v2
pip install -e .
cd concordia/prefabs/entity/negotiation/evaluation
pip install -r requirements.in
pip install transformers==4.44.0 accelerate==0.33.0
huggingface-cli login

# Quick test (~30 sec with --ultrafast)
python run_deception_experiment.py --scenario-name ultimatum_bluff --trials 1 --fast --ultrafast --device cuda --dtype bfloat16

# Full experiment - FASTEST (~20-30 min with --ultrafast)
python run_deception_experiment.py --fast --ultrafast --device cuda --dtype bfloat16

# Full experiment - Balanced (~2 hrs with --fast only)
python run_deception_experiment.py --fast --device cuda --dtype bfloat16
```

---

## Configuration

### Optimized Settings (Current)

| Setting | Value | Reason |
|---------|-------|--------|
| Model | `google/gemma-2-2b-it` | Fast inference, fits in 24GB |
| GPU | RTX 4090 (24GB) | Best cost/performance for 2B |
| Scenarios | 3 | ultimatum_bluff, hidden_value, alliance_betrayal |
| Trials | 40 per condition | Statistical significance |
| Rounds | 3 per trial | Captures deception decision |
| Max tokens | 128 | Sufficient for negotiation responses |
| Fast mode | `--fast` | Disables ToM, ~3x speedup |
| Ultrafast mode | `--ultrafast` | Uses minimal agents, ~5x additional speedup |
| **Hybrid mode** | `--hybrid` | HuggingFace generation + TransformerLens capture, ~20x speedup |
| **SAE mode** | `--hybrid --sae` | Adds Gemma Scope SAE feature extraction |

### Command Line Arguments

```
--mode          emergent|instructed|both (default: emergent)
--model         HuggingFace model name (default: google/gemma-2-2b-it)
--device        cuda|cpu (default: cuda)
--dtype         bfloat16|float16|float32 (default: bfloat16)
--scenarios     Number of scenarios 1-6 (default: 3)
--scenario-name Run specific scenario (for parallel pods)
--trials        Trials per condition (default: 40)
--max-rounds    Rounds per trial (default: 3)
--max-tokens    Max tokens per response (default: 128)
--fast          Disable ToM module for ~3x speedup
--ultrafast     Use minimal agents for ~5x additional speedup
--hybrid        Use HuggingFace+TransformerLens hybrid for ~20x speedup
--sae           Enable Gemma Scope SAE feature extraction (requires --hybrid)
--sae-layer     Layer for SAE extraction (default: 12)
--output        Output directory (default: ./experiment_output)
--train-only    Only train probes on existing data
--data          Path to activations file for --train-only
```

---

## Parallel Execution (3 Pods)

### Setup (same for all pods)
```bash
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia && git checkout emergent-deception-v2
pip install -e .
cd concordia/prefabs/entity/negotiation/evaluation
pip install -r requirements.in
pip install transformers==4.44.0 accelerate==0.33.0
huggingface-cli login
```

### Run Commands (Recommended: --fast --ultrafast)
- **Pod 1:** `python run_deception_experiment.py --scenario-name ultimatum_bluff --fast --ultrafast --device cuda --dtype bfloat16 --output ./outputs/ultimatum_bluff`
- **Pod 2:** `python run_deception_experiment.py --scenario-name hidden_value --fast --ultrafast --device cuda --dtype bfloat16 --output ./outputs/hidden_value`
- **Pod 3:** `python run_deception_experiment.py --scenario-name alliance_betrayal --fast --ultrafast --device cuda --dtype bfloat16 --output ./outputs/alliance_betrayal`

### Merge Results
```bash
python merge_results.py outputs/ --train-probes
```

---

## Time & Cost Estimates

| Configuration | Trials | Time | Cost (RunPod) |
|---------------|--------|------|---------------|
| 1 trial test `--fast --ultrafast` | 2 | ~30 sec | ~$0.01 |
| Single pod `--fast --ultrafast` | 240 | ~20-30 min | ~$0.30 |
| 3 parallel pods `--fast --ultrafast` | 240 | ~10-15 min | ~$0.40 |
| Single pod `--fast` only | 240 | ~2-3 hrs | ~$1-2 |
| 3 parallel pods `--fast` only | 240 | ~1-1.5 hrs | ~$1.50 |
| Single pod with ToM (no flags) | 240 | ~8-12 hrs | ~$4-5 |

---

## Performance Analysis: LLM Calls Per Round

Understanding why different modes have vastly different speeds:

### Standard Mode (no flags)
Each round makes **~14+ LLM calls** per round:
- Main agent (base_negotiator.py):
  - `question_about_situation` (QuestionOfRecentMemories)
  - `question_about_self` (QuestionOfRecentMemories)
  - `question_about_action` (QuestionOfRecentMemories)
  - `AllSimilarMemories` (memory retrieval)
  - `ConcatActComponent` (final action)
- Theory of Mind module adds 4+ more calls
- Counterpart agent: Same 5 base calls

**Total: ~14+ calls × 3 rounds × 240 trials = ~10,000+ LLM calls**

### --fast Mode
Disables ToM, but base agent still has 5 components:
- **~10 LLM calls per round** (5 per agent × 2 agents)
- **Total: ~7,200 LLM calls**

### --fast --ultrafast Mode
Uses minimal agents with only 1 LLM call per action:
- **~3 LLM calls per round** (1 per agent × 2 agents + 1 for GM deception detection)
- **Total: ~2,160 LLM calls**
- **~3x fewer calls than --fast alone**

### Trade-offs

| Mode | LLM Calls | Cognitive Richness | Ground Truth | Recommended For |
|------|-----------|-------------------|--------------|-----------------|
| Standard | 10,000+ | Full (ToM, reasoning) | GM LLM-based | Final research |
| `--fast` | 7,200+ | Medium (no ToM) | GM LLM-based | Balanced speed/quality |
| `--fast --ultrafast` | 2,160 | Basic (action only) | GM LLM-based | Quick iteration |

All modes now use **GM LLM-based ground truth detection**, which:
- Compares agent response against known ground truth params
- Returns nuanced scores (0.0-1.0) instead of binary
- Catches subtle deception that regex misses

---

## Known Issues & Fixes

### 1. `ModuleNotFoundError: No module named 'absl'`
```bash
pip install absl-py
```

### 2. `AttributeError: module 'enum' has no attribute 'StrEnum'`
Python version too old. Need Python 3.11+.
Use RunPod template with Python 3.11+ (e.g., "RunPod Pytorch 2.4")

### 3. `ModuleNotFoundError: Could not import module 'BertForPreTraining'`
```bash
pip install transformers==4.44.0 accelerate==0.33.0
```

### 4. `Access denied` for Gemma model
1. Accept license at https://huggingface.co/google/gemma-2-2b-it
2. Run `huggingface-cli login` with valid token

### 5. Theory of Mind (ToM) too slow
ToM makes multiple LLM calls per round (emotion detection, mental modeling).
Use `--fast` flag to disable ToM for ~3x speedup.

### 6. Stop running experiment
```bash
# In same terminal
Ctrl+C

# Or from another terminal
pkill -f run_deception_experiment
```

---

## File Structure

```
concordia/prefabs/entity/negotiation/evaluation/
├── run_deception_experiment.py    # Main entry point
├── interpretability_evaluation.py # InterpretabilityRunner, TransformerLensWrapper
├── emergent_prompts.py            # 6 emergent deception scenarios
├── deception_scenarios.py         # Instructed deception scenarios
├── train_probes.py                # Ridge probe training
├── sanity_checks.py               # Probe validation
├── mech_interp_tools.py           # TransformerLens utilities
├── merge_results.py               # Merge parallel pod outputs
├── run_parallel.sh                # Parallel execution script
├── requirements.in                # Dependencies
├── RUNPOD_SETUP.md                # Setup guide
└── EXPERIMENT_REFERENCE.md        # This file
```

---

## Architecture Overview

### Data Flow
```
run_deception_experiment.py
    │
    ├── InterpretabilityRunner (interpretability_evaluation.py)
    │   └── TransformerLensWrapper (captures activations)
    │
    ├── Scenarios (emergent_prompts.py)
    │   ├── ultimatum_bluff (false final offer claims)
    │   ├── hidden_value (inflated asking price)
    │   └── alliance_betrayal (assure ally, consider betrayal)
    │
    ├── Agents
    │   ├── advanced_negotiator.py (full agent with --fast)
    │   ├── minimal.py (ultrafast agent with --ultrafast)
    │   └── Optional: theory_of_mind module (disabled with --fast)
    │
    └── Analysis
        ├── train_probes.py (ridge regression on activations)
        └── sanity_checks.py (validation)
```

### Key Classes
- `InterpretabilityRunner`: Main orchestrator, runs trials, collects activations
- `TransformerLensWrapper`: Wraps HookedTransformer, captures layer activations
- `ActivationSample`: Dataclass storing activations + labels per sample

### Labels Captured
- **Agent labels** (from ToM module, empty if --fast):
  - perceived_deception, emotion_intensity, trust_level, cooperation_intent
- **GM labels** (ground truth from scenario rules):
  - actual_deception, commitment_violation, manipulation_score, consistency_score

**GM-BASED GROUND TRUTH**: The experiment now uses LLM-based deception detection via
`_detect_deception_with_llm()`. This method:
1. Knows the ground truth params (true_walkaway, true_days, etc.)
2. Uses LLM reasoning to compare agent's response against ground truth
3. Returns nuanced scores (0.0-1.0) for deception, manipulation, withholding

This adds **1 LLM call per agent action** for deception detection, but provides
more accurate and nuanced ground truth than regex pattern matching.

---

## --ultrafast Mode Explained

The `--ultrafast` flag replaces the full Concordia agent with a minimal agent.

### What --ultrafast changes:
| Component | Full Agent | Minimal Agent |
|-----------|------------|---------------|
| `question_about_situation` | LLM call | ❌ Removed |
| `question_about_self` | LLM call | ❌ Removed |
| `question_about_action` | LLM call | ❌ Removed |
| `AllSimilarMemories` | LLM call | ❌ Removed |
| `ConcatActComponent` | LLM call | ✅ Kept |

### What --ultrafast does NOT change:
- ✅ **Ground truth labels** - GM LLM-based detection still runs
- ✅ **Activations captured** - same TransformerLens hooks on response generation
- ✅ **Probe training** - same activation vectors fed to ridge regression
- ✅ **Scenario prompts** - same instructions given to agent

### Bottom line:
`--ultrafast` reduces LLM calls from ~10/round to ~3/round without affecting the core
research outputs (activations + GM ground truth labels). The agent just "thinks less"
before responding, but we still use LLM-based GM detection for accurate ground truth.

---

## ToM Performance Issue

The Theory of Mind module in `components/theory_of_mind.py` makes extra LLM calls:
1. `_detect_emotions()` - analyzes counterpart's emotional state
2. `_build_mental_model()` - constructs opponent model
3. `_assess_deception_indicators()` - checks for deception signals

Each call adds ~30-60 seconds per round. With 3 rounds × 2 agents × multiple calls = very slow.

**Workaround:** Use `--fast` flag to disable ToM entirely.

**Potential fix:** Batch these calls or cache results within a round.

---

## Output Files

After experiment completes:
```
experiment_output/
├── activations_emergent_YYYYMMDD_HHMMSS.pt  # Activation dataset
└── probe_results.json                        # Probe training results
```

### Loading Results
```python
import torch

data = torch.load('experiment_output/activations_emergent_*.pt')
print(f"Samples: {len(data['samples'])}")

# Check deception rates
for sample in data['samples']:
    print(f"{sample.emergent_scenario}: deception={sample.actual_deception}")
```

---

## Research Quality Notes

### What --fast --ultrafast preserves:
- Activation capture from main agent responses
- GM ground truth labels (from scenario rules)
- Cross-scenario generalization testing
- Probe training on layer activations
- Core deception detection capability

### What --fast --ultrafast loses:
- Rich agent labels (perceived_deception, trust_level, etc.)
- Agent perspective on counterpart's mental state
- Recursive belief modeling
- Multi-step reasoning components (situation/self/action questions)

### Research Validity
For initial deception detection research, `--fast --ultrafast` is acceptable because:
1. **Ground truth is external**: Deception labels come from scenario rules (regex on response), not agent cognition
2. **Activations are complete**: Layer activations are captured on the same response generation
3. **Probes work identically**: Ridge probes train on the same activation vectors

The main difference is the agent's internal reasoning process, which we don't use for ground truth labels.

---

## Version Info

- Branch: `emergent-deception-v2`
- Model: Gemma 2 2B (`google/gemma-2-2b-it`)
- TransformerLens: requires `transformers==4.44.0`
- Python: 3.11+ required

# Resource Dilemma Example

This example implements common pool resource (CPR) experiments where agents must
decide how much of a shared resource to harvest. It includes four scenarios:
Pasture (tragedy of the commons with grazing), Irrigation (shared water
infrastructure), Network (bandwidth allocation), and Fishery (sustainable
fishing). Each scenario supports two governance modes: "standard" (unregulated
free-for-all) and "election" (agents elect leaders and vote on harvest
policies).

## Quick Start

### Using OpenAI

```bash
pip install concordia[contrib] sentence-transformers

python -m concordia.examples.resource_dilemma.run \
  --api_type=openai \
  --model_name=gpt-4o-mini \
  --scenario=pasture \
  --mode=standard \
  --num_cycles=6
```

### Using Google AI Studio

```bash
export GOOGLE_API_KEY=your_key_here

python -m concordia.examples.resource_dilemma.run \
  --api_type=gemini \
  --model_name=gemini-2.0-flash \
  --scenario=irrigation \
  --mode=election
```

### Using Ollama (local)

```bash
python -m concordia.examples.resource_dilemma.run \
  --api_type=ollama \
  --model_name=llama3 \
  --scenario=fishery \
  --num_cycles=3
```

### Testing (no LLM required)

```bash
python -m concordia.examples.resource_dilemma.run \
  --disable_language_model \
  --use_dummy_embedder \
  --scenario=network \
  --num_cycles=2
```

## Scenarios

| Scenario      | Resource                | Description                                      |
| ------------- | ----------------------- | ------------------------------------------------ |
| `pasture`     | Grazing land            | Herders decide how many cattle to graze on a shared pasture. |
| `irrigation`  | Water infrastructure    | Irrigators share a canal system and decide how much water to draw. |
| `network`     | Bandwidth               | Users allocate bandwidth on a shared network link. |
| `fishery`     | Fish stock              | Fishers decide how much to harvest from a shared fishery. |

## Governance Modes

- **`standard`** — Unregulated. Agents independently choose how much of the
  resource to harvest each cycle with no collective rules or enforcement.
- **`election`** — Governed. Agents elect leaders and vote on harvest policies.
  Elections occur every `--election_every_n` cycles (default: every cycle).

## Key Parameters

| Flag                       | Default                           | Description                                    |
| -------------------------- | --------------------------------- | ---------------------------------------------- |
| `--scenario`               | `pasture`                         | CPR scenario (`pasture`, `irrigation`, `network`, `fishery`). |
| `--mode`                   | `standard`                        | Governance mode (`standard`, `election`).       |
| `--num_cycles`             | `6`                               | Number of harvest cycles to simulate.           |
| `--election_every_n`       | `1`                               | How often to hold elections (election mode only).|
| `--api_type`               | `openai`                          | Language model provider.                        |
| `--model_name`             | `gpt-4o`                          | Language model name.                            |
| `--api_key`                | `None`                            | API key (if not set via environment variable).  |
| `--disable_language_model` | `false`                           | Use a mock model for testing.                   |
| `--use_dummy_embedder`     | `false`                           | Use a zero-vector embedder instead of sentence-transformers. |
| `--output_dir`             | `/tmp/resource_dilemma_results`   | Directory for output files.                     |
| `--seed`                   | `42`                              | Random seed for reproducibility.                |

## Output

Results are written to `--output_dir`
(default: `/tmp/resource_dilemma_results`):

- **HTML log** — `<scenario>_<mode>_log.html` containing the full simulation
  transcript.

## File Structure

```
resource_dilemma/
├── README.md                         # This file
├── run.py                            # Main entry point
├── simulation_state.py               # Shared simulation state
├── resource_logger.py                # Resource logging utilities
├── __init__.py
├── scenarios/
│   ├── __init__.py
│   ├── pasture.py                    # Pasture scenario
│   ├── irrigation.py                 # Irrigation scenario
│   ├── network.py                    # Network scenario
│   └── fishery.py                    # Fishery scenario
├── personas/
│   ├── __init__.py
│   ├── pasture_personas.py           # Herder agent configs
│   ├── irrigation_personas.py        # Irrigator agent configs
│   ├── network_personas.py           # Network user agent configs
│   └── fishery_personas.py           # Fisher agent configs
└── gamemaster/
    ├── __init__.py
    ├── resource_components.py        # Resource tracking components
    ├── harvesting_game_master.py     # Harvesting Game Master
    ├── discussion_game_master.py     # Discussion Game Master
    └── voting_game_master.py         # Voting/election Game Master
```

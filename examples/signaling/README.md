# Signaling Marketplace Example

This example implements the experiment from *"A Generative Model of Conspicuous
Consumption and Status Signaling"*, a multi-day marketplace simulation where
agents buy and sell goods, then engage in social interactions (dates) where they
signal their status through their consumption choices.

## Quick Start

### Using OpenAI

```bash
pip install concordia[contrib] sentence-transformers

python -m concordia.examples.signaling.run \
  --api_type=openai \
  --model_name=gpt-4o \
  --num_days=3 \
  --num_agents=10 \
  --condition=social
```

### Using Google AI Studio

```bash
export GOOGLE_API_KEY=your_key_here

python -m concordia.examples.signaling.run \
  --api_type=google_aistudio \
  --model_name=gemini-2.0-flash \
  --num_days=3 \
  --num_agents=10
```

### Using Ollama (local)

```bash
python -m concordia.examples.signaling.run \
  --api_type=ollama \
  --model_name=llama3 \
  --num_days=1 \
  --num_agents=4
```

### Testing (no LLM required)

```bash
python -m concordia.examples.signaling.run \
  --disable_language_model \
  --use_dummy_embedder \
  --num_days=1 \
  --num_agents=4
```

## Experimental Conditions

The `--condition` flag controls the experimental manipulation:

| Condition | Marketplace | Personal Events | Date Conversation |
|-----------|:-----------:|:---------------:|:-----------------:|
| `social` (default) | ✅ | ✅ | ✅ |
| `asocial` | ✅ | ❌ | ❌ |
| `asocial_personal` | ✅ | ✅ | ❌ |

- **`social`**: Full simulation. Agents shop in the marketplace, experience
  personal daily events, and go on dates where they can observe each other's
  consumption choices.
- **`asocial`**: Marketplace only. Agents shop but have no social interactions.
  Tests whether signaling behavior emerges without social feedback.
- **`asocial_personal`**: Marketplace + personal events but no dates. Agents
  experience daily life but don't interact with potential partners.

## Architecture

Each simulated day has two phases:

```
Day Loop
├── Phase 1: Marketplace
│   ├── Agents receive budgets and browse goods (food, clothing, gadgets)
│   ├── Clearing-house market matches buyers and sellers
│   └── Agents reflect on purchases
│
└── Phase 2: DIAL (Day-in-a-Life + Dialogue)  [skipped in asocial condition]
    ├── Personal mundane events are generated for each agent
    ├── Mixed-sex dyads are formed for date conversations
    ├── Agents observe each other's clothing/accessories (signaling!)
    └── Post-date reflections and ratings
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_days` | 5 | Number of simulated days |
| `--num_agents` | 10 | Number of agents (max 50) |
| `--num_marketplace_rounds` | 5 | Trading rounds per day |
| `--num_dial_rounds` | 80 | Conversation steps per dyad |
| `--item_list` | original | Goods set: original, synthetic, subculture, neutral_tag, both |
| `--seed` | 42 | Random seed for reproducibility |

## Output

Results are written to `/tmp/`:

- `signaling_marketplace_day_N.html` — HTML log of marketplace activity
- `signaling_marketplace_day_N.json` — JSON log of marketplace activity
- `signaling_dial_day_N_Name1_and_Name2.html` — HTML log of each date
- `signaling_trades.json` — All marketplace trades across all days
- `signaling_prices.json` — Price history across all days

## File Structure

```
examples/signaling/
├── run.py              # Entry point (BYO model)
├── simulation.py       # Multi-day orchestration loop
├── dial.py             # DIAL dyad setup and runner
├── agents/
│   ├── consumer.py     # Marketplace agent prefab
│   └── convo_agent.py  # Conversation agent prefab
├── configs/
│   ├── goods.py        # Goods definitions (food, clothing, gadgets)
│   ├── signaling.py    # Signaling scenario configurations
│   └── personas.py     # 50 hardcoded agent personas + utilities
└── README.md
```

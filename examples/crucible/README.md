# Crucible (Twelve Words)

**Twelve Words** is the game these toys became: one standing order of at
most twelve words, read three ways by a bold, a cautious and a literal crew
member, turn after turn. It is the `orders` toy. Below, the toys it grew out
of: grey-box toys for an original game on Concordia, built mechanic first (see
[DESIGN.md](DESIGN.md) and [ROADMAP.md](ROADMAP.md)). Each toy is one
candidate generative rule in which a language model judges what words mean
and deterministic code applies the consequences.

## Quickstart

Offline, with the deterministic mock (about 0.1 s):

```bash
python -m examples.crucible.run --toy=claims --mode=mock --seat=skilled
```

Play the seat yourself at the terminal (type `?` and a draft to preview it):

```bash
python -m examples.crucible.run --toy=claims --mode=mock --seat=human
```

With a local Ollama model reading the claims and playing the rival:

```bash
python -m examples.crucible.run --toy=claims --mode=ollama \
    --model_name=qwen3:8b --seat=skilled --max_calls=300
```

The relay toy: carry and reword the messages between two parties:

```bash
python -m examples.crucible.run --toy=relay --mode=mock --seat=human
```

The orders toy: command a crew with one standing order:

```bash
python -m examples.crucible.run --toy=orders --mode=mock --seat=human
python -m examples.crucible.run --toy=orders --scenario=free --seat=human  # no goal
```

Orders scenes (the first arc): `one_hand` (isolate), `two_hands` (develop),
`supply_run` (twist), `voyage` (catharsis); `free` has no goal.

Play any toy from a phone: a plain-text page where you sit in the seat
through `HumanActComponent` (the contrib `BlockingInput` reader), see the whole
prompt a model in your seat would read, preview a draft move and send it.
Every game is recorded in the telemetry folder.

```bash
python -m examples.crucible.web.server --port 8901 --base_path /crucible \
    --live=ollama --model_name=qwen3:8b
```

It listens on 127.0.0.1 only, with no authentication: reach it through a
private tunnel, never the open internet.

Does skill show? Bots at 16 seeds:

```bash
python -m examples.crucible.play_audit --toy=claims --seeds=16
```

The orders toy's matrix and golden sets (does a model read orders by the
laws?):

```bash
python -m examples.crucible.audit
python -m examples.crucible.golden_cli --mode=mock
python -m examples.crucible.golden_cli --mode=ollama --model_name=qwen3:8b
```

Each run writes a folder with `log.html` (Concordia's `SimulationLog`),
`summary.json` (outcome and every move's receipt) and `moves.csv`.

## Files

```
examples/crucible/
├── DESIGN.md, ROADMAP.md, README.md
├── run.py               # CLI: play one game, write a run folder
├── play_audit.py        # CLI: bots at many seeds
├── audits.py            # play audit
├── harness.py           # builds a toy as a Concordia Simulation (contrib rulebook GM, seated entities)
├── seats.py             # bots (contrib PolicyInput) and the terminal as HumanInput readers
├── model_setup.py       # mock, Ollama, Together
├── web/                 # the phone page: app (BlockingInput seat, telemetry), server CLI, index.html
├── mock_language_model.py
├── data/claims.yaml     # every number and word of the claims toy
├── data/relay.yaml      # ... and of the relay toy
├── data/orders.yaml     # ... and of the orders toy (Twelve Words): scenes, campaign, laws
├── data/orders_look.yaml   # one glyph, word and colour per kind
├── data/orders_matrix.yaml # the interaction matrices and golden cases
└── toys/
    ├── base.py          # the Toy interface (a contrib Rulebook)
    ├── claims/          # Toy A: world (physics), toy, lexicon, mock, bots, tests
    ├── relay/           # Toy B: toy (ledger and tone), mock, bots, tests
    └── orders/          # Toy C: toy (ship ledger, crew readings), mock, bots, tests
```

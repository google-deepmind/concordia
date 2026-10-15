# Crucible

Grey-box toys for an original game on Concordia, built mechanic first (see
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

Does skill show? Bots at 16 seeds:

```bash
python -m examples.crucible.play_audit --toy=claims --seeds=16
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
├── mock_language_model.py
├── data/claims.yaml     # every number and word of the claims toy
└── toys/
    ├── base.py          # the Toy interface (a contrib Rulebook)
    └── claims/          # Toy A: world (physics), toy, lexicon, mock, bots, tests
```

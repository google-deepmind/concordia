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

Orders scenes, four arcs of isolate / develop / twist / catharsis (the
campaign, in order): `one_hand`, `two_hands`, `supply_run`, `voyage` (orders
and temperaments); `lookout`, `squalls`, `night_watch`, `crossing` (the
watch); `contrary`, `mixed_hands`, `the_mender`, `all_hands` (Dex and "not");
`rota`, `watches`, `no_volunteers`, `long_night` (the rota and "someone");
`the_bosun`, `word_of_mouth`, `contrary_bosun`, `chain_of_command` (the bosun);
`two_orders`, `shifts`, `a_span_is_a_guess`, `ahead_of_time` (timing).
`free` has no goal. A daily challenge rotates through the four catharsis
scenes (`voyage`, `crossing`, `all_hands`, `long_night`) and ends with a line
to share. Scenes can also put members off duty (a rota), let squalls hurt the
most worn, shorten the watch in storms, carry only an order's first clause
through a storm, and pass orders through a bosun.

Play any toy from a phone: a plain-text page where you sit in the seat
through `HumanActComponent` (the contrib `BlockingInput` reader), see the whole
prompt a model in your seat would read, preview a draft move and send it.
Every game is recorded in the telemetry folder.

```bash
python -m examples.crucible.web.server --port 8901 --base_path /crucible
# Offer live crew readers too (Together needs TOGETHER_API_KEY; paid readers
# stop being offered once the spend ledger reaches the budget):
python -m examples.crucible.web.server --port 8901 --base_path /crucible \
    --live=together,ollama --live_budget_usd=3.0
```

It listens on 127.0.0.1 only, with no authentication: reach it through a
private tunnel, never the open internet.

Does skill show? Bots at 16 seeds:

```bash
python -m examples.crucible.play_audit --toy=orders --scenario=rota --seeds=16
```

The orders toy's matrix and golden sets (does a model read orders by the
laws?):

```bash
python -m examples.crucible.audit
python -m examples.crucible.golden_cli --mode=mock --repeats=2
python -m examples.crucible.golden_cli --mode=ollama --model_name=qwen3:8b --repeats=2
```

Each run writes a folder with `log.html` (Concordia's `SimulationLog`),
`summary.json` (outcome and every move's receipt) and `moves.csv`.

## What building it taught

Notes for anyone building a game on Concordia this way (details in
`DESIGN.md`):

- **Let the model judge and the ledger account.** Language models read the
  order for each member; deterministic rule components (`toys/orders/rules.py`)
  apply the consequences. Anything that is arithmetic (prices, who is the
  freshest volunteer) belongs to the ledger: models asked to count got it
  wrong, models told the result read correctly (DESIGN §2k).
- **A mock that reads by the same laws is the CI.** Every prompt the toy
  writes, the mock answers by the written laws, so bots, balance sweeps and
  tests run without a model and stay byte-identical across refactors.
- **Golden sets find holes in your own rules.** Fixed situations with the
  answers the laws allow (`golden.py`, contrib `golden_set`) showed where the
  law text said less than the mock assumed: a clause for someone never said
  it came first, and looked "plain" by the first law's definition (§2r).
- **Run them twice.** Providers vary at temperature 0; hard laws were stable
  while soft judgements moved 74-94% run to run on unchanged prompts.
- **Thinking models need their stops applied to the answer.** With a newline
  terminator a thinking model's open question came back empty (contrib
  Ollama `think` option, §2j); on a CPU-only host one reading can think for
  minutes, so give it a timeout.
- **Make every scene say what it teaches.** Scenes list the matrix cells they
  exercise; the audit reports any rule no scene asks a player to learn
  (§2p).

## Files

```
examples/crucible/
├── DESIGN.md, ROADMAP.md, README.md
├── run.py               # CLI: play one game, write a run folder
├── play_audit.py        # CLI: bots at many seeds
├── audits.py            # play audit
├── audit.py             # CLI: the interaction matrix audit (cells, evidence)
├── golden.py, golden_cli.py # golden sets: does a model read orders by the laws?
├── campaign.py          # chapters gated by stars; the daily challenge
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
    └── orders/          # Toy C, Twelve Words: rules (ledger; command, hazards, crew, watch, rota), toy (readings, view), mock, bots, tests
```

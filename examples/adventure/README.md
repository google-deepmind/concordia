# Adventure

An open-world interactive fiction simulation built on Concordia's
`Asynchronous` engine (`concordia/environment/engines/asynchronous.py`),
`HumanActComponent`, and `TopicBus`.

A human player or synthetic playtester acts inside a persistent world ledger
while NPCs deliberate on background threads, move across locations, and publish
off-screen events to a gossip bus. A zero-LLM `StoryDirector` selects narrative
beats from a declarative `WorldBible`, tracks faction standings and hidden
truths, and coordinates each entity's internal parliament of Sub-Mind voices.

## Quickstart

Run an offline headless playtest with the deterministic mock model (no API key
required):

```bash
python -m examples.adventure.run --mode=playtest --use_mock \
    --bible=bermondsey --steps=12 --output_dir=/tmp/adventure_run
```

Start the browser UI server with Gemini via AI Studio:

```bash
GEMINI_API_KEY=... python -m examples.adventure.run --mode=server \
    --bible=bermondsey --port=8765
```

Run a live playtest with any provider supported by
`concordia.contrib.language_models`:

```bash
python -m examples.adventure.run --mode=playtest \
    --api_type=together_ai \
    --model_name=deepseek-ai/DeepSeek-V3 \
    --bible=glassharbor \
    --steps=16 \
    --output_dir=/tmp/adventure_run
```

## Shipped World Bibles

| `--bible` | Setting |
| --- | --- |
| `bermondsey` | *The Bermondsey Drop* — South London crime caper across a snooker hall, railway arches, a greyhound stadium, and a meat market. |
| `glassharbor` | *Glassharbor* — Steampunk harbor mystery with guild wardens, dockworkers, and pressurized sluice vaults. |
| `vampire_romance` | *The Night Shift at St. Jude's* — Nocturnal hospital drama generated from the `author_world` template library. |
| `dystopian_scifi` | *The Sabotage at Outpost 7* — Off-world habitat thriller generated from the `author_world` template library. |

You can also draft and validate a custom `WorldBible` JSON file:

```bash
python -m examples.adventure.run --mode=author_world \
    --author_title="The Deptford Ledger" \
    --author_tone=noir_mystery \
    --author_output=/tmp/deptford_bible.json
```

## Architecture

* **`world_ledger.py`**: Thread-safe spatial graph, NPC roster, inventory,
  gossip ring buffer, and narrative log.
* **`story_state.py` & `story_director.py`**: Declarative `WorldBible` schema
  (factions, truths, clues, storylets, doom clock, and Sub-Mind voices) and the
  zero-LLM `StoryDirector` that injects scene notes into the GM and
  private Sub-Mind whispers into the player's observation stream.
* **`roster_manager.py`**: Partitions NPCs by graph distance from the player
  into **Hot** (co-located, full async deliberation), **Warm** (1 hop away,
  periodic off-screen actions published to the `TopicBus`), and **Cold** (2+
  hops away, paused via `EntityScheduler` interrupt masks until approached).
* **`adventure_specific_components.py`**: Single-call JSON event resolver
  (`SingleCallResolveComponent`) and procedural frontier architect for
  unauthored exits.
* **`playtester_bot.py` & `evaluate_run.py`**: Five synthetic playtester
  archetypes (`CuriousExplorer`, `ChaosBoundaryTester`, `SocialInvestigator`,
  `DetectiveInvestigator`, `CouncilFollower`) and a two-layer evaluator
  combining deterministic architectural invariants with an 11-dimension
  rubric autorater (`adventure_rubric.md`).
* **`server.py` & `adventure_cli.py`**: HTTP + SSE server for browser play and
  a terminal client that connects to a running server or starts an in-process
  session.

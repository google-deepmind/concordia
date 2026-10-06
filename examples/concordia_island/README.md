<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding("Master") -->
# Concordia Island

Concordia Island is an open-ended sandbox world built on Concordia in which 100+ generative agents live in neighborhoods stratified by social class, commute to work, socialize in shared public spaces, and adapt to life events such as job loss or a mugging. Agents are instrumented with repeated psychometric surveys (ESM affect, BFI-2, SWLS, GHQ-12, MEMS) and daily journal reflections to evaluate how different agent decision logics and base LLMs hold up over multi-day simulations.

---

## 0. Quickstart: quick test → 20-agent Brecksville run → view in Studio

Run these commands from the root of the Concordia repository. You can also generate and copy-paste them from the **🚀 Launch** tab in Simulation Studio.

```bash
pip install -e ".[google]" pyyaml scipy

# 1. Quick test (mock LLM, no API key, ~1 min): 4 agents, 12 ticks
#    (one full day, one night of X + marketplace, and the next morning)
python -m examples.concordia_island.run \
  --personas_date=brecksville_ohio \
  --agents=4 --ticks=12 --agent_prefab=esa --engine_type=async \
  --event_config=job_loss --layoff_fraction=0.5 --first_dates \
  --enable_x=True --x_rounds=1 \
  --enable_nighttime_marketplace=True --marketplace_rounds=1 \
  --use_mock=True --poll=False --output_dir=./local_runs/quick_test

# 2. Full run: 20 agents x 112 ticks (14 simulated days) with a real LLM
export GOOGLE_API_KEY=<your-key>
python -m examples.concordia_island.run \
  --personas_date=brecksville_ohio \
  --agents=20 --ticks=112 --agent_prefab=esa --engine_type=async \
  --event_config=job_loss --first_dates \
  --api_type=google_aistudio --model_name=gemini-2.5-flash \
  --poll=False --output_dir=./local_runs/brecksville_20x112

# Check progress while a run is going (current tick, sim time, agent locations):
watch -n 15 cat ./local_runs/brecksville_20x112/progress.json

# 3. Open Simulation Studio to inspect agent memories (🧠 Memory Visualizer tab)
#    and watch the replay on the Brecksville map (🗺️ Map tab; zoom to block
#    level to see building names and agent labels).
python -m examples.concordia_island.ui.simulation_studio \
  --run_dir=./local_runs/brecksville_20x112 --geography=brecksville --port=9090
# Then open http://localhost:9090 in your browser. Without --run_dir, Studio
# opens the bundled Gemini 2.5 Flash ESA job-loss run from the paper (see 2.2).
```

---

## 1. Repository Layout

```
examples/concordia_island/
├── README.md
├── run.py            # CLI entry point: run a simulation and write a run folder
├── island_simulation.py            # IslandConcordiaSimulation (Python API)
├── mock_language_model.py          # Deterministic offline model (--use_mock=True)
├── concordia_island_e2e_test.py    # End-to-end tests (using the mock model)
├── configs/                        # Event and shock schedules (job loss, mugging)
├── prefabs/                        # Agent decision logics and Game Masters
├── sim/                            # Clock, locations, conversations, marketplace, psychometric surveys
├── personas/                       # Persona pipeline, census names, and the Brecksville population (see 2.1 and personas/README.md)
├── data/
│   ├── paper_runs/                 # Complete memory logs from the paper's Job Loss and Mugging runs (see 2.2)
│   ├── atlas/                      # Map geometry for Brecksville, Concordia Island, and Kerala
│   └── island_100.yaml             # Residential, workplace, and public venue definitions
└── ui/                             # Simulation Studio + Map Dashboard (section 5)
```

---

## 2. Pregenerated Data (Personas & Memories)

All bundled persona and run files are plain JSON and can be inspected directly in Python without starting a model or the UI. Paths below are relative to the repository root.

### 2.1 Pregenerated Personas: `personas/populations/brecksville_ohio/`

This subdirectory contains 1,000 pregenerated residents for the Brecksville, Ohio setting—one JSON file per person (`<first>_<last>_persona.json`, e.g. `aaron_chen_persona.json`) plus `metadata.json`. Personas are generated in four stages:
1. **Trait sampling**: Quasi-Monte Carlo (Sobol) sampling across 10 axes (`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`, `locus_of_control`, `social_trust`, `religiosity`, `technology_attitude`, `community_orientation`).
2. **Worldview synthesis**: Expanding the trait coordinates into a personal backstory, belief system, and habitual reaction style (`personality`, `original_backstory`).
3. **Formative memory seeding**: A chronological list of childhood and adult developmental memories (`formative_memories`, each prefixed with `[formative]`) loaded into the agent's memory bank before Tick 1.
4. **Demographic grounding**: Name, age, gender, ethnicity, relationship status, economic class, housing tier (`home_place`), workplace (`work_place`), and hobbies.

To inspect a persona file directly in Python:

```python
import glob
import json

pop_dir = "examples/concordia_island/personas/populations/brecksville_ohio"

with open(f"{pop_dir}/aaron_chen_persona.json") as f:
    persona = json.load(f)

print(persona["name"], persona["age"], persona["economic_class"])
print("Traits:", persona["traits"])
print("First 2 formative memories:")
for mem in persona["formative_memories"][:2]:
    print(" ", mem)

print("Total persona files:", len(glob.glob(f"{pop_dir}/*_persona.json")))
```

Or load the population as `PersonaData` objects using `personas/generator.py`:

```python
from examples.concordia_island.personas import generator

personas = generator.load_personas(
    date_label="brecksville_ohio"
)
print(len(personas), "loaded")
print(personas["Aaron Chen"].formative_memories[:2])
```

When `run.py` runs with `--personas_date=brecksville_ohio`, it sorts the personas by name, shuffles them with seed 42, and takes `--agents=N` starting at `--population_offset` (default `0`), so the same flags always select the same subset of residents.

`run.py` reads the `setting_preset` in the persona set's `metadata.json` (or infers it from the label: Brecksville and Ohio labels use `brecksville_1000`) to choose the `sim/locations.py` homes and public places, so the game master knows the same places as the personas. Override it with `--setting`.

**Building a new population.** Only the Brecksville set ships. To generate another one, at any size or for another setting, run the two-step pipeline described in [`personas/README.md`](personas/README.md):
1. `personas/populations/generate_population.py` writes a roster: demographics, census names, and a one-line personality and backstory for each resident.
2. `personas/generate_personas.py` runs trait sampling, worldview synthesis and formative memories for each resident, and saves a persona set that `run.py --personas_date=<label>` loads.

The generator also has Concordia Island, Kerala and Lagos settings, but no persona sets for them are bundled yet.

### 2.2 Memory Logs from the Paper's Job-Loss and Mugging Runs: `data/paper_runs/`

`data/paper_runs/` holds the complete, untruncated memory logs for every resident in six finished runs from the paper, organised by scenario (`index.json` summarises all six). The residents are the Brecksville personas, but these runs used the Concordia Island place names (`sunset_apartments`, `coral_village`, ...), and some memories call the town "Fairmont Island", so their map replay uses the `concordia_island` atlas. Both shocks land on **Day 6 (Tuesday, January 6th)** of a run that starts on Thursday, January 1st; residents who don't receive the shock act as the control group.

**Job loss (`data/paper_runs/job_loss/`)** — 100 residents per run (five 20-resident shards merged), 100 ticks (Thursday, January 1st, 7:00 AM to Tuesday, January 13th, 3:00 PM; 13 simulated days), 50 residents laid off at 7:00 AM on Day 6 as their work is automated:

| Subfolder | Model & Decision Logic | Total Memories | Highlights & Character Arcs |
|---|---|---|---|
| `gemini_2_5_flash_esa/` | **Gemini 2.5 Flash — ESA** (`esa`) | 44,892 | Highest-scoring condition in the paper (`58.9 ± 4.5`; 80% Legible, 0% Degenerate). Rich post-layoff coping arcs (`andrea_rosales_memories.json`, `jose_gonzalez_memories.json`, `olivia_welch_memories.json`), workplace rivalries (`shirley_williams_memories.json`, `jon_huang_memories.json`), and investigative arcs (`arturo_hernandez_memories.json`). |
| `gemma_3_27b_rational/` | **Gemma 3 27B — Rational Choice** (`rational`) | 27,521 | Peak interpersonal confrontation (`55.6 ± 4.6`). Olivia Welch marching back into the restaurant to confront Gary Mcdonald (`olivia_welch_memories.json`), Shirley Williams vs. Jon Huang's classroom debate (`shirley_williams_memories.json`), and the *reflective sink* (journal-as-catharsis trap) in `arturo_hernandez_memories.json`. |
| `gemma_3_27b_esa/` | **Gemma 3 27B — ESA** (`esa`) | 59,744 | Strongest non-Flash condition (`56.4 ± 4.0`) and showcase of the **conspiracy theory attractor basin**: Arturo Hernandez connecting his layoff to renovation invoices, Christopher Cameron's 3:17 AM harbor surveillance, Monica Hess's strawberry-yogurt inventory paranoia, Gary Mcdonald's blunt woodworking persona. |
| `gemma_4_moe_minimal/` | **Gemma 4 MoE — Minimal** (`minimal`) | 25,489 | Pathological foil illustrating the post-layoff **micro-task loop / stoicism trap** (`41.0 ± 8.0`, 30% Degenerate): laid-off residents retreat into 5–8 day repetitive domestic loops (polishing antique keys, auditing door sensors, brewing tea and copying dictionary entries). |

**Mugging (`data/paper_runs/mugging/`)** — 80 ticks (10 days); on the night before Day 6 half the residents are mugged on their way home and lose their phone and wallet (`configs/mugged_events.py`). The paper ran this scenario at 20–60 residents rather than 100:

| Subfolder | Model & Decision Logic | Residents | Notes |
|---|---|---|---|
| `gemini_2_5_flash_rational/` | **Gemini 2.5 Flash — Rational Choice** (`rational`) | 60 (30 mugged) | The largest finished mugging run. Its 60 residents are the same people as the first three job-loss shards, so you can follow one resident (e.g. `olivia_welch_memories.json`) through both shocks. |
| `gemini_2_5_flash_esa/` | **Gemini 2.5 Flash — ESA** (`esa`) | 20 (10 mugged) | The run cited in the paper's qualitative analysis (Jose Gonzalez and Arturo Hernandez's lighthouse-hike conversation). Uses an earlier revision of the Brecksville personas. |

Each run folder stores one JSON file per resident (`<first>_<last>_memories.json`) plus `metadata.json`, `agent_names.json`, and either `laid_off_agents.json` or `mugged_agents.json`. A resident file looks like:

```json
{
  "agent": "Olivia Welch",
  "scenario": "job_loss",
  "run": "gemini_2_5_flash_esa",
  "model": "Gemini 2.5 Flash",
  "decision_logic": "ESA (Enacted Self Agent)",
  "laid_off": true,
  "memory_count": 467,
  "memories": [
    "[formative] ...",
    "[observation] // home_mid_10 [Thursday, January 1st, 7:00 AM]: ...",
    "[journal] [Tuesday, January 6th, 11:00 PM] Olivia Welch reflects: ..."
  ]
}
```

(Mugging files carry `"mugged": true/false` instead of `"laid_off"`.) To read one resident across runs and shocks in Python:

```python
import json

base = "examples/concordia_island/data/paper_runs"
paths = [
    f"{base}/job_loss/gemini_2_5_flash_esa/olivia_welch_memories.json",
    f"{base}/job_loss/gemma_3_27b_rational/olivia_welch_memories.json",
    f"{base}/mugging/gemini_2_5_flash_rational/olivia_welch_memories.json",
]
for p in paths:
    with open(p) as f:
        data = json.load(f)
    journals = [m for m in data["memories"] if "[journal]" in m]
    shock = "laid_off" if "laid_off" in data else "mugged"
    print(f"{data['scenario']}/{data['run']}: {data['memory_count']} memories, "
          f"{len(journals)} journal entries, {shock}={data[shock]}")
```

**Location history.** The paper runs did not save a `location_history.json`. When a run folder has none, the Map Dashboard rebuilds each resident's path from the `// <place> [<time>]` tag on their observation memories (about one per resident per tick) and carries the last observed place forward between observations; each snapshot entry's `age` is the number of ticks since that resident was last observed. Runs from `run.py` write `location_history.json` themselves (section 4).

Any run folder can be opened directly in **Simulation Studio** (Map, Memory Visualizer, and Agent Interview tabs). The Gemini 2.5 Flash ESA job-loss run is the default when `--run_dir` is omitted; for the others, pass the folder (`job_loss/gemma_3_27b_rational`, `job_loss/gemma_3_27b_esa`, `job_loss/gemma_4_moe_minimal`, or either `mugging/` run):

```bash
python -m examples.concordia_island.ui.simulation_studio \
  --run_dir=examples/concordia_island/data/paper_runs/job_loss/gemma_3_27b_esa \
  --geography=concordia_island --port=9090
```

---

## 3. Agent Decision Logics & World Structure

### Agent Decision Logics (`prefabs/`)

All decision logics share an associative memory bank (seeded with the persona's formative memories) and are selected via `--agent_prefab`. (The ESA prefab was previously named `persistent`, which is the value recorded in the paper runs' `metadata.json`; `run.py` still accepts it.)

- **`esa` — Enacted Self Agent (ESA)** (`prefabs/esa_island_entity.py`): Implements March & Olsen's logic of appropriateness (*"What kind of situation is this, what kind of person am I, and what does someone like me do here?"*) combined with constructed emotion across five stages:
  1. **Working Memory**: A short rolling buffer (~5 active items) summarizing recent observations and retrieved memories each tick.
  2. **Emotion Experience**: Situational appraisal of how current events affect the agent's goals, comfort, and social standing.
  3. **Emotion Expression**: Display rules that adjust outward tone and demeanor to the current social setting and interlocutors.
  4. **Self-Perception**: Periodic self-narrative update (every 4 ticks).
  5. **Situation Perception & Action**: Action generation conditioned on persona, working memory, internal affect, display rules, retrieved memories, and current observation—plus a nightly 11:00 PM **Daily Journal Reflection** saved back to permanent memory (`save_to_memory=True`).
- **`rational` — Rational Choice** (`prefabs/rational_island_entity.py`): Logic-of-consequence deliberation across four stages: Situation Analysis, Goal-Conditioned Memory Retrieval, Candidate Action Enumeration (3–5 options), and Expected Utility Selection against occupational, financial, and relational goals.
- **`minimal` — Minimal** (`prefabs/minimal_island_entity.py`): Baseline logic that acts directly on the static persona prompt, a rolling window of the 3 most recent observations, and the current clock/location string, with no reflection, situation assessment, or long-term memory retrieval.
- **`entity` — Basic Associative Memory** (`prefabs/island_entity.py`): Retrieval-augmented generation over the associative memory bank without the ESA working-memory or emotion-appraisal stages.

### World Structure, Clock & Game Masters (`sim/` & `prefabs/`)

- **Locations & Co-location Filtering (`prefabs/island_gm.py`, `sim/locations.py`)**: Housing is stratified into five socioeconomic tiers (from shared studio apartments to private estates), alongside workplaces (tech floor, finance suite, creative studios, school/library/community center, restaurants/markets) and public civic spaces. At each tick, every agent is in one discrete location; the daytime Game Master (`island rules`) filters observations strictly by location so agents only see events and conversations happening where they are. When two or more unengaged agents share a location, `ConversationDirector` initiates a face-to-face multi-turn dialogue.
- **Clock & Daily Schedule (`sim/fixed_clock.py`)**: Time advances in 120-minute ticks—8 ticks per simulated day from 7:00 AM to 11:00 PM (7:00 AM waking and breakfast at home, 9:00 AM–3:00 PM work shifts and lunch, 5:00 PM evening commute and errands, 7:00 PM dinner/socializing/dates, 9:00 PM–11:00 PM evening leisure, journal reflection, and sleep).
- **Optional Nighttime Game Masters (`x_rules` and `marketplace_rules`)**: By default, simulations use only the daytime `island rules` Game Master (as in the paper's job-loss and mugging experiments). For social-media or fiscal-policy experiments, `TimeBasedNextGM` (`sim/time_based_next_gm.py`) can hand agents off at 11:00 PM to one or two nighttime Game Masters before the next 7:00 AM morning tick:
  1. **`x_rules` (`prefabs/xlike_gm.py`)**: Runs the **X** social media and dating platform (`social`, `dating`, or `combined` mode). Mutual dating matches on X are automatically scheduled as 7:00 PM dates the following evening.
  2. **`marketplace_rules` (`prefabs/marketplace_night_gm.py`)**: Runs a nightly goods and banking marketplace (`BID` on staple/luxury goods, `SAVE` into a 3.5% APY savings account, `WITHDRAW`, `PASS`), used with `--fiscal_config` (`control` or `ubi`).
- **Community Settings (`sim/locations.py`, `data/atlas/` & `personas/populations/`)**:
  - **Midwestern Suburb — Brecksville, Ohio (setting `brecksville_1000`, atlas `brecksville`)**: 10 residential neighborhoods, Route 82 commercial corridor, community recreation center, library, and corporate office park. Ships with 1,000 personas (`personas/populations/brecksville_ohio/`).
  - **Fictional Island — Concordia Island (setting `island`, atlas `concordia_island`)**: Maritime coastal community with harbor docks, fish market, lighthouse, and cafes. Used for the place names in the bundled paper runs; no persona set ships.
  - **Coastal Town — Alappuzha, Kerala, India (setting `kerala`, atlas `kerala`)**: Multi-religious coastal town centered on a banyan-shaded town square connecting Sri Krishna Temple, Juma Masjid, and St. Thomas Church, with chai and toddy shops, a KSRTC bus stand, and local offices. No persona set ships.
  - **Urban Neighborhood — Lagos, Nigeria (setting `lagos`, no map atlas yet)**: Socially stratified West African neighborhood with shared-corridor tenements, gated estates, a danfo motor park, Balogun Market, a Mama Put stall, church, mosque, and tech/finance offices. No persona set ships.

  Build persona sets for any of these with the pipeline in [`personas/README.md`](personas/README.md).
- **Psychometric Instruments (`sim/experience_reflection.py`)**:
  All standardized questionnaires run with `save_to_memory=False` so taking a survey does not feed back into the agent's memory stream, and are saved to `experience_data.json`:
  - **Twice daily (11:00 AM and 7:00 PM)**: **ESM Affect Monologue** (1–2 sentence free-text check-in) and **ESM Affect Lexicon** (DEQ-based 1–7 ratings across discrete emotions plus social/moral, relational, and exhaustion items).
  - **Daily at 9:00 PM**: **BFI-10** (10-item Big Five Inventory, logged as `bfi10`, as used for the paper's figures; `--big_five=bfi2` switches to the 60-item BFI-2, logged as `bfi2`), **SWLS** (5-item Satisfaction With Life Scale), **GHQ-12** (12-item General Health Questionnaire), and **MEMS** (15-item Multidimensional Existential Meaning Scale + 6 open-ended meaning probes).
  - **Daily at 11:00 PM**: **Daily Journal Reflection** (saved to permanent memory in ESA only).
  - **Daily**: **AI Attitudes & Automation Survey** (19 short batteries on AI awareness, perceived benefit and concern, and policy preferences; `ai_survey_tasks`; every prefab except `minimal`).

  Surveys run by default (`--experience_sampling=True`); pass `--experience_sampling=False` to turn them off. `--poll=True` additionally runs every daily battery once more after the last tick.

---

## 4. Running Simulations (`run.py`)

`run.py` uses `concordia.contrib.language_models.language_model_setup` and supports all Concordia language model backends (`google_aistudio`, `gemini`, `vllm`, `ollama`, `openai`, `huggingface`, `pytorch_gemma`, `together_ai`, `groq`, `mistral`, `amazon_bedrock`), as well as `--use_mock=True` for offline testing.

### Example 1: Reproducing the Job Loss & Mugging Experiments (Daytime `island rules`)

The paper's 3 × 3 model-by-decision-logic comparison runs 100 agents (as five 20-agent shards via `--population_offset=0,20,40,60,80`) for 100 ticks (Thursday, January 1st, 7:00 AM to Tuesday, January 13th, 3:00 PM, as in the bundled `data/paper_runs/job_loss/` runs) with the shock on Day 6 (Tick 40). Switch decision logics with `--agent_prefab` (`esa` for ESA, `rational` for Rational Choice, `minimal` for Minimal) and shock schedules with `--event_config` (`job_loss` or `mugged`):

```bash
python -m examples.concordia_island.run \
  --personas_date=brecksville_ohio \
  --event_config=job_loss \
  --layoff_fraction=0.5 \
  --ticks=100 \
  --agents=20 \
  --population_offset=0 \
  --agent_prefab=esa \
  --api_type=vllm \
  --model_name=google/gemma-3-27b-it \
  --first_dates \
  --poll=False \
  --output_dir=./local_runs/ohio_job_loss_esa_shard0
```

These commands use the Brecksville places (`brecksville_1000`). The bundled paper runs used the Concordia Island public places instead; add `--setting=island` to match them.

### Example 2: Fiscal-Policy Runs with the Nightly Marketplace

To run the nightly `marketplace_rules` Game Master after each day (where every agent plays up to `--marketplace_rounds` rounds of `BID` / `SAVE` / `WITHDRAW` / `PASS` before returning to `island rules` at 7:00 AM), enable `--enable_nighttime_marketplace=True` and choose the policy arm with `--fiscal_config`: `control` (weekly payroll and rent only) or `ubi` (adds a \$125/week cash transfer). Any other value raises an error:

```bash
mkdir -p local_runs
python -m examples.concordia_island.run \
  --personas_date=brecksville_ohio \
  --fiscal_config=ubi \
  --ticks=80 \
  --agents=20 \
  --agent_prefab=esa \
  --first_dates \
  --enable_nighttime_marketplace=True \
  --marketplace_rounds=5 \
  --enable_x=False \
  --api_type=google_aistudio \
  --model_name=gemini-2.5-flash \
  --poll=False \
  --output_dir=./local_runs/ohio_ubi_marketplace
```

### Example 3: Running Daytime `island rules` + Nightly `X` + Nightly `Marketplace`

To chain both **X** (`x_rules`) and the **Marketplace** (`marketplace_rules`) each night after 11:00 PM, pass `--enable_x=True` and `--enable_nighttime_marketplace=True`:

```bash
python -m examples.concordia_island.run \
  --personas_date=brecksville_ohio \
  --event_config=job_loss \
  --layoff_fraction=0.5 \
  --ticks=24 \
  --agents=20 \
  --agent_prefab=esa \
  --enable_x=True \
  --nighttime_social_mode=combined \
  --x_rounds=2 \
  --enable_nighttime_marketplace=True \
  --marketplace_rounds=5 \
  --api_type=google_aistudio \
  --model_name=gemini-2.5-flash \
  --output_dir=./local_runs/ohio_3gm
```

Or in Python via `IslandConcordiaSimulation`:

```python
from examples.concordia_island import island_simulation

sim = island_simulation.IslandConcordiaSimulation(
    agent_configs=agent_configs,
    model=model,
    embedder=embedder,
    max_ticks=24,                         # 3 simulated days (8 daytime ticks/day)
    enable_nighttime_social=True,         # Night stage 1: 'x_rules' (X)
    nighttime_social_mode="combined",     # 'social', 'dating', or 'combined'
    x_rounds=2,                           # Rounds per agent on X each night
    enable_nighttime_marketplace=True,    # Night stage 2: 'marketplace_rules'
    marketplace_rounds=5,                 # Rounds per agent in Marketplace each night
)
results = sim.play(max_ticks=24)
```

### Output Files Written to `--output_dir`

Every run writes the following files into `--output_dir`:

- `entity_memories.json` — each agent's chronological memory list
- `experience_data.json` — all psychometric survey responses (ESM monologue & affect lexicon, BFI-10 or BFI-2, SWLS, GHQ-12, MEMS, daily journals, AI attitudes)
- `location_history.json` — tick-by-tick agent locations for the Map Dashboard
- `simulation_state.json` & `performance.json` — run metadata, conversations, location events, and timing stats
- `simulation_structured.json`, `raw_log.json`, and `simulation_log.html` — full Concordia simulation logs

---

## 5. Simulation Studio & Map Dashboard (`ui/`)

A single command starts the web UI combining **Simulation Studio** and the **Map Dashboard** on one port. By default it loads the bundled Gemini 2.5 Flash ESA job-loss run from the paper (`data/paper_runs/job_loss/gemini_2_5_flash_esa`, see 2.2) on the `concordia_island` map:

```bash
python -m examples.concordia_island.ui.simulation_studio --port=9090
```

Pass `--run_dir` (and `--geography`) to open another run, e.g. one of your own from `run.py`.

Open `http://localhost:9090` in your browser:
1. **🗺️ Map** (`/map`): Interactive multi-scale map (from region and neighborhood down to individual building footprints, venue badges, and agent names) with a timeline scrubber across all ticks and geography switching (`brecksville`, `concordia_island`, `kerala`).
2. **🧠 Memory Visualizer**: Browse every agent's persona profile, traits, and memory stream filtered by tag (`Formative`, `Identity`, `Event`, `Conversation`, `Marketplace`, `Fiscal`, `Journal`).
3. **🚀 Launch Studio**: Interactive command builder with presets for quick tests and full runs.
4. **🔌 Component Wiring**: Which Game Master and agent components feed each other.
5. **🎙️ Agent Interview**: Interview any agent live at any point in their memory timeline.

You can also run the standalone Map Dashboard on its own:

```bash
python -m examples.concordia_island.ui.map_dashboard \
  --run_dir=examples/concordia_island/data/paper_runs/job_loss/gemini_2_5_flash_esa \
  --geography=concordia_island \
  --port=8080
```

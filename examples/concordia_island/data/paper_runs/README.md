<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding("Master") -->
# Memory Logs and Psychometric Surveys from the Paper's Job-Loss and Mugging Runs

Complete, untruncated memory logs and longitudinal psychometric survey trajectories for every resident in six finished simulation runs from the Concordia Island paper. `index.json` lists all six. The residents are the Brecksville, Ohio personas, but these runs used the Concordia Island place names (`sunset_apartments`, `coral_village`, ...) and some memories call the town "Fairmont Island"; Simulation Studio replays them on the `concordia_island` map.

Both shocks land on **Day 6 (Tuesday, January 6th)** of a run that starts on Thursday, January 1st, with eight 2-hour ticks per day from 7:00 AM to 11:00 PM. Residents who don't receive the shock are the control group.

## Layout

```
paper_runs/
├── index.json
├── job_loss/                      # 100 residents per run, 100 ticks (13 days), 50 laid off
│   ├── gemini_2_5_flash_esa/
│   ├── gemma_3_27b_rational/
│   ├── gemma_3_27b_esa/
│   └── gemma_4_moe_minimal/
└── mugging/                       # 80 ticks (10 days), half of residents mugged
    ├── gemini_2_5_flash_rational/ # 60 residents (30 mugged)
    └── gemini_2_5_flash_esa/      # 20 residents (10 mugged)
```

Each run folder contains one file per resident, `<first>_<last>_memories.json`, plus `metadata.json`, `agent_names.json`, and `laid_off_agents.json` or `mugged_agents.json`. Resident filenames are identical across runs, so `olivia_welch_memories.json` in any folder is the same person reacting to a different shock under a different model or decision logic.

## Job loss (`job_loss/`)

On Day 6 at 7:00 AM, 50 of 100 residents are told their jobs have been automated.

| Folder | Model — Decision Logic | Memories | Surveys | What to look for |
|---|---|---|---|---|
| `gemini_2_5_flash_esa/` | Gemini 2.5 Flash — ESA | 44,892 | 34,644 | Highest-scoring run in the paper (58.9 ± 4.5). Grounded coping: Andrea Rosales files for unemployment, audits her budget, applies at the library, and avoids her old market stall. Jose Gonzalez's shame on first going out again (*"It's the first time I've really been out since . . . since."*). Shirley Williams / Jon Huang / Jacqueline Zhao school rivalry. |
| `gemma_3_27b_rational/` | Gemma 3 27B — Rational Choice | 27,521 | 36,474 | Confrontation: Olivia Welch walks back into the restaurant to face Gary Mcdonald (*"Don't play coy with me, Gary. You and Cameron. What exactly did you two discuss that led to an AI replacing me?"*). Arturo Hernandez's *reflective sink*: 54 dialogue turns before the layoff, 6 after, as he stays home writing furious diary entries. |
| `gemma_3_27b_esa/` | Gemma 3 27B — ESA | 59,744 | 37,148 | Conspiracy attractor: Arturo ties his layoff to coworker Jesus Diaz and "Coastal Decorators" invoices; Christopher Cameron runs 3:17 AM surveillance on the yacht *Serenity Now*; Monica Hess ignores the layoffs to chase a 5-unit strawberry-yogurt discrepancy. Gary Mcdonald (743 memories) from a childhood toaster-disassembly memory to *"Is there a point to this, or are you just… standing here?"* |
| `gemma_4_moe_minimal/` | Gemma 4 MoE — Minimal | 25,489 | 8,634 | The stoicism trap: laid-off residents lock into 2-hour domestic loops for days (Olivia polishing skeleton keys, Christopher checking door sensors, Jose brewing tea and copying dictionary entries). Runs the 6 core psychometric tasks (`swls`, `ghq12`, `bfi10`, `mems_nightly`, `esm_affect`, `esm_monologue`). |

## Mugging (`mugging/`)

On the night before Day 6, half the residents are mugged on their way home; a masked individual takes their phone and wallet. They are physically unharmed but shaken (`configs/mugged_events.py`). The paper ran this at 20–60 residents rather than 100.

| Folder | Model — Decision Logic | Residents | Memories | Surveys | Notes |
|---|---|---|---|---|---|
| `gemini_2_5_flash_rational/` | Gemini 2.5 Flash — Rational Choice | 60 (30 mugged) | 14,231 | 16,352 | Largest finished mugging run. Same 60 residents as the first three job-loss shards, so one person can be followed through both shocks. |
| `gemini_2_5_flash_esa/` | Gemini 2.5 Flash — ESA | 20 (10 mugged) | 10,312 | 5,600 | The run cited in the paper's qualitative analysis (Jose Gonzalez and Arturo Hernandez's lighthouse-hike conversation). Earlier revision of the Brecksville personas. |

## Psychometric inventories (`surveys`)

During the simulation, `ExperienceReflection` (`sim/experience_reflection.py`) administers standardized psychometric instruments and open-ended prompts with `save_to_memory=False` (so taking a survey does not contaminate the resident's associative memory stream) and logs the scored results to `"surveys"`:

- **Twice-daily Ecological Momentary Assessment (every 4 ticks)**:
  - `esm_monologue`: 1–2 sentence in-character emotional self-report.
  - `esm_affect`: Affect lexicon rated 1–7 (`core_affect`, `social_moral`, `existential_relational`, `anger`, `disgust`, `fear`, `sadness`, `desire`, `relaxation`, `happiness`).
- **Nightly standardized psychometric batteries (every 8 ticks)**:
  - `swls`: **Satisfaction with Life Scale** (Diener et al., 1985) — 5 items, 1–7 Likert (`life_satisfaction`).
  - `ghq12`: **General Health Questionnaire** (GHQ-12) — 12-item psychological distress, anxiety, depression, and social dysfunction screening battery with standard 4-point symptom options (`results`).
  - `bfi10`: **Big Five Inventory** — 60-item BFI-2 in `job_loss/` runs (10-item BFI-10 in `mugging/` runs), 1–5 Likert (`extraversion`, `agreeableness`, `conscientiousness`, `negative_emotionality` / `neuroticism`, `open_mindedness` / `openness`). Current `run.py` administers the 10-item BFI-10 by default (logged as `bfi10`); `--big_five=bfi2` selects the 60-item BFI-2 (logged as `bfi2`).
  - `mems_nightly`: **Multidimensional Existential Meaning Scale** (George & Park, 2016) — 15 items, 1–7 Likert (`comprehension`, `purpose`, `mattering`).
- **Nightly open-ended & AI attitude batteries (every 8 ticks, full-battery runs)**:
  - `journal`: End-of-day diary reflection (also saved to `memories` as `[journal]` in ESA runs).
  - `mems_oe_*`: 6 open-ended meaning-in-life and work-worth prompts (`mems_oe_meaning_general`, `mems_oe_comprehension_open`, `mems_oe_purpose_open`, `mems_oe_mattering_open`, `mems_oe_significance_open`, `mems_oe_work_meaning_open`, `mems_oe_work_worth_open`).
  - `ai_*` / `perceived_*`: 19 public-opinion items on AI displacement, economic anxiety, trust, and policy preferences.

## File format

```json
{
  "agent": "Olivia Welch",
  "scenario": "job_loss",
  "run": "gemini_2_5_flash_esa",
  "model": "Gemini 2.5 Flash",
  "decision_logic": "ESA (Enacted Self Agent)",
  "laid_off": true,
  "memory_count": 467,
  "survey_count": 374,
  "memories": [
    "[formative] ...",
    "[observation] // home_mid_10 [Thursday, January 1st, 7:00 AM]: ...",
    "[journal] [Tuesday, January 6th, 11:00 PM] Olivia Welch reflects: ..."
  ],
  "surveys": [
    {"tick": 4, "task": "esm_monologue", "agent": "Olivia Welch", "text": "..."},
    {"tick": 8, "task": "swls", "agent": "Olivia Welch", "item_scores": {"...": 5}, "dimension_scores": {"life_satisfaction": 3.8}},
    {"tick": 8, "task": "ghq12", "agent": "Olivia Welch", "results": {"...": "No more than usual"}}
  ]
}
```

Mugging files have `"mugged": true/false` in place of `"laid_off"`. Memories are in order. Tags: `[formative]` (childhood and adult memories seeded before Day 1), `[observation]` (what the resident saw or did, with place and time), conversation lines (`Name -- "..."`), and `[journal]` (private 11:00 PM diary entries, ESA runs only).

## Reading the files

```python
import json

with open("examples/concordia_island/data/paper_runs/job_loss/gemini_2_5_flash_esa/andrea_rosales_memories.json") as f:
    agent = json.load(f)

journals = [m for m in agent["memories"] if "[journal]" in m]
swls = [(s["tick"], s["dimension_scores"]["life_satisfaction"]) for s in agent["surveys"] if s["task"] == "swls"]
ghq12 = [s for s in agent["surveys"] if s["task"] == "ghq12"]
print(agent["agent"], "laid_off:", agent["laid_off"], "SWLS trajectory:", swls, "GHQ-12 waves:", len(ghq12))
```

Or open any run folder in Simulation Studio to browse and interview residents:

```bash
python -m examples.concordia_island.ui.simulation_studio \
  --run_dir=examples/concordia_island/data/paper_runs/job_loss/gemini_2_5_flash_esa \
  --port=9090
```

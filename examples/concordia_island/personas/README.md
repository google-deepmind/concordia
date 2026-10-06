# Persona Pipeline

How the bundled Brecksville population was built, and how to build a new one.
Commands run from the repository root.

```
generate_population.py  ->  roster JSON  ->  generate_personas.py  ->  persona set  ->  run.py
   (step 1)                                    (step 2)                 (directory)
```

## Step 1: Roster (`populations/generate_population.py`)

Builds one entry per resident:

-   **Demographics** come from a seeded RNG: economic tier (sets `home_place`,
    e.g. `millbrook_apts_unit_12`), workplace, age, gender, sexual
    orientation, ethnicity, relationship status, political orientation,
    hobbies and, for Brecksville, neighborhood.
-   **Names** come from census name tables (`census_names.py`). They are also
    seeded and always unique.
-   **Personality and backstory**: the LLM writes a one-line personality and
    a one-sentence backstory, in batches of 20.
-   **Couples** are paired and share a home.

```bash
export GOOGLE_API_KEY=<your-key>
python -m examples.concordia_island.personas.populations.generate_population \
  --setting=brecksville --num_agents=1000 --seed=42 \
  --api_type=google_aistudio --model_name=gemini-2.5-flash \
  --output_path=/tmp/brecksville_roster.json
```

The `--setting` options are listed below.

| `--setting`                     | Community               | Sim preset (`sim/locations.py`) | Map atlas          |
| ------------------------------- | ----------------------- | ------------------------------- | ------------------ |
| `brecksville` (default)         | Brecksville, Ohio       | `brecksville_1000`              | `brecksville`      |
| `ohio_suburb`                   | Same as `brecksville`   | `brecksville_1000`              | `brecksville`      |
| `island`                        | Concordia Island        | `island`                        | `concordia_island` |
| `kerala`                        | Alappuzha, Kerala       | `kerala`                        | `kerala`           |
| `lagos`                         | Lagos, Nigeria          | `lagos`                         | none yet           |

If a batch's LLM output can't be parsed after `--max_retries` attempts, the
script stops with an error. It never writes placeholder text. The same applies
to duplicate names.

`--dry_run` skips the LLM and writes only demographics and names. Use it to
check tier sizes and names. Step 2 rejects dry-run rosters.

## Step 2: Persona set (`generate_personas.py`)

Runs `PersonaGenerator` (`generator.py`) on every roster entry:

1.  **Trait sampling**: Sobol (quasi-Monte Carlo) points across 10 axes: Big
    Five, locus of control, social trust, religiosity, technology attitude
    and community orientation.
2.  **Worldview synthesis**: a three-call LLM chain (worldview → situation →
    reaction) that turns the trait point into a worldview memory.
3.  **Formative memories**: chronological memories from childhood to the
    present. They are loaded into the agent's memory before tick 1.
4.  **Assembly and save**: one `<first>_<last>_persona.json` per resident,
    plus `metadata.json`, which records `setting_preset`.

```bash
python -m examples.concordia_island.personas.generate_personas \
  --input_roster=/tmp/brecksville_roster.json \
  --date_label=my_brecksville \
  --api_type=google_aistudio --model_name=gemini-2.5-flash --workers=16
```

The set is written to `personas/populations/<date_label>/` (override with
`--output_dir`). The script refuses to overwrite a non-empty directory. If any
resident fails a stage, it raises and saves nothing.

## Step 3: Run it

```bash
python -m examples.concordia_island.run --personas_date=my_brecksville \
  --agents=20 --ticks=16 --api_type=google_aistudio \
  --model_name=gemini-2.5-flash --output_dir=./local_runs/my_brecksville
```

If you used `--output_dir` in step 2, add `--personas_dir=<that dir>`.
`run.py` reads `setting_preset` from `metadata.json` to choose homes and public
places; `--setting` overrides it.

## Bundled set

`populations/brecksville_ohio/` holds 1,000 Brecksville residents built with
the same two steps (`--setting=ohio_suburb`). The formative memories of a few
residents were regenerated afterwards; their names are listed under
`formative_memories_regenerated` in `metadata.json`. That file predates the
`setting_preset` key, so `run.py` infers `brecksville_1000` from the label.

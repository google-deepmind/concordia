# Lifeways

SimLife as artificial social science. A small 2D world where people are born,
work, find partners, raise children and die, generation after generation, for
as long as you let it run. What evolves is culture. A person's **customs** are
components of that person's own Concordia entity. Like plasmids, they are
inherited at birth (copied, or recombined from two parents, domain by domain),
offered by others, taken up or dropped, and mutated as they are copied.

## Quickstart

Offline, with the deterministic mock model:

```bash
python -m examples.lifeways.run --years=200 --soak
```

The run writes these files to `--out_dir`:

| File | Contents |
|---|---|
| `census.jsonl` | a row a year: population, births, deaths, mean age, customs recorded, food |
| `dead.jsonl` | each person who died: their entity's final state and params |
| `summary.json` | the run's totals, with memory use and seconds per year (`--soak`) |

## How it fits Concordia

It runs on the stock `prefabs/simulation/generic.Simulation` and the stock
`Asynchronous` engine: every person has an observe-act loop of its own.

**People** (`prefabs/person.py`) are entities with these components:
- `AssociativeMemory`, `ObservationToMemory`, `LastNObservations`;
- a `Constant` identity;
- the `Learner` (`components/culture.py`);
- their **customs**, one `Custom` component each.

Customs are added and removed while a person lives (`EntityAgent.add_component`
/ `remove_component`). They are rebuilt on restore through the entity's
`component_factory`.

A person acts through the stock `ConcatActComponent`, or a person's
`HumanActComponent`, or a rule (`act_component_factory`). The `Learner` takes up
the customs offered in observations, by the person's own learning biases, and
keeps the person within capacity: a new custom displaces the oldest of a full
domain.

**The World GM** (`prefabs/world_gm.py`) is `SwitchAct` over:

| Key | Component |
|---|---|
| `__next_acting__` | `TurnGate`: each participant answers once a year. The year closes when the last answer is in. |
| `__next_action_spec__` | `Questions` |
| `__resolution__` | `Answers`, which records each answer |
| `__make_observation__` | the stock `MakeObservation` |
| `__terminate__` | the stock `Terminate` |
| `lifeways_world` | the world (`world/`) |

The world is built from the GM's own params; no object is shared with any
entity.

**Closing a year.** Answers are resolved together, in a fixed order, when the
last one is in. A year's outcome therefore never depends on thread timing:
two runs with the same seed are identical.

**Inheritance without mind reading.** Every custom enters the world through
the GM, which keeps the public record of variants (`world/culture.py`). A
parent expecting a child is asked what it passes on, and names custom ids. The
GM copies them, with copying errors, into the newborn's params.

Work answers name the custom they follow (`gather food c3`). The world
credits a skill only if the GM once gave it to that person.

**Births and deaths.**
- A birth adds an entity with `Simulation.add_entity`; the engine starts the
  newborn's loop.
- A death archives the person's state, then calls `Simulation.remove_entity`,
  which ends that loop.
- The `Lifecycle` that `simulation.py` wires carries both.

**Indefinite runs stay bounded.**
- The raw log is trimmed (`play(log_retention=...)`).
- The async engine streams its log in batches.
- The dead leave the simulation; history windows roll.

**Checkpoints** are the standard ones (`Simulation.make_checkpoint_data`):
- The restore (`simulation.load_checkpoint`) removes founders who died and adds
  people born since, before `load_from_checkpoint`.
- A restored year plays on exactly as the live one.

This example relies on three core changes: entities joining and leaving the
asynchronous engine, entity agents adding and removing components, and
bounded async logs.

## Tests

```bash
pytest examples/lifeways
```

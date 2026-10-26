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
| `census.jsonl` | a row a year: the world's census (population, births, deaths, moves, offers by channel, variants, the skill people work by) and, under `held`, what the living hold (read from their entities) |
| `dead.jsonl` | each person who died: their entity's final state and params |
| `customs.jsonl` | each variant of a custom once nobody living holds it, with its parent: the long record of culture |
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
the customs offered in observations, by the person's own learning biases:
- each further person seen doing the same is another chance to take it up;
- a kind of custom in a domain is a locus, like a plasmid's incompatibility
  group: a new custom replaces the one the person holds there;
- a skill is taken up only if it is better than the one held;
- past `per_domain` customs of a domain, or `capacity` in all, the oldest goes.

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

Work answers name the customs they follow (`gather food c3 c6`). The world
credits a skill only if the GM once gave it to that person.

**Transmission by enactment.** What people do passes on what they do it by.
The GM sees which customs an answer names and offers copies of them, with the
channel's copying errors, as lines of next year's observations:

| Channel | Who is offered what |
|---|---|
| birth | the newborn: what its parents passed on, recombined by domain |
| upbringing | a child: what its parents do, and what they `teach` |
| observation | anyone within a radius: what the nearest few did it by, with how well fed they are |
| apprenticeship | a youth who chose to `learn a craft`: the skill of the nearest skilled worker, sometimes a level better |
| sanction | one shamed (`shame <name>`): the norm the shamer shames by |
| practice | one who keeps a norm only expected of it: that norm as its own |

**Norms in two forms.** A norm is *internalized* (a person's own, kept when
nobody watches) or *known* (expected by others, kept when someone is within
sight). Children raised in a norm make it their own; so may those who keep a
norm they only know, or are shamed for breaking one. What one sees others
keep, one knows. A known norm never displaces an internalized one.

**Sharing and shame.** A person with enough may `share food with` a needy
person within sight. One who keeps full stores and gives nothing to the
needy near them is seen doing so; next year, those who saw may `shame` them.
The world reports what was seen; whether it was wrong is each person's to
judge.

Copying errors lose know-how (a skill comes out a level lower) and never add
any; a skill improves only by an apprentice's care. A scenario's `channels`
setting sets each channel's reach, fidelity and the
form a norm comes in by it, and which channels exist. A founding custom's
`held_by` is the share of founders who hold it. What people take up is their own Learners' choice; the harness
reads it for the census (`Run.customs_census`), the GM never does.

**Arrivals.** While the valley is nearly empty, strangers may come (the
`arrivals` setting): in the Sandbox, the Hill people, who keep their elders'
ways, feed only their own and have children young. Their customs spread, or
not, by the same channels as anyone's.

**Partners.** One who looks for a partner looks until paired, with the
nearest who is looking too. Parents and children, and siblings, never pair.

**Births and deaths.**
- A birth adds an entity with `Simulation.add_entity`; the engine starts the
  newborn's loop.
- A death archives the person's state, then calls `Simulation.remove_entity`,
  which ends that loop.
- The `Lifecycle` that `simulation.py` wires carries both.

**Indefinite runs stay bounded.**
- The raw log is trimmed (`play(log_retention=...)`).
- The async engine streams its log in batches; each component keeps only its
  most recent logging data.
- The dead leave the simulation, and the variants nobody holds leave the
  world's record (both go to disk); history windows roll.
- `Run.play` sends what the engine prints to the null device.

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

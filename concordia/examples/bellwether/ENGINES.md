# Analogous scenarios using standard Concordia engines

The main Bellwether night remains **Sequential**. Its agenda, watch boundaries,
private conversation and consent/repair stages require ordered resolution.
Changing an engine argument alone is not an equivalent experiment.

## What the repository actually provides

| Standard engine | Scheduling / step semantics | Required configuration | Evidence here |
|---|---|---|---|
| `Sequential` | Select one actor, deliver observation, act, resolve; one step per selected actor | One next-actor name, one action spec, ordered GM state | Existing complete Bellwether fixture/live tests |
| `Simultaneous` | Select a group; get each spec; collect actions concurrently; resolve one multiline batch; one step per round | Group selection, complete-batch GM rule, isolated player contexts; no current-round peer choices in initial observations | New one-round council fixture and batch-contract tests |
| `Asynchronous` | Independent per-player observe/act loops, no round barrier; per-player iteration cap | ReactiveMeasurements/capture and thread-safe GM/components; explicit rules for concurrent state changes | Code audit only, **not run or certified compatible** with either example |

The modules are in `concordia/environment/engines/`. The simultaneous engine
already uses standard concurrency utilities and a GM observation/log lock.
The asynchronous engine validates `ReactiveMeasurements` and has different
pause/capture behavior. Do not implement another scheduling loop or treat
async completion order as simultaneous choice.

## Runnable simultaneous resource council

This is a separate, deliberately small analogue of cooperative resource
governance, **not the full Bellwether game**. Nell, Ivo and Sam each hold one
unit. Each independently chooses:

- `contribute`: explicitly authorize and execute donation of *their own* unit
  once the complete batch has arrived;
- `keep`: retain their unit.

Unlike Bellwether's request/accept/transfer/work stages, this contract makes the
choice itself authorization for immediate execution. There are no promises,
watch costs, hidden material effects, majority seizure or claimed social welfare
metrics. A member's choice cannot supply another member's consent.

From the contribution checkout:

```sh
python -m pip install -e '.[dev]'
python -m concordia.examples.bellwether.council --output runs/council-fixture
```

The command uses standard `NoLanguageModel`, fixed choice ordering and zero
fixture embeddings. Every fixture actor picks the first choice, `contribute`.
Expect one completed simultaneous round, three actions, Community3 / members0,
and the original3 units conserved. Outputs:

- `trace.json` and `trace.html`: standard SimulationLog exports;
- `outcome.json`: fixture label, complete flag, round count, decisions and
  inventory for this declared case.

The output proves software wiring and accounting, not cooperation among people.
One round is a bound, not a method for hiding incomplete results: check
`complete`, participant set and recorded inventory.

## Composition, reuse and failure semantics

`council.configuration()` returns a fresh standard Config and CouncilLedger.
It composes existing minimal actors, SwitchAct, NextActingAllEntities,
FixedActionSpec, MakeObservation/ObservationQueue and Inventory.apply.
CouncilLedger implements only the *scenario-specific* complete-batch rule.
The executable uses generic.Simulation with standard Simultaneous; no custom
engine or server is introduced.

All member/spec names are known before execution. Inputs are exact validated
choices. The engine may omit a failed actor task from its batch; the ledger
therefore rejects incomplete, duplicate, unknown or malformed batches **before**
any inventory mutation. A single Inventory.apply validates all contribution
deltas. Batch arrival/line ordering does not change the resulting balances.
Re-resolving the identical complete batch cannot spend resources twice.
An invalid batch raises an error; no silent partial round or inferred vote is
treated as success.

The fixed initial public observation contains rules, not other actors' current
choices. Actions are exposed to the GM only for collective resolution; this
example has no public browser view or per-user authentication of its own.
Raw developer traces can contain actor context. Do not publish them unchanged
if adding private information.

## Plug in real policies or human input

Use the same returned Config with a configured standard language model for
actors, retaining the explicit moderator components. Reuse the local timeout,
call-limit and profiling wrappers in `run.py`; a live sample is not a guarantee
of the all-contribute fixture outcome. Do not call the fixture a live model.

For human inputs:

```python
from concordia.environment.engines import simultaneous
from concordia.examples.bellwether import council
from concordia.prefabs.simulation import generic

config, ledger = council.configuration(
    human_readers={'Nell': nell_input, 'Sam': sam_input}
)
simulation = generic.Simulation(
    config, model, embedder, engine=simultaneous.Simultaneous()
)
log = simulation.play(max_steps=1)
```

Readers implement the existing HumanInput protocol and must be independently
routed/thread-safe. Both can be pending concurrently; **do not share one
blocking terminal prompt or assume the sequential browser's one-active-turn
projection supports this council unchanged**. The main multiplayer transport
remains built for Bellwether's ordered night. HumanAct's normal choice validation
remains in force; other actors keep their standard policies.

## Making another analogous example

1. State the timing semantics: sealed simultaneous intentions, sequential
   observable moves, or truly asynchronous interactions.
2. Use standard prefabs and choose the standard engine matching those semantics.
3. Define what a choice authorizes, what is only speech, and what evidence
   material transitions need. Keep rules, beliefs, enforcement and compliance
   distinct.
4. Test missing/failed actor behavior, partial effects, conservation, privacy,
   scheduling boundaries and termination before interpretation.
5. Record engine/model/config/revision and full private traces, then produce
   intentionally sanitized research outputs. Seeds do not make hosted models
   deterministic, nor do traces create empirical warrant.

For example, the same council contract can teach independent mutual-aid
contributions or a cooperative reserve decision. An institutional dispute with
later observable replies better fits sequential turns. Asynchronous emergency
reports need their own thread-safe state/observation design; they are a future
example, not an untested toggle offered here. Component serialization alone
does not establish engine/checkpoint continuation or scientifically comparable
branches.

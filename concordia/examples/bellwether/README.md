# Last Light at Bellwether

**Full three-watch game:** see [GAME.md](GAME.md) for the playable fixture and
local live-resident modes, the two strategies, and measured validation limits.

## Original shared-service slice (retained with --mode slice)

A **one-turn fixture**, not the complete live game or a research experiment.
This example teaches a reusable boundary between a human player, the standard
Concordia visual editor, and a noninteractive attached CLI. All three share one
`OperationService`; the two HTTP listeners are instances of the existing
`SimulationServer`, not independent simulations.

## Run from this contribution's source checkout

```sh
python -m pip install -e '.[dev]'
python -m concordia.examples.bellwether.run --mode slice --editor-port 8784 --player-port 8785 --output runs/bellwether-fixture
```

Open <http://127.0.0.1:8784> for the editor and
<http://127.0.0.1:8785> for the player. **No simulation executes on opening**:
select `run.start` in the editor's operation panel and Apply once. Then submit
one proposal from the player screen. The run uses `Simulation.play`, standard
`Sequential`, minimal actor prefabs, `HumanActComponent`, `SwitchAct`,
`MakeObservation(allow_llm_fallback=False)`, and fixed context components.
Four resident actors are instantiated but do not act in this slice. The only
acting entity is the human coordinator. `NoLanguageModel` is used; no API key,
paid model, or model download is required.

The fixture acknowledges the action without granting anyone's consent,
transferring fuel, doing repair work, or predicting a resident's behavior.
Standard `SimulationLog.to_json()` and `to_html()` outputs are written to the
chosen directory. Ctrl-C stops these private listeners and cancels pending input.

## A shared edit in both directions

Before Run, select `component.edit` in the editor. Its only parameter, `value`,
is the exact replacement for Mara's **designated Constant context component**
`PreviousStormAccount.state`. This is **not an arbitrary episodic-memory edit**.
Quotes, line breaks, Unicode and literal markup are preserved. Initial prefab
configuration, other components, and other residents' knowledge stay unchanged.
The event records before/after, audience origin, operation ID and effective time.
The updated snapshot appears in all editor clients and the attached CLI:

```sh
concordia-session --url http://127.0.0.1:8784 discover
concordia-session --url http://127.0.0.1:8784 state > state.json
concordia-session --url http://127.0.0.1:8784 watch
```

For an attached edit, prepare a JSON envelope using a freshly queried revision:

```sh
python - <<'PY'
import json
from pathlib import Path
state = json.loads(Path('state.json').read_text())
request = {
    'operation': 'component.edit',
    'arguments': {'value': 'Mara remembers a disputed account.\nNo one else is informed.'},
    'references': state['references'],
    'revision': state['revision'],
    'retry_key': 'my-edit-1',
}
Path('edit.json').write_text(json.dumps(request, ensure_ascii=False))
PY
concordia-session --url http://127.0.0.1:8784 call --input edit.json
concordia-session --url http://127.0.0.1:8784 call --input edit.json
```

The second call returns the **original** result, not a second edit. Standard
input is supported with `call --input -`. Use `python -m
concordia.command_line_interface.concordia_session` if running without installing
the console entry point. Watch emits one JSON object per SSE state update.
Timeout/disconnection has a nonzero exit and requires reattaching; it is not a
durable event-replay cursor. Errors are structured JSON on stderr: exit 2 for
an HTTP rejection, exit 3 for input/connection errors. A stale revision requires
querying current state and reviewing the edit, not blindly changing its revision.

## Supported operation inventory

| Operation | Audience | GUI | CLI |
|---|---|---|---|
| `session.inspect` | developer | operation selector / state panel | `call` |
| `component.edit` | developer | exact text field + Apply | `call` |
| `run.start` | developer | explicit one-turn Run | `call` |
| `run.pause` | developer | request pause | `call` |
| `player.view` | player | player screen / filtered state | `call` on player port |
| `human.respond` | player | action box | `call` on player port |

`discover`, `state`, and `watch` use `/api/operations`, `/api/state` and
`/api/events`. Every mutation uses `/api/dispatch` with the same registry input
schema. Only required scalar string/integer arguments are currently supported;
this is **not a general JSON Schema validator**. Each descriptor documents the
input fields. Session/project/run/branch references are explicit and
process-lifetime, not checkpoint identifiers. Retry keys are retained for that
lifetime; if the ledger reaches capacity, new mutations are rejected rather than
evicting replay protection.

## Execution and privacy boundaries

A pause flag does not prove quiescence. This first increment deliberately rejects
edits while the execution thread is alive, **including when awaiting a human**.
It permits the designated edit before starting, and after the worker has fully
returned with the controller paused. One run cannot be replaced or restarted.
Continuing, restoring, cancelling model calls, or editing in-flight work is not
supported. No engine or StepController stepping semantics are changed.

The editor's graph/inspector remains a read-only view of initial prefab data;
its attached operation panel explicitly shows last-safe runtime state and traces.
Legacy run/edit endpoints are disabled for capability-bound listeners so no GUI
or CLI caller can bypass revisions or the quiescence guard. Normal
`SimulationServer` users without an operation service keep the existing routes.

The player listener delivers only public scenario configuration and the human
coordinator's own `HumanSession` snapshot. It never sends developer checkpoints,
resident private observations, or the edited private account. Unknown/legacy
player routes are denied; the player cannot select a developer audience in JSON.
SSE uses that same filtered view and replays a current snapshot on reconnect.
The service chooses audiences **at listener construction**, not from user input.
These loopback listeners are not authentication against another process on the
same machine; the developer port is for a trusted author. Do not publicly expose
it. The example creates no tailnet route, lobby, credential, or access-control
configuration. Player and developer browsers must use their designated ports.

## Tests and scope limits

```sh
python -m pip install pytest playwright
python -m playwright install chromium
python -m pytest -n 0 concordia/examples/bellwether/service_test.py concordia/examples/bellwether/browser_test.py
```

Tests use real standard components and HTTP/CLI/Chromium, but prohibit
`Simulation.play` in the service/browser suite. They verify exact edit isolation,
fresh ownership, stale/invalid atomicity, simultaneous retries, actual worker
liveness versus pause, GUI/CLI semantic parity, reconnect, and player filtering.
A separate manually launched one-turn fixture walkthrough verifies human input,
standard sequential resolution and logs. Automated checks are not human usability
or social/scientific validation.

**Beyond the original one-turn mode:** three-watch accounting, consent/work,
live residents, two strategies and dawn are now provided by [GAME.md](GAME.md).
Human 15–25 minute usability remains unmeasured. Still backlog: project authoring,
full editor-family parity, assets/undo/breakpoints, isolated checkpoint continuation,
branches, experiments, measurement/export and fresh-environment showcase tours.
The full Bellwether A–H acceptance and six-workflow baseline are not satisfied by
this slice. Mechanics values are explicit initial fixture configuration only;
no narrative can be interpreted as a material or consent change.

## Portable public account

Full-night players and approved spectators can download a script-free public
timeline and JSON accounting without sharing private journals. See
[PUBLIC-ACCOUNT.md](PUBLIC-ACCOUNT.md) for scope, CLI use and non-replay limits.

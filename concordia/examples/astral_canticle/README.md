# The Astral Canticle — a human in the loop

A mobile-friendly, Zork-like science-fantasy adventure. You control **Ilyra Venn**;
**Sable-9** and **Thorn of Io** retain their own language-model acting policies.
The GM uses the normal `SwitchAct` component. No player action is invented by an
LLM, and there are no lobbies, accounts, multiplayer sessions or paid services.

The Auric Vesper / sleeping moon / star-loom setting is a small demonstration
of human input rather than a research experiment. The opening scene has inspectable objects,
connected locations and command suggestions. Narrative and consequences are
model-generated, not a hard-coded puzzle or a deterministic inventory simulator.

## Run locally

From the repository root, with Python 3.12+ and a running local Ollama:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[ollama]'
.venv/bin/python -m pip install -r concordia/examples/astral_canticle/requirements.txt
ollama pull llama3.2:3b
.venv/bin/python -m concordia.examples.astral_canticle.web \
  --model llama3.2:3b --port 8767 --max-steps 30 --output runs/human
```

Open <http://127.0.0.1:8767/>. The command starts exactly **one** adventure. Page
loads/reloads never launch or reset a simulation. A turn waits indefinitely for
you; model responses can take a while. Other entities act once each between your
turns. The chapter ends after `--max-steps` entity actions (not human rounds).

Try **EXAMINE the resonance cradle**, **TAKE the tuning fork**, **INVENTORY**,
**WEST**, **TALK TO Sable-9**, or describe an idea in ordinary words. Suggestions
fill the command box; **Act** submits. Enter submits; Shift+Enter inserts a line.
**Guide**, **Aa** and **Journal** provide help, larger text, and a transcript export.

A dropped phone connection does not cancel a prompt. Reload reconnects to the
same process, and drafts are retained in that browser's local storage. Responses
carry unique request IDs: stale actions are rejected and retries of an accepted
POST do not act twice. Drafts are not automatically sent after reconnecting.

Completed steps are written as standard `SimulationLog` artifacts in the output
directory: `simulation.json` and a self-contained `log.html`. These contain all
entities' detailed logs and are **host-side only**. The player download contains
only that player's observations/actions. A host restart starts a **new chapter**;
old prompt IDs cannot be replayed. This example does not resume an interrupted
simulation process from a checkpoint.

## Select the human basic prefab

Add `--player-prefab basic` to either the web or terminal command:

```sh
.venv/bin/python -m concordia.examples.astral_canticle.web \
  --player-prefab basic --model llama3.2:3b --port 8769 \
  --max-steps 30 --output runs/human-basic
```

Ilyra is built directly by the library's `concordia.prefabs.entity.basic.Entity`.
Only `ConcatActComponent` is replaced with `HumanActComponent`; the prefab's
instructions, observation-to-memory, observation history, goal,
`SituationPerception`, `SelfPerception`, `PersonBySituation`, memory settings and
logging all remain unchanged. **The three perception components still call the
LLM before each human prompt.** The complete assembled context is visible under
**Your context** on the phone, including all three perception answers. It is one
verbatim string in the basic prefab's own Concat order, with original labels and
line breaks. It also appears in the standard host-side logs. The Journal download
remains your observations and submitted actions. The final action is always your
input, not the model's suggested behavior.
Sable-9, Thorn and the GM keep their existing composition and policies.

The default remains `--player-prefab minimal`, the previous lightweight
configuration with no LLM-based player context processing. In `--role gm`, this
option still selects Ilyra's prefab, but Ilyra uses its normal LLM acting policy
and the human controls the GM. Python callers can use
`build_cast(..., player_prefab='basic')` or `play(..., player_prefab='basic')`.

The library injection is reusable independently of this example or transport:

```python
from concordia.components.agent.human_act_component import HumanActComponent
from concordia.prefabs.entity import basic

player = basic.Entity(params={'name': 'Ilyra Venn', 'goal': 'Repair the loom.'}).build(
    model=model,
    memory_bank=memory_bank,
    act_component_factory=lambda order: HumanActComponent(read, component_order=order),
)
```

The factory receives the exact order calculated inside the prefab (including
the goal placement); no ordering list is duplicated in the example. Both basic
and minimal also accept a prebuilt `act_component`, but the two options are
mutually exclusive. Omitting both leaves the library's default `ConcatActComponent`
(including its ordering and choice/name-prefix options) unchanged. No parallel
copy or subclass of the basic prefab is maintained.

## Phone access on an existing tailnet

The web server binds **only to 127.0.0.1**. Use Tailscale Serve, not Funnel. Inspect
current configuration first and add only the new path; never reset other routes.
For a host named `your-host.your-tailnet.ts.net`:

```sh
tailscale serve status --json
# Include these flags on the web command above:
# --allowed-host your-host.your-tailnet.ts.net --root-path /astral-canticle-play
tailscale serve --bg --set-path /astral-canticle-play http://127.0.0.1:8767/astral-canticle-play
```

Open `https://your-host.your-tailnet.ts.net/astral-canticle-play/` on your phone.
The trailing slash matters for relative asset URLs. The proxy target preserves the mount
prefix; the adapter also accepts prefix-stripping proxies. Same-origin JSON
POSTs, host validation and a restrictive CSP protect the browser boundary. This
is a **single trusted tailnet controller**, not an authorization system: anyone
with access to this route shares control. Do not expose it publicly. Existing
`/astral-canticle`, `/astral-canticle-data`, and `/game-design-forum` are separate
log/forum routes and need not be changed.

## Human game master and terminal input

Launch with `--role gm` to control the GM instead. All three player entities then
use their default LLM policies. You provide observations, pick actors, create
valid next-action specs, resolve attempts and decide termination. GM decisions
are never silently filled in by model-based context components. Choice buttons
use exact option values; the action-spec builder creates validated JSON. Both
roles see only the controlled entity's complete pre-act context; unrelated NPC or
GM component outputs are never added to it. This is a developer-oriented secondary UX;
the default player mode is the polished single-human experience.

The same adventure also works without **any web dependencies**:

```sh
.venv/bin/python -m concordia.examples.astral_canticle.terminal --role player
# Use --role gm for terminal GM control.
```

A minimal transport implementation is simply a callable:

```python
from concordia.components.agent.human_act_component import HumanActComponent

def read(request):
  print(request.context)  # Already ordered and labeled; do not reassemble the map.
  if request.error:
    print(request.error)
  return input(request.action_spec.call_to_action + '> ')

act_component = HumanActComponent(read)
# Install in EntityAgentWithLogging, or pass to minimal.Entity.build(
#     model=model, memory_bank=memory_bank, act_component=act_component).
```

`concordia.typing.human_input.HumanInputRequest` contains the entity name, unique
request ID, immutable context map, preassembled `context` string, action spec and
retry feedback. Core has no knowledge of HTTP, browsers, threads or Tailscale. Independent readers can be
wired to additional human entities in a future application; this example keeps
one inbox per controller and deliberately implements no multiplayer routing.

Only `{name}` in a call-to-action is interpolated, without corrupting literal
JSON braces. Free prose must be nonempty, numbers finite, choices exact, and a
GM's `NEXT_ACTION_SPEC` must decode to a valid player `ActionSpec`. Skip actions
return without requesting input. Cancellation/EOF is raised, never converted to
an invented action. Context lifecycle and standard logging remain unchanged.
LLM-based context components can still use a model when someone composes a
human entity with them; the default minimal selection uses only non-LLM human
context components, while basic preserves its LLM-backed perceptions.

### Context ordering contract

`HumanActComponent(reader, component_order=None)` uses the context mapping's
iteration order. An explicit sequence uses that order first, then appends
unspecified keys **alphabetically**, exactly as the existing Concat implementation
does. Empty strings are skipped; all other labels, spaces and line breaks remain
intact. Explicitly named absent keys raise `KeyError` when acting; duplicate keys
raise `ValueError` at construction. `get_context_concat_order()` returns the
snapshotted order, and state saves/restores it with Concat's existing conventions
(including its empty-order-to-None state normalization). The reader is not saved.

Both policies call the same standard `concat.concat_contexts` helper; transports
render `request.context` verbatim rather than adding dictionary headings. The
player and GM UI both show it expanded in the scrolling page (not a cramped
composer panel), and terminal input prints the very same string. Context only
comes from the entity attached to this HumanActComponent. Showing the human's
own perception outputs does not expose another entity's private components.

## Composition and verification

The example reuses `minimal.Entity` / `basic.Entity`, `EntityAgentWithLogging`, observation/memory
components, `SwitchAct`, `NextActingInFixedOrder`, `FixedActionSpec`,
`Sequential`, `SimulationLog` and its standard `to_html()` renderer. A constant small
embedding is sufficient because this composition retrieves chronological
observations, not semantic neighbors. The GM sees the recent event history and
setting; we do not duplicate an engine, event resolver, clock or logging system.
Unlike the heavier time-and-place prefab, the lean GM tracks the Canticle Clock
in narrative context rather than generating separate clock/location LLM calls.

```sh
.venv/bin/python -m pip install pytest pytest-xdist pyink isort
.venv/bin/python -m pip install -r concordia/examples/astral_canticle/requirements-test.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest -n 0 \
  concordia/components/agent/human_act_component_test.py \
  concordia/components/agent/context_order_test.py \
  concordia/environment/engines/sequential_test.py \
  concordia/prefabs/entity/basic_test.py \
  concordia/prefabs/entity/prefabs_test.py \
  concordia/examples/astral_canticle
```

Unit and browser tests use mock models or transport requests, never a live model.
They cover exact Concat/Human context/order parity, full basic perception outputs
visible verbatim in player/GM mobile views, basic/default component parity and
perception lifecycle, CLI selection, player/GM contracts, normal NPC policies, a complete standard-engine
round, stale/duplicate actions, cross-origin rejection, reconnects, draft recovery,
mobile layout, safe text rendering, and GM action-spec controls. Web tests skip
when their optional dependencies are absent. The web dependencies are not added
to core installation requirements.

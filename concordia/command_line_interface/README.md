# Design simulations interactively with concordia-session

`concordia-session` attaches to the same registered project service as the visual
editor. It can inspect, author, validate, save and control simulations using the
same operation validation and draft helpers. It is a restricted command prompt,
**not a shell or Python interpreter**. Commands never implicitly run or save a
simulation. The server chooses the available templates, components and audience.

## Setup and attach

From a Concordia checkout containing the editor changes, install the package and
its dependencies in your Python environment:

```sh
python -m pip install -e .
node --version
concordia-session --help
```

Install Node.js separately if unavailable. Local draft actions execute the exact
packaged editor JavaScript in Node with JSON input; they do not execute user code.
Runtime operations and machine-oriented `state`/`discover`/`watch` do not require
Node. No terminal emulator package is needed. Browser dependencies such as
Playwright are needed only for development tests.

Attach to a trusted editor service already running, or start the optional
[project editor example](../../examples/project_editor/README.md) in a separate
terminal. That example is proposed in dependent PR #389; it is not included in
the standalone core-editor PR #314. Its default backend is NoLanguageModel;
preview/edit/save do not call a provider. This command starts the editor service,
not a simulation (Run remains explicit):

```sh
python -m examples.project_editor.run --port 8080 --model-backend none
```

For another registered project, use that application's documented launch command
and URL. Do not start a second service on an occupied port. Model configuration
belongs to the launcher, not this prompt; no API key entry is required by the CLI.
A no-model Run exercises the configured mock behavior, not real language-model
text. A live-model Run may incur provider charges.

```sh
concordia-session --url http://127.0.0.1:8080 interactive --draft my-draft.json
```

The URL and journal stay selected until exit. The journal is **client-owned JSON**
containing the document, saved revision, preview metadata, selection and undo/redo
stacks. It is not an exported project and not command history. Treat authored
content and exported logs as your data. New journals initialize from the current
saved project on the first draft action. Existing journals reopen their own
unsaved work; opening a prompt never discards or saves it.

Use `help`, `files`, `history`, and `exit`. `history` is in-memory only, never
loaded from or written to a terminal history file. `!1` displays entry 1 **without
executing it**; copy the command back to submit explicitly. EOF exits without
saving/resetting; Ctrl-C at the prompt cancels input. Interrupting a submitted
request cannot undo acceptance: inspect `state` before retrying. Where Python readline is available, Up/Down recalls this prompt’s submitted
commands in memory; existing process history is restored on exit. The loop has
no asynchronous output thread.

## Discover your project and inspect it

At the `concordia>` prompt (omit the prompt itself when copying):

```text
help
discover
catalog templates
catalog prefabs
catalog components
list instances
list components
list groups
list scene_types
list scenes
inspect simulation
inspect alice
select alice
```

`catalog templates` lists the registry's allowed template names without
constructing each template. `catalog prefabs` lists current-template prototypes,
defaults, editable field metadata and references. `catalog components` shows
registered types, dependencies, defaults and compatible prototypes. It does not
list arbitrary installed Python classes. Switching templates means loading a
valid project JSON registered by this service; there is no arbitrary prefab
import or template constructor command.

The following walkthrough uses the example's existing stable IDs `alice`, `bob`,
`conversation`, and `opening`. It also works against the guarded scene test
fixture. Other applications should first discover their IDs and prototype names.
Instances have bare IDs; other record selectors have a prefix, such as
`components:reminder` or `scenes:encore`. `.` means the current selection in local
draft commands. It is not a filesystem path or a runtime actor wildcard.

`inspect alice` returns the authored record, editable field descriptions, owned
components and available preview state. To inspect one built component, use
`inspect alice Instructions` (choose an actual key returned by `inspect alice`).
`select alice Instructions` selects and displays that component in the CLI and
expands it in the browser. Preview component state reflects the last save/load,
not unbuilt edits; a newly added actor needs Save to build its preview.

## Create cast, components, groups and scenes

This is a complete authoring journey: all IDs below are either existing example
IDs or explicitly created by these commands. `set` takes JSON, so string values
need JSON double quotes inside command single quotes. Keep `--id` values unique.

```text
set alice params.goal '"Find music everyone enjoys"'
add instance alice --id charlie
set . params.name '"Charlie"'
set charlie params.custom_instructions '"Charlie listens carefully."'
add component constant charlie --id reminder
set components:reminder params.state '"Listen before replying."'
move-component components:reminder bob
add group --id trio
set groups:trio participants '["alice","bob","charlie"]'
add scene-type --id meeting
set scene_types:meeting group '"trio"'
set scene_types:meeting game_master '"conversation"'
set scene_types:meeting premise '"Choose a song together."'
add scene --id encore
set scenes:encore scene_type '"meeting"'
set scenes:encore participants '["alice","bob","charlie"]'
set scenes:encore num_rounds 3
set scenes:encore premise null
move scenes:encore up
undo
redo
set simulation max_steps 40
validate
save
```

`null` inherits the scene type premise; a JSON string overrides it. Scene order
and round counts are distinct from the simulation's maximum engine-step count.
The standard SceneTracker and engine determine scheduling. `move ID up|down`
reuses existing ordering helpers; components move among their owner's authored
components, actors among the same role. A boundary move reports no change.
Field edits may temporarily be invalid; failed structural operations leave the
draft/history unchanged. Save validates references, registered fields and bounds.

Create another GM using `add instance conversation --id second-gm`, then inspect
it and change its existing `params.FIELD` values. Use `references conversation`
to find incoming uses. `replace conversation second-gm` rewrites compatible
references (including scene-type GM links); it does not transfer owned components
or delete the original. `move-component` transfers only to a compatible owner,
preserving settings and placing the component last there. `remove conversation`
then succeeds only if role/reference constraints permit it. Undo can restore it.

`duplicate charlie` copies the actor and its owned components and selects the new
record; `inspect` shows its generated ID. `remove` removes the selection.
`duplicate`/`remove` also work for component/group/scene-type/scene selectors.
Referenced groups/types cannot be deleted until their users are changed. These
are the same safeguards as the editor, not raw unrestricted JSON mutation.

## Correct errors, export and reopen

```text
set simulation max_steps 0
validate
locate '$.max_steps: must be positive'
undo
validate
save
export project.json
reload --discard
load project.json
inspect simulation
```

The invalid value deliberately demonstrates recovery. Use the **actual registry error
message starting with `$.`** (omit the CLI `Error: HTTP ...` prefix) with `locate 'MESSAGE'` to get the matching record/field and change the
local selection. Browser commands focus the field; external CLI prints the
selection and field identifier. Ambiguous or stale paths are rejected, not
guessed. The GUI's error link additionally checks that its submitted draft has
not changed; a CLI `locate` applies the supplied message to the current draft.

`export PATH` writes this client's document, including unsaved edits, without
publishing. `load PATH` validates/builds a local preview and resets local undo
history; it refuses to replace an unsaved draft. `reload --discard` deliberately
replaces local work and history with the saved server definition. **Only `save`
publishes to the session.** Loading a file does not save it. File paths are local
to this CLI, relative to its working directory; quote paths containing spaces.
There is no shell expansion, arbitrary server file access or browser filesystem
access. Export paths overwrite existing destination files explicitly; journal and
import/output path collisions are rejected. Keep project exports and the journal
in different files.

A stale revision rejects Save/Run. Export your work before `reload --discard`,
then reapply intended edits. A new server session requires export or explicit
reload before continued edits. CLI and browser drafts/selection/history are
independent: neither can silently modify another tab's unsaved fields. Successful
Save updates shared saved state; another dirty client must resolve its revision.

## Run, inspect and intervene

After Save, with a runner configured by the application:

```text
run --steps 10
state
pause
state
step
state
play
state
reset
```

These are successive controls to use as the run reaches each state, **not a batch
script to paste blindly**. `run` starts fresh from the saved definition; it does
not resume. `play` resumes an acknowledged pause; `step` grants one engine step
from a pause. A pause request may wait for the current engine boundary. Reset
uses the existing controller's safe boundary handling, not process termination.
Runs may complete before you pause. Invalid/unavailable operations report errors
without inventing success. No dirty/stale draft is silently saved or discarded.

Requested steps default to 10 (clamped to the saved maximum); the numeric GUI
field and `run --steps N` use the same server bound. CLI has no persistent GUI
numeric field: specify the next run's length each time. A Run response reports
**acceptance**, followed by a current run snapshot: waiting, active, completed or
failed. Acceptance alone does not mean model output has arrived. Provider failures
remain sanitized by the configured adapter/server and appear in state/events.

Use `watch` inside the prompt for one blocking event stream. Ctrl-C returns to
the prompt; EOF/timeout ends that stream. There is no background stream racing
with input, no automatic CLI reconnect and no command replay. For continuous
monitoring alongside typing, open a second terminal:

```sh
concordia-session --url http://127.0.0.1:8080 watch
```

While paused, inspect actual component keys before editing:

```text
view runtime
inspect alice
inspect alice Instructions
edit alice Instructions "Alice now asks before choosing a song."
inspect alice Instructions
view definition
inspect alice
```

Only existing editable Instructions/Goal fields are supported by `edit`; the
server enforces pause and instance/component restrictions. Runtime changes do not
alter initial definitions. CLI edits are submitted directly, with no pending
runtime text form; browser typed edits reject conflicting unsaved runtime fields.

## Analyze and export structured logs

Current-log analysis requires the runner to have supplied a finished structured
log (or its supported raw-log fallback). The console transcript is not that log.

```text
log overview --source current
log entities --source current
log actions --source current Alice
log context --source current Alice --step 1
log step --source current 1
log timeline --source current Alice --verbose
log search --source current music
log memories --source current Alice
log components --source current --entity Alice --component Instructions
log dump --source current --output inflated.json
log export --source current --output run.json
log import run.json
log overview --source imported
log bundle --source imported --output viewer.html
```

These reuse `concordia-log` analysis, not a second implementation. `log export`
uses standard `SimulationLog.to_json()` for a reimportable archive. `log dump`
is inflated analysis JSON for tools such as jq; it is not the structured import
format. `step` executes;
`log step` inspects. Browser exports download files; browser imports use pickers.
Only the interactive CLI adds explicit path operands/`--output` inside the loop.
Larger than 900 kB logs should use standalone `concordia-log` locally because the
session operation API has bounded request sizes. Missing attached memories are
not synthesized. Imported logs remain local and never change the project.

## Automation remains compatible

```sh
concordia-session --url http://127.0.0.1:8080 discover
concordia-session --url http://127.0.0.1:8080 state
concordia-session --url http://127.0.0.1:8080 call --input request.json
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json --line 'inspect alice'
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json --line export --file project.json
```

One-shot results stay JSON, errors go to stderr. Friendly `call OPERATION
'JSON_OBJECT'` uses fresh envelope references/retry keys and the same audience,
revision and state checks; it rejects dirty/stale client drafts. `discover` gives
parameters, not authorization to bypass restrictions. Transport failures never
cause automatic mutation retries. Reconnect explicitly, inspect `state`, and
reconcile saved revision before proceeding.

## Editor/CLI coverage matrix

The browser column means typed commands in the current tab; all domain commands
also work in the external interactive CLI unless stated. Tests cover the shared
authoring journey, intercepted browser commands and existing operation contracts;
they do not establish physical Android usability or live provider behavior.

| Actual editor capability | Browser typed command | External CLI equivalent / limitation |
|---|---|---|
| Registered prefab picker, template/component descriptions/defaults/dependencies | `catalog`, `catalog templates`, `catalog prefabs`, `catalog components` | Same structured catalogs, restricted to registered/current-template entries |
| Actor/GM/initializer fields, name/persona/goal/reference choices | `inspect ID`, `set ID params.FIELD JSON` | Same fields and registry validation; inspect lists field metadata |
| Add/duplicate/remove actor or GM; minimum-role guard | `add instance PROTOTYPE [--id ID]`, `duplicate [ID]`, `remove [ID]` | Same helpers/undo; generated result becomes selection |
| Component CRUD, name/params/ownership | `add component TYPE OWNER [--id ID]`, `set components:ID name JSON`, `set components:ID params.FIELD JSON`, `duplicate`, `remove`, `move-component components:ID OWNER` | Same compatibility and dependency constraints |
| Built component list/expanded state and params | `inspect ID [COMPONENT]`, `select ID COMPONENT` | Focused JSON state/params; no graphical expand/collapse animation |
| Incoming references, reference navigation/replacement | `references ID`, `select TARGET`, `replace SOURCE TARGET` | Same references and compatible replacement; does not move another tab |
| Group CRUD/name/participants | `add group [--id ID]`, `set groups:ID name JSON`, `set groups:ID participants JSON`, `duplicate`, `remove` | Same list membership and validation |
| Scene-type CRUD/GM/group/default premise | `add scene-type [--id ID]`, `set scene_types:ID FIELD JSON`, `duplicate`, `remove` | Same scene-aware GM restrictions |
| Scene CRUD/type/participants/rounds/premise inheritance/order | `add scene [--id ID]`, `set scenes:ID FIELD JSON`, `move scenes:ID up|down`, `duplicate`, `remove` | Same record helpers; typed order also exposes retained ordering helpers (no current GUI reorder toolbar) |
| Simulation premise and authoring maximum | `set simulation premise JSON`, `set simulation max_steps N` | Same authoring fields; separate from requested run length |
| Steps-to-run input and Run | `run --steps N` | Same bound, explicit length per invocation; no remote numeric-field mutation |
| Validate/error target | `validate`, `locate 'MESSAGE'` | Same validation; prints target instead of clicking/focusing DOM |
| Save/load/export/reload | `save`, `load`, `export`, `reload --discard` | `save`, `load PATH`, `export PATH`, `reload --discard`; explicit client paths, separate journal |
| Undo/redo and unsaved status | `undo`, `redo`, `state` | Same draft history helper, stored only in this explicit journal |
| Definition/runtime source and inspection | `view definition|runtime`, `inspect ID [COMPONENT]` | Same source with textual record/component output; no SVG graph |
| Pause/step/resume/reset and state | `pause`, `step`, `play`, `reset`, `state` | Same service/controller state guards; Run acceptance is not completion |
| Paused Instructions/Goal Save button | `edit ID Instructions TEXT`, `edit ID Goal TEXT` | Same runtime operation; no pending browser input form |
| Hierarchy search, selection and panel tabs | `search TEXT`, `select ID [COMPONENT]`, `panel hierarchy|inspector|simulation|log` | Search returns matches; inspector returns selected content, hierarchy returns records, simulation/log return run/step snapshot; does not resize/activate another client |
| Splitter drag/keyboard/min/max/persistence | `layout`, `layout left|right|terminal PIXELS` | Explicitly browser-only; CLI rejects `layout`, resize your terminal using its host controls |
| Responsive typography, zoom, graph layout/scroll | Automatic CSS/browser controls | Terminal font/zoom/scrollback belong to terminal host, not model/domain actions; no simulated graph parity |
| Console input/history/help/results | `help`, Enter, Up/Down | Persistent prompt, `help`, `history`, `!N` display-only recall; no disk history or terminal emulation |
| Connection/error notifications, scrollback follow, dedup | Automatic single Simulation log; `watch` reports existing stream | Prompt reports request errors; blocking `watch` emits events; no async prompt notifications, automatic reconnect or replay |
| Structured logs/import/export/bundle | `log ... --source current|imported`, `log import` picker | Same analysis; `log import PATH`, `--output PATH` in interactive mode |
| Operation discovery/raw call/watch | `discover`, `call OPERATION JSON`, `watch` | Same discovery/dispatch checks; watch is one explicit stream rather than browser's existing subscription |

The CLI does not edit another browser's draft, pending runtime fields, layout,
selection or imported-log choice. Browser-only file pickers, focus, graph, zoom,
scrollback-follow and splitters have honest local/textual equivalents above,
not no-op parity commands. See [editor commands](../docs/editor-commands.md) for
GUI behavior and [utilities](../utils/README.md) for integration APIs.

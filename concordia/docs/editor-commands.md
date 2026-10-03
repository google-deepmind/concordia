# Editor and session commands

The **Simulation log** is the editor's message destination: errors, warnings,
connection changes, command results, player entity actions and completion messages appear
there. Validation errors retain **Show invalid field/item** buttons in their log
entry. Status labels in the toolbar remain compact. Reading older entries stops
automatic scrolling; new entries follow the bottom only if you were already near
it. Polling and reconnection deduplicate recorded steps and completion/failure
messages. Starting another run retains earlier entries. A page reload starts a
new local console and reconstructs the current run's recorded entries, without
replaying commands.

Type in **Simulation log command** and press Enter or Send. Up/Down recalls this
tab's command history; **Command help** lists syntax, including while offline.
This input is a command language, not a shell: it cannot execute programs, Python,
JavaScript or arbitrary server paths. Quotes keep arguments together; `set` takes a JSON
value, so string values need JSON quotes inside command quoting.

```text
help
state
run --steps 10
pause
step
play
reset
```

`run` starts a new saved definition and uses the same bounds as the Run button.
Without `--steps`, the browser uses its numeric **Steps to run** field; the CLI
uses the advertised default (10, clamped to the saved maximum). `play` resumes a
paused run. A paused `run` is rejected: use `play` or reset first. `step` grants
one engine step and requires an acknowledged pause. A dirty local draft blocks
Run. No command silently saves it.

## Draft, authoring and navigation

GUI edits and typed commands in a browser tab use the same document and undo
history. Structural commands call the existing registered prefab/component
helpers. The server validates saves with the same registry and revision checks
as the GUI. Field edits may leave an invalid draft for correction or Undo.

```text
set alice params.goal '"Find music both roommates enjoy"'
set simulation max_steps 40
validate
save
undo
redo
add instance minimal --id Charlie
add component constant alice
duplicate alice
remove INSTANCE_ID
references alice
replace OLD_INSTANCE_ID REPLACEMENT_INSTANCE_ID
move-component components:COMPONENT_ID bob
select bob
select bob SelfPerception
inspect bob
search "Instructions"
view runtime
edit alice Instructions state '"Alice prefers quiet music."'
view definition
panel log
export
load
reload --discard
```

Use stable IDs shown in the inspector. Instances use their bare ID; other records
use `components:ID`. `simulation`
selects the simulation settings. `duplicate` and `remove` without an ID use the
current selection. `set` edits an existing scalar/list record field or
`params.FIELD`; it cannot change identity, prefab registration or schema fields.
Component name and settings
use `name` and `params.FIELD`. Existing ownership/type/reference restrictions
apply; a failed action leaves the draft/history intact.

`validate` checks without saving. `save` saves the current draft. `export` exports
the **current local draft**, including unsaved edits; it does not save it.
`load` opens a JSON picker, validates and prepares a local preview, then replaces
the local draft, **without saving to the session**. It refuses to overwrite an
unsaved draft. Save/export it first or explicitly `reload --discard`. Load and
reload reset local undo history. A newly loaded file remains local until Save.

Runtime `inspect ID [COMPONENT]` reads the selected entity/component from the current runtime view. `edit` uses the existing paused
runtime edit operation and supports the same advertised dynamic component fields as
the inspector. The browser rejects a typed edit if it would conflict with pending
runtime input fields; save those fields first. Runtime edits never change the
initial definition. Selection, search, panel and view are local presentation
state; they do not move another user's browser or mutate a running simulation.

## Structured log analysis

These commands call the existing `concordia-log` handlers and
`AIAgentLogInterface`, rather than reimplementing analysis. They require an
explicit source: the current session's finished run or a client-imported
structured log. The console's messages are not a SimulationLog.

```text
log import
log overview --source current
log entities --source imported
log actions --source current Alice
log context --source current Alice --step 1
log step --source current 1
log timeline --source current Alice --verbose
log search --source imported "music"
log memories --source current Alice
log components --source current --entity Alice --component Instructions
log export --source current
log dump --source current
log bundle --source imported
```

`step` changes execution; `log step` only inspects an existing record. Log commands
accept the corresponding `concordia-log` analysis flags, including component keys
and step ranges. Output is structured JSON text. `log dump` and `log bundle`
download JSON/portable HTML in the browser; no browser command writes server
paths. `log import` opens a local file picker and does not alter the project.
Imports share the operation API's size limit (900 kB text, 1 MiB request); use
standalone `concordia-log` locally for larger logs.

The example runner retains its returned standard SimulationLog, including
attached memories. Other runners can call `SimulationServer.set_project_log(log)`.
If a finished runner has not supplied one, analysis uses its existing raw log;
that fallback does not synthesize attached memories. Current-log analysis is
unavailable until a run finishes. New Run clears the current structured-log
selection; local imported logs remain available. Reset retains the last runtime
and its log for inspection.

## External CLI

Existing machine interfaces are unchanged:

```sh
concordia-session --url http://127.0.0.1:8080 discover
concordia-session --url http://127.0.0.1:8080 state
concordia-session --url http://127.0.0.1:8080 call --input request.json
concordia-session --url http://127.0.0.1:8080 watch
```

Use the same language with `command --line`:

```sh
concordia-session --url http://127.0.0.1:8080 command --line help
concordia-session --url http://127.0.0.1:8080 command --line 'run --steps 10'
concordia-session --url http://127.0.0.1:8080 command --line pause
concordia-session --url http://127.0.0.1:8080 command --line 'log overview --source current'
concordia-session --url http://127.0.0.1:8080 command \
  --line 'log bundle --source imported' --file log.json --output viewer.html
```

An external CLI cannot read another browser tab's unsaved draft, history,
selection or pending runtime fields. Authoring requires an explicit local
`--draft JOURNAL.json`. A new journal initializes from the saved session; it
stores the local draft, metadata, original project revision, selection and
undo/redo stacks. Draft actions reuse the exact packaged JavaScript classes used
by the editor, executed with Node and JSON input. Node is required for those
local actions, not for runtime commands or standalone log analysis.

```sh
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json \
  --line 'set simulation max_steps 40'
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json --line undo
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json --line save
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json \
  --line load --file initial-project.json
concordia-session --url http://127.0.0.1:8080 command --draft my-draft.json \
  --line export --file exported-draft.json
```

Only Save publishes the CLI draft to the shared session. Stale saves and Run with
an unsaved/stale journal are rejected. `reload --discard` explicitly replaces the
journal's draft/history from the saved session. CLI view/panel/search/selection
change its journal, not any browser tab; `inspect`/`search` return JSON instead of
moving a graphical panel. `log import` is a browser picker action: the CLI uses
`--file LOG.json` with `--source imported`. `--output` is the explicit local file
for log dump/bundle. Results remain JSON on stdout; errors are JSON on stderr.
Transport failures are reported, never automatically retried as commands.

### Operation discovery, calls and watching

`discover` lists the registered operations available to the server-selected
client audience. `call OPERATION 'JSON_OBJECT'` dispatches literal arguments,
for example `call session.plan '{"line":"help"}'`. Mutations use the same
revision, session references, retry key and audience validation as GUI operations.
They do not bypass state restrictions. Calls reject unsaved client fields (or a
stale CLI journal); save or explicitly reload first. Use discover to obtain
required parameters, including saved project revision for project.save/run.
Raw calls operate saved server state, never another client's draft.

`watch` in the browser reports its existing live connection; it creates no new
stream and never replays commands. Reconnection remains automatic. On the CLI,
`command --line watch` is an alias for the existing `watch` transport: it emits
JSON event lines until interruption, EOF or transport timeout, using one stream.
Legacy `discover`, `state`, `call --input FILE` (or stdin), and `watch` modes are
unchanged. `command --line discover` and `command --line 'call ...'` emit JSON.

The graph summary contains only compact step/entity indicators. Narrative action
text and commentary appear only in Simulation log. Entity inspector and component
contents are a separate state display, so they may contain descriptive text;
they are not a second command-result or notification destination.

### Terminal scrollback and panel sizing

Simulation log and its command prompt share one scrollport. The prompt is the
last content line, not a fixed footer: scroll up to read older output and it
scrolls out of view. New entries appear before the prompt. Output follows the
bottom only when you are already near it; otherwise your reading position stays
put. Output does not clear or replace a command you are composing. Enter,
Up/Down history, IME input and Command help keep their existing behavior. This
is still the restricted editor command language, not an operating-system shell.

On wide layouts, drag the boundaries beside Hierarchy/Inspector or above
Simulation log with a pointer or touch. The terminal can take most of the
available workspace. Keyboard users can Tab to each named separator and use
arrow keys (Shift for larger increments), Home for its minimum, or End for its
maximum. Minimum sizes preserve usable neighboring panes. Sizes are stored as
ratios in this browser origin's localStorage and clamped to the current viewport;
unavailable storage does not prevent resizing. At widths of 700 CSS pixels or
less, the existing full-panel tabs replace adjacent panes and separators are
hidden. Desktop sizing is restored and bounded on return to a wide layout.

Fine-pointer desktop preserves the inspector's established hierarchy: title
14px, section headers 12px, parameter rows 11px, component state/class text 10px
at the default root size, all expressed in rem. Hierarchy and simulation controls
use compact 12px text. Narrow screens and coarse pointers enlarge actual
inspector detail descendants to 16px, alongside other touch text and controls.
This depends on viewport/input capability, never hostname or URL scheme; browser
zoom/text sizing remains enabled. SVG graphs use their native layout width in rem
rather than expanding to fill a wide panel. Touch/narrow layouts apply an explicit
1.4 scale for graph legibility; smaller panels scroll the graph instead of
shrinking its labels. Root text sizing scales the graph as well.

For a complete interactive authoring walkthrough and explicit editor/CLI coverage,
see the [simulation design CLI tutorial](../command_line_interface/README.md).

New shared discovery/navigation commands: `catalog [templates|prefabs|components]`,
`list [instances|components]`, `locate "MESSAGE"`,
`move ID up|down`. Creation accepts optional `--id STABLE_ID`; `.` selects the
current local draft record. `layout` reports browser bounds;
`layout left|right|terminal PIXELS` uses the same bounded/persisted splitters.
The external CLI rejects browser sizing explicitly.

Use `concordia-session --url http://127.0.0.1:8080 interactive --draft draft.json`
to keep connection arguments and a client journal across commands. This prompt
adds explicit `load PATH`, `export PATH`, `log import PATH` and log
`--output PATH`, plus `files`, in-memory `history`, display-only `!N`, and `exit`.
It has no asynchronous stream: `watch` blocks until interrupt/timeout/EOF.

`log export` uses standard SimulationLog serialization for reimportable JSON;
`log dump` remains inflated analysis JSON and is not an import archive.


## Editor/CLI coverage matrix

The browser column means typed commands in the current tab; all domain commands
also work in the external interactive CLI unless stated. Tests cover the shared
authoring journey, intercepted browser commands and existing operation contracts;
they do not establish physical Android usability or live provider behavior.

| Actual editor capability | Browser typed command | External CLI equivalent / limitation |
|---|---|---|
| Registered prefab picker, template/component descriptions/defaults/dependencies | `catalog`, `catalog templates`, `catalog prefabs`, `catalog components` | Same structured catalogs, restricted to registered/current-template entries |
| Player entity/game master entity/initializer fields, name/persona/goal/reference choices | `inspect ID`, `set ID params.FIELD JSON` | Same fields and registry validation; inspect lists field metadata |
| Add/duplicate/remove player entity or game master entity; minimum-role guard | `add instance PREFAB_OR_PRESET [--id ID]`, `duplicate [ID]`, `remove [ID]` | Same helpers/undo; generated result becomes selection |
| Component CRUD, name/params/ownership | `add component TYPE OWNER [--id ID]`, `set components:ID name JSON`, `set components:ID params.FIELD JSON`, `duplicate`, `remove`, `move-component components:ID OWNER` | Same compatibility and dependency constraints |
| Built component list/expanded state and params | `inspect ID [COMPONENT]`, `select ID COMPONENT` | Focused JSON state/params; no graphical expand/collapse animation |
| Incoming references, reference navigation/replacement | `references ID`, `select TARGET`, `replace SOURCE TARGET` | Same references and compatible replacement; does not move another tab |
| Simulation premise and authoring maximum | `set simulation premise JSON`, `set simulation max_steps N` | Same authoring fields; separate from requested run length |
| Steps-to-run input and Run | `run --steps N` | Same bound, explicit length per invocation; no remote numeric-field mutation |
| Validate/error target | `validate`, `locate 'MESSAGE'` | Same validation; prints target instead of clicking/focusing DOM |
| Save/load/export/reload | `save`, `load`, `export`, `reload --discard` | `save`, `load PATH`, `export PATH`, `reload --discard`; explicit client paths, separate journal |
| Undo/redo and unsaved status | `undo`, `redo`, `state` | Same draft history helper, stored only in this explicit journal |
| Definition/runtime source and inspection | `view definition|runtime`, `inspect ID [COMPONENT]` | Same source with textual record/component output; no SVG graph |
| Pause/step/resume/reset and state | `pause`, `step`, `play`, `reset`, `state` | Same service/controller state guards; Run acceptance is not completion |
| Initial dynamic field Save button | `state-field ID COMPONENT FIELD JSON`, then `save` | Same advertised fields; fresh-build validation and persistent draft overrides |
| Paused dynamic field Save button | `edit ID COMPONENT FIELD JSON` | Same runtime operation and pause boundary; no pending browser input form |
| Central viewer selection/refresh | `viewer [default|NAME]`, `viewer-refresh [NAME]` | `viewer-refresh NAME` fetches fresh HTML; lists/fetches registered view data; `--file PATH` exports HTML; does not change another tab |
| Open viewer HTML/URL | `viewer-load`, `viewer-url URL` | `viewer-load PATH` checks a local HTML file; URL command returns a browser destination; terminal does not render HTML |
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

## File versions and recovery

Version 4 contains initial `dynamic_states` keyed by entity ID, component key and
field name, alongside authored component recipes. It has no scene-specific
editor records. Versions 1 and 2 retain their registered contracts. Version 3
requires explicit offline recovery: preserve the original file and use its
original editor to export/read its settings, then recreate them in a version 4
document. Merely changing the version number is insufficient. No saved files are
rewritten on load and there is no bundled automatic migration tool.

## Generic component state contract

`state-field ID COMPONENT FIELD JSON` edits an initial draft override. `save` and
`validate` construct a fresh preview and apply it through
`Simulation.set_component_dynamic_state`. Runtime `edit ID COMPONENT FIELD JSON`
uses that same setter at an acknowledged pause boundary, for either entity role.
The original `edit ID COMPONENT TEXT` remains shorthand for the `state` field.
Components advertise editable fields with `get_dynamic_state()` and own validation
in `set_state()`. JSON is preserved as objects/lists/numbers/booleans/null; text
fields remain text. State fields that are not advertised are inspection-only.
A failed Save leaves the saved definition and existing runtime unchanged.
`state-reset ID [COMPONENT [FIELD]]` removes overrides; **Reset to prefab** offers
the same recovery, including for components no longer present after a parameter
change. Explicit overrides take precedence over constructed state until reset.
Entity/component duplication remaps owned overrides; move transfers them and
removal clears them. Component display-name changes preserve the stable key.

Hosts that want initial overrides set `Template.editable_state=True` (component
recipe templates already enable them) and return a fresh standard Simulation
from their `configure_project(preview=...)` callback. Existing checkpoint-only
preview callbacks remain usable for inspection and reject initial state overrides.
Trusted non-scalar constructor parameters can be named in
`Template.fixed_parameters`, keyed by prototype ID; JSON never imports or replaces
those Python objects. Their components may expose an editable dynamic surface.
Call `Registry.apply_dynamic_states(document, simulation)` after a headless build;
`SimulationServer.set_simulation` applies them when binding an editor run.

SceneTracker is one optional component using this contract. Its `scenes` field
contains standard SceneSpec configuration (including nested SceneTypeSpec), using
runtime entity names. Inspect the field first, edit its JSON, then Save like any
other component. There is no separate scene editor or command language.
A partial record preserves omitted fields; the list replaces the ordered
schedule. Literal premises, optional participant restrictions, action specs and
ISO start times round-trip. An empty possible-participant list keeps standard
SceneTracker's unrestricted semantics. Premises containing Python callables
remain host-configured and are not advertised as JSON-editable. Schedule edits
preserve the memory cursor and reject schedules ending before current progress.
Entity renaming does not infer or rewrite arbitrary references inside component
state; update that component's advertised settings as part of your design.

## Engines and HTML viewer registration

The editor uses the Simulation/Engine contract; it does not instantiate an engine
itself. The host constructs `generic.Simulation(engine=chosen_engine, ...)` and
passes the normal controller and callbacks to `Simulation.play`. Engine identity
comes from that bound instance's checkpoint metadata and is displayed above the
panels. Sequential, Simultaneous and Asynchronous implement the controller callback
contract. The controller tracks each active worker; Simulation completes that
worker’s step after its callback. A pause is acknowledged only after all active
workers finish, including workers finishing their final iteration;
a custom engine must implement that contract for those controls to function.
A checkpoint-only host that omits metadata is visibly reported as not reporting
its engine, rather than being silently labeled Sequential.

`SimulationServer.configure_project` accepts a trusted `viewers` mapping. Values
are HTTP(S) URLs or zero-argument providers returning HTML, up to 5 MB:

```python
server.configure_project(
    registry, document, integrated=True, preview=build_simulation,
    run_with_steps=run_simulation,
    viewers={
        "Dashboard": "https://example.org/simulation-view",
        "Notes": lambda: "<h1>Experiment notes</h1><p>Current results</p>",
    },
)
```

Providers can read application-owned current state and regenerate HTML on
**Refresh viewer**. For the standard social-media example
`examples/social_media/scenario_00_robo_alchemy.py`, obtain its standard
`ForumState` component as the scenario host already does in
`examples/social_media/shared.py`, then register `lambda: forum_state.to_html()`.
This uses exactly the same provider contract as any other HTML: no forum-specific
endpoint, parser or editor controls. No server filesystem paths are accepted.

A local **Open viewer HTML** file is read in the browser. Relative file assets
should be inlined or use reachable URLs; the editor does not serve adjacent local
files. Iframes allow scripts, forms and popups, with an isolated opaque origin
(no same-origin access to the editor). URLs must be HTTP(S) without embedded
credentials. A remote site can refuse framing; browsers do not expose every
cross-origin failure, so the log explains opening that URL separately. Switching
views retains editor draft, selection, run and log, while reloading a view resets
that embedded page's own script state. HTML is fetched only on explicit selection
or refresh; it is not polled or used to start a simulation.


### Prefab creation keys and named presets

`catalog prefabs` lists each creation `key`, its `kind` (`prefab` or `preset`),
role, trusted prefab description and starting parameters. The **Prefab or named
preset** picker uses the same keys. In the roommate example, `add instance minimal
--id Charlie` creates a player entity from minimal; `set Charlie params.name
'"Charlie"'` overrides its name. `duplicate alice` copies Alice's current draft,
including authored components and state overrides.

Hosts opt in with `Template(prefab_prototypes={'minimal': 'alice', ...})`.
Each mapping explicitly binds a registered prefab name to an existing prototype's
editing/reference/component contract. Editable scalar defaults come from the
registered `Config.prefabs[name].params`; registered reference IDs and fixed
constructor values retain their host-owned contract. Fields without compatible
prefab defaults are rejected. For example, mapping `dialogic` to `conversation`
retains the registered next-game-master reference instead of inventing a target
from the prefab's example name. Runtime names are made unique when adding.

Existing preset keys and saved `prototype` IDs remain unchanged; no schema bump
or saved-file migration is needed. Multiple presets can use one prefab: the host
must explicitly select the contract for its prefab creation key. No first-match
inference occurs. A creation key colliding with a preset is a registration error.
Catalog keys and entity IDs are case-sensitive and have distinct meanings.

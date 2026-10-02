# Editor and session commands

The **Simulation log** is the editor's message destination: errors, warnings,
connection changes, command results, actor actions and completion messages appear
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
JavaScript or arbitrary server paths. Quotes group arguments; `set` takes a JSON
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
history. Structural commands call the existing registered prefab/component/scene
helpers. The server validates saves with the same registry and revision checks
as the GUI. Field edits may leave an invalid draft for correction or Undo.

```text
set alice params.goal '"Find music both roommates enjoy"'
set simulation max_steps 40
validate
save
undo
redo
add instance alice
add component constant alice
add group
add scene-type
add scene
duplicate alice
remove INSTANCE_ID
set groups:GROUP_ID participants '["alice", "bob"]'
set scenes:SCENE_ID num_rounds 3
references alice
replace OLD_INSTANCE_ID REPLACEMENT_INSTANCE_ID
move-component components:COMPONENT_ID bob
select bob
select bob SelfPerception
inspect bob
search "Instructions"
view runtime
edit alice Instructions "Alice prefers quiet music."
view definition
panel log
export
load
reload --discard
```

Use stable IDs shown in the inspector. Instances use their bare ID; other records
use `components:ID`, `groups:ID`, `scene_types:ID` or `scenes:ID`. `simulation`
selects the simulation settings. `duplicate` and `remove` without an ID use the
current selection. `set` edits an existing scalar/list record field or
`params.FIELD`; it cannot change identity, prefab registration or schema fields.
Groups/scenes accept `participants` as a JSON list. Component name and settings
use `name` and `params.FIELD`. Existing ownership/type/reference restrictions
apply; a failed action leaves the draft/history intact.

`validate` checks without saving. `save` saves the current draft. `export` exports
the **current local draft**, including unsaved edits; it does not save it.
`load` opens a JSON picker, validates and prepares a local preview, then replaces
the local draft, **without saving to the session**. It refuses to overwrite an
unsaved draft. Save/export it first or explicitly `reload --discard`. Load and
reload reset local undo history. A newly loaded file remains local until Save.

Runtime `inspect` reads the current runtime view. `edit` uses the existing paused
runtime edit operation and supports the same Instructions/Goal text fields as
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

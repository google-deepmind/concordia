# What should the roommates play?

Edit a small Concordia simulation, run it, pause it, and inspect its components
in one graphical editor. The same interface fits an Android portrait viewport
and a desktop's hierarchy, scene, inspector, and log panes.

Alice uses standard `minimal.Entity`; Bob uses `basic.Entity`; their conversation
uses `dialogic_and_dramaturgic.GameMaster` with standard scene tracking. Execution uses `generic.Simulation` and
`sequential.Sequential`. The bundled **NoLanguageModel is a free development
stub**: the trace demonstrates editing and execution, not a realistic discussion
or scientific evidence. The default mode needs no paid account, model download,
or API key. Live model use is an explicit opt-in described below.

## Open the editor

From a source checkout with Python 3.12+ and Concordia installed:

```sh
python -m examples.project_editor.run --port 8080 --output project-run
```

Open `http://127.0.0.1:8080/`. Opening, previewing, editing, and saving do not
execute the simulation. **Run** is explicit. The mock example pauses for one
second after each completed step to give a person time to use the controls;
`--step-delay 0` disables this presentation delay.

For direct local access, omit `--public-origin`. If you choose `--port 8081`,
open `http://127.0.0.1:8081/` instead. `http://localhost:8081/` also works when
opened directly, but hostname and port must match the browser's request origin.

New projects have **40 scene rounds and a 40-step simulation maximum**.
The **Steps to run** numeric box initially requests **10 steps**, or the saved
maximum if it is lower. Choose a whole number from 1 to that maximum before Run.
This is separate from the authored **Simulation maximum steps** field. Standard Sequential
resolves one actor action per step; SceneTracker advances one round per resolved
action, not once per full cast turn. At the default mock pacing, the initial 10-step request gives roughly
10 seconds to observe repeated actions and try Pause/Step. Changing the requested
count changes only the next run; editing and saving the simulation maximum
changes the bound available to future runs. Scene limits can also shorten a run; the game master can also end it earlier. The saved limit
and completion reason appear in the editor and completion is retained in the log.

The Simulation log labels engine-reported **actor actions**, not dialogue or GM
resolution. The mock model returns empty text; standard action formatting can
produce name-only stubs such as `Alice: Alice` or `Bob: Bob`. These are not a
conversation. Card actions are restored from the recorded steps in Current
runtime; Initial definition has an explicit empty state. Log timestamps are the
browser’s local wall-clock time when an entry is displayed, including replay
after reconnect. They are not simulated time or stored event timestamps.

## Select a language model

The default `--model-backend none` uses standard `NoLanguageModel` for Run and
headless execution. For Together AI, select the existing Concordia backend and
an explicit model identifier:

```sh
python -m examples.project_editor.run --port 8080 \
  --model-backend together_ai --model-name "$TOGETHER_MODEL" \
  --step-delay 0 --output together-project-run
```

Set `TOGETHER_MODEL` to a currently available model supported by Concordia's
Together adapter. Install the provider's optional dependency
(`pip install 'gdm-concordia[together]'`) and configure credentials outside the
project. For `together_ai`, this launcher passes the standard Together SDK
`TOGETHER_API_KEY` environment variable through the factory's `api_key`
parameter. When absent or empty, it preserves the adapter's existing
`TOGETHER_AI_API_KEY` environment fallback. If both are set, `TOGETHER_API_KEY`
takes precedence. Credential lookup happens only on explicit execution. Never put
credentials in project JSON or command-line arguments. Model availability and
pricing are provider-dependent; the example does not select a paid model for you.

`--model-backend` (alias `--api-type`) uses
`concordia.contrib.language_models.language_model_setup`; other registered
backends can be selected in the same way. A live backend requires
`--model-name`; supplying a model name with `none` is rejected. Backend imports,
credentials and provider-specific model validation happen only on explicit
**Run** or `--headless`. Unknown backends, missing dependencies/configuration and
exceptions raised during execution are reported through the editor's retained Run failure or the
headless exception; they never silently switch to mock output.

Opening, previewing, editing and saving always build with `NoLanguageModel`,
even with a live backend selected, and never initialize that provider. Each
explicit Run creates a fresh selected adapter. The editor banner and console
identify either **Free mock** or **Live model configured**, including backend
and model name; “configured” does not mean a request has succeeded. Run uses
that model for both actors and the game master. Model settings are local to the
launcher process, so reopening exported JSON requires the same CLI options.
The example retains its constant dummy embedder in either mode.

**Steps to run** is a per-run request, never an edit to the saved definition.
The server validates it against the registered configuration's
`Config.default_max_steps` and passes it to standard
`Simulation.play(max_steps=...)`. Changing a saved maximum does not silently
rewrite a count already entered: correct the count if it is now out of range.
The field is locked while starting, running or paused; Resume continues the
same run and its original target. Reset permits a fresh request. Reopening the
page initializes the next request to 10 (clamped to the saved maximum); an
active run shows its authoritative requested count. Status retains the run's
requested count and maximum separately. Scene termination can still end early.

The launcher does not override the authored maximum. For a short live check,
enter the desired count in the editor before pressing Run. Each engine step can
involve multiple model requests and adapter retries: a step count is **not a
monetary or token cap**. `--step-delay` only paces editor steps and does not limit
model cost.

For an explicit headless execution, add `--headless` to the same command.
Headless execution requests 10 steps, clamped to the saved configuration maximum,
and preserves that maximum in the exported definition.
Verify actual generated actor text and the saved standard log before describing
an execution as a successful live-model run. Mock tests do not establish that.

## Build a project without source editing

New projects use the registered `scene-builder-v2` template (document schema 3).
Choose a minimal actor, basic actor, or scene-aware GM prototype in **Registered
prefab prototype**, then **Add instance**. **Duplicate** copies the selected
instance's authored fields, gives it a new stable ID and unique name, and maps
self-references to the copy. Change its name, Instructions or Goal in the
inspector. An empty Goal omits that optional standard component on the next
build; basic actors retain their standard perception/reasoning architecture.

For a nontrivial project, duplicate Alice, name the new actor Charlie, and give
Charlie distinct instructions and a goal. Add a second scene-aware GM and select it in a scene type’s **Game master** field.
Rename the original GM:
references retain IDs and resolve to its new runtime name when built. Authored
order is retained in JSON and passed to standard Config. The toolbar has no
generic reorder controls.
**Remove** rejects referenced instances and the last actor or GM. Change incoming
references first. The project supports at most 100 instances.

The initial-definition inspector's **Used by** section lists incoming registered
references, group/scene participation and owned components. Select a listed use
to inspect that record. For an actor or GM, choose a **Replacement instance** and
**Replace references** to update all incoming references in one undoable edit.
Only compatible instances are offered; scene types require a scene-aware GM.
Replacing a participant already in a group or scene keeps one occurrence and
preserves the order of the resulting participant list. Component ownership,
component text and other literal fields stay unchanged. You can then remove the
old instance separately; removing it also removes its owned authored components.
**Undo** restores each edit, and **Save draft** validates the complete result.
Groups and scene types offer navigation to their uses; edit those references in
the linked inspector. These controls apply only to authored definitions, never
to a running simulation. Fixed schema-1 templates retain their existing editors.

**Undo/Redo** restores authored edits and structural operations, including IDs
and selection. Typing in one focused field forms one history entry. History
retains up to 50 entries in this browser tab; Save preserves it. Open JSON,
Reload saved, and adopting another tab's saved revision while clean start a new
history. Invalid drafts can be undone. Undo does not override stale revisions
or change runtime state. Search finds names, IDs, prefab names and built
component names. New components appear in the preview after **Save draft**;
preview construction does not run a simulation. **Export JSON** exports the current local draft, including unsaved edits; it
does not save to the session.

If **Save draft** rejects a value, use **Show invalid field** in the error message
to open its inspector and focus the field. Errors applying to a whole record
offer **Show invalid item**. Your draft and the last valid saved definition stay
intact. Navigation is offered only when the error identifies an unambiguous item
in the exact draft you submitted; errors from importing a different file or a
stale saved revision do not point into the current draft. Correct the value and
save again. Runtime errors retain their normal messages.

Select an actor or GM and use **Component catalogue → Add component**. Both actor
prefabs support `constant` context and `recent-observations`; the scene-aware GM
supports `constant`. Select the new component in Hierarchy to edit its display
name, context label, and literal text or observation count (1–1000). The display
name is independent of the stable component ID. **Duplicate** and **Remove**
apply to the selected authored component.
Saved scene, participant, instance and component order is preserved; the toolbar
has no generic reordering controls. **Move component to → Move component**
relocates a configured component
only to a compatible registered actor or GM, keeping its stable ID, display name,
and settings. It becomes the last authored component of its new owner; **Undo**
restores the former owner and order, and **Redo** reapplies the move. Save and
reopen to retain the new ownership. This changes the initial definition, never
an active runtime entity. Duplicating an actor copies its authored components
with fresh IDs. Save
rebuilds the standard components and their acting-context order. Built-in memory,
perception, and GM control components retain their prefab-defined structure;
Instructions and optional Goal remain configured through the owner’s fields.

Scene types select a scene-aware GM and a participant group. Scenes select a
type, participants, round count, and optional literal premise override. Groups
are reusable participant lists. Referenced objects must be unlinked before
removal. These controls construct standard SceneTypeSpec/SceneSpec objects.

The catalogue permits only registered scalar recipes and verified prefab hosts.
Recent observations use the owner’s standard memory; project files cannot
supply imports, Python, constructors, callables, or arbitrary dependencies.
Authored component settings never edit a running entity. Existing
`conversation-v1`/`conversation-v2` schema-1 files and `scene-builder-v1` schema-2
files keep their contracts; no implicit migration occurs. Component composition
is available in the new schema-3 template.

## Try the authoring journey

1. In **Hierarchy**, choose Alice. Change **Instructions · initial text** to
   describe a roommate who prefers quiet music. Set her **Goal** to find a song
   both people enjoy. A nonempty goal builds the standard optional Goal component;
   an empty goal omits it on the next build.
2. Choose Bob and change his goal. Inspect his `SituationPerception`,
   `SelfPerception`, and `PersonBySituation` components. Those standard basic
   prefab components distinguish his architecture from Alice's minimal prefab.
3. Choose **Simulation settings** to change the premise and maximum steps, then
   **Conversation** to change acting order, termination permission, or its name.
   The next-GM selector stores a stable instance reference, so renaming is safe.
4. **Save draft** validates and stores the initial definition in server memory.
   **Export JSON** downloads the saved definition. **Open JSON** validates and
   loads it as a local draft; Save explicitly to update the session. Try reopening
   it in a fresh editor process. Invalid edits retain
   the last saved document and the unsaved fields for correction.
5. Set **Steps to run** (initially 10, bounded by the saved maximum).
   Press **Run**, then **Pause**. The status first says `pausing`; `paused` means
   the engine is actually waiting at its permission boundary. An in-flight step
   can finish before pause is acknowledged.
6. Select **Current runtime** and choose Alice's Instructions or Bob's Goal.
   While paused, change the component's text and press its **Save** button.
   These runtime edits do not alter the saved initial definition. Other component
   values are read-only in this bounded demo.
7. **Step** grants one engine step and returns to paused (or completes the run).
   **Resume** continues. **Simulation** shows the current action; **Log** retains
   the completed-step trace. No second browser tab is required.
8. **Reset** stops the active runner and waits for it to exit, preserving the
   saved initial definition and prior output. Press **Run** for fresh components
   and memory. Changing names, premise, goals in the initial definition, or other
   construction parameters requires finishing/resetting the active run first.

Each run writes its own directory under `--output` with `initial-project.json`,
`log.json`, and `log.html`, using standard `SimulationLog` serialization. Reset
and subsequent runs do not overwrite those artifacts. These are initial
configuration and logs, not restartable checkpoints.

A lost connection disables submission and retains browser drafts. The bounded
server run continues. Reconnection reads authoritative state; it does not
restart the run or replay button presses. A second tab cannot silently overwrite
an older saved revision. **Reload saved** deliberately discards browser drafts.
A process restart requires reopening an exported file; there is no project DB.

## Private Android access

The editor binds only to loopback. Use a private Tailscale Serve HTTPS endpoint
with access restricted to the intended developer. The Android device must be
connected to the same tailnet and permitted to reach that endpoint. Pass the
exact HTTPS origin using `--public-origin` when configuring a proxy.

With `--public-origin`, use that HTTPS address on **both the Mac and Android**.
The startup message prints the browser address, not the proxy's local backend.
Opening the backend's `http://127.0.0.1:PORT/` may display the editor, but Run,
Save, and other mutations are rejected with **Same-origin requests only**.
Use the configured HTTPS address, or restart in local-only mode without
`--public-origin` if remote access is not needed. The scheme, hostname and port
are part of the origin; localhost aliases are not interchangeable origins.
Forwarded headers do not override this check.

Inspect existing Serve routes first and preserve them. Prefer an unused,
dedicated HTTPS port with this application at `/`; the API URLs are root-relative.
Do not mount it under a shared subpath without adding and testing base-path
support. Do not enable Funnel or a public tunnel. The editor grants developer
capabilities to anyone allowed onto its listener: tailnet membership alone is
not a substitute for restricting access to the intended user.

Physical acceptance must include typing with the Android keyboard, selecting
components, exporting/reopening JSON, Run/Pause/Step/Reset, and returning after
backgrounding or losing connectivity. Record the phone and Chrome versions.
Chromium viewport emulation is a separate check, not physical-device evidence.

## Supported definitions and compatibility

`template.registry()` is a Python-owned allowlist. JSON cannot import code
or construct arbitrary Python objects. The saved `schema_version` identifies
the JSON layout; the `template` key selects the registered cast and editable
fields. Fixed-template documents (schema 1) expose parameters on a predefined
cast. Structural documents (schema 2) add trusted prefab prototypes, stable
instance IDs and authored ordering. Component-and-scene documents (schema 3)
also contain registered component recipes and optional scene/group records.
Each format validates the permitted prefab, role and field types.
All formats preserve literal strings, booleans
and safe integers; `max_steps` is 1–1000. Unknown fields, versions, templates,
prototypes, duplicate IDs/names/keys, wrong types, and invalid role references
are rejected atomically. Inspector metadata supplies labels/choices and does
not relax validation. Checkpoint continuation and arbitrary object codecs
remain unsupported.

The integrated listener uses existing `OperationService` operations and its
scope/revision/retry ledger. It delegates to `SimulationServer` project methods
and exposes only the capability routes. It does not combine these with legacy
mutation endpoints. Existing callers can still use the legacy project mode.

## Verification

The following tests build previews and use controlled callbacks; they prohibit
`Simulation.play()` and model calls:

```sh
python -m pytest -n 0 concordia/utils/project_config_test.py \
  concordia/utils/project_operations_test.py \
  concordia/utils/simulation_server_project_test.py \
  examples/project_editor/run_test.py
python -m pytest -n 0 concordia/utils/project_browser_test.py
```

After displaying the exact command and receiving the required launch
acknowledgement, this separate bounded validation **executes real Sequential
runs with NoLanguageModel**, without a network listener:

```sh
python -m examples.project_editor.engine_validation --output engine-validation
```

It checks pause acknowledgement, runtime-only text changes, one-step permission,
resume/completion, Reset, new run identity/components, and retained run artifacts.
The module is intentionally not an automatically collected pytest test.

A headless run of the same saved initial definition is also explicit:

```sh
python -m examples.project_editor.run --project concordia-project.json \
  --headless --output headless-run
```

An application can pass an already configured Concordia model to
`build(config, model=model)`. The default example and engine validation require
no paid provider; live runs use the explicitly selected provider.


## Commands in the Simulation log

Errors, warnings, connection notifications and command results appear in the
**Simulation log**. Validation links live in the relevant log entry. Toolbar
status stays compact; reading older entries preserves the log scroll position.

Type `help` in **Simulation log command**, then Enter. Up/Down recalls commands.
Use `run --steps 10`, `pause`, `step`, `play` (resume), `reset`, `validate`, `save`,
`undo` and `redo`. Authoring commands share the GUI draft and history; for example,
`set alice params.goal '"Find music both roommates enjoy"'`, `duplicate alice`,
`add component constant bob`, `references alice`, `select bob` and `view runtime`.
This is not a shell and cannot run arbitrary code.

`load`/Open JSON validates and previews a local draft without saving it. It refuses
to overwrite unsaved edits. `export`/Export JSON downloads the current draft,
including unsaved edits. `reload --discard` explicitly discards the local draft
and undo history. Runtime edits remain separate from the initial definition.

The example retains the completed standard SimulationLog for
`log overview --source current`, `log actions --source current Alice`, and the
other existing concordia-log analyses. `log import` chooses a structured log;
use `--source imported` explicitly. `log step --source current 1` inspects a
recorded step; `step` advances a paused run. `log dump`/`log bundle` download files.
No browser command accesses arbitrary server paths.

The same language is available through `concordia-session --url URL command
--line 'run --steps 10'`. External CLI authoring uses an explicit local
`--draft JOURNAL.json`, not another browser tab's unsaved state. See
[Editor and session commands](../../concordia/docs/editor-commands.md) for the full
syntax, parity inventory, Node requirement for CLI draft actions and file options.

### Standalone launch with a locally entered Together key

Some credential proxies authorize only the parent session: inheriting a variable
in a child process does not extend that authorization. This optional local prompt
accepts a separately available key without copying credentials from a proxy or
changing its configuration. Environment-based launches remain supported.

If you have your own Together key available, run this command **yourself in a
local terminal**, then enter it at the hidden prompt. Do not enter a key into a
chat, shared tool transcript, command argument, pipe or redirected input.

```sh
TOGETHER_BASE_URL=https://api.together.xyz/v1 python -m examples.project_editor.run \
  --port 8088 --model-backend together_ai \
  --model-name deepseek-ai/DeepSeek-V4.1-Flash --prompt-api-key \
  --step-delay 0 --output project-run
```

Run from the repository root and use an available local port.
Open `http://127.0.0.1:8088/` locally. The endpoint variable above is nonsecret;
no key is assigned to an environment variable. Starting the editor does not
call the model; explicit Run uses the entered key and can incur provider charges.
Preview/edit/save continue to use NoLanguageModel. Default launch without model
flags still uses NoLanguageModel.

`--prompt-api-key` is opt-in and Together-only. Flags/project input are validated
before prompting. Input must be from an actual interactive TTY; getpass warnings
are errors, so it cannot fall back to echoed input. Empty input, EOF or Ctrl-C
stops startup before constructing the editor server. The key stays in process
memory, overrides environment credentials for that process, and is passed to the
standard model factory only on execution. It is excluded from selection repr,
labels, dataclass asdict, project documents and editor snapshots; it is never
persisted by this launcher. Masked-entry and prompted-key initialization errors
use fixed messages. Without this flag, existing environment behavior
is unchanged. The terminal is still a live secret-bearing process; stop it when
you are finished. This is a manual standalone fallback, not an extension of a credential proxy’s authorization.


### Provider failure feedback

Run acceptance is not completion. The Simulation log shows “Run accepted;
waiting for runner output” while work begins, then records completed steps or a
terminal failure. A Together401 fails promptly in the log with authentication
guidance;403 identifies access denial, and request/model errors identify rejected
configuration. Transient errors have bounded retries. Failed/empty provider
responses do not become empty actor actions or advance the step counter.

Credential selection is **prompted key (if explicitly selected), then
TOGETHER_API_KEY, then the adapter's TOGETHER_AI_API_KEY fallback**. TOGETHER_API_KEY
is a correct supported variable. The failure reports the selected source NAME,
never its value or fingerprint. A401 means the provider rejected authentication;
it does not prove a variable name is wrong or that a credential store lacks a
key. Check the key/account/endpoint configuration in your own shell. Restart your
own editor from that same credential-bearing shell after correction. Manual key
prompting is optional and is not required by this diagnostic workflow. There are
no model/authentication probes on page load, preview or save.

For a step-by-step terminal authoring journey, see the
[interactive CLI design tutorial](../../concordia/command_line_interface/README.md),
including the explicit editor/CLI coverage matrix and client-local draft rules.

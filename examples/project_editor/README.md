# What should the roommates play?

Edit a small Concordia simulation, run it, pause it, and inspect its components
in one graphical editor. The same interface fits an Android portrait viewport
and a desktop's hierarchy, scene, inspector, and log panes.

Alice uses standard `minimal.Entity`; Bob uses `basic.Entity`; their conversation
uses `dialogic.GameMaster`. Execution uses `generic.Simulation` and
`sequential.Sequential`. The bundled **NoLanguageModel is a free development
stub**: the trace demonstrates editing and execution, not a realistic discussion
or scientific evidence. No paid account, model download, or API key is needed.

## Open the editor

From a source checkout with Python 3.12+ and Concordia installed:

```sh
python -m examples.project_editor.run --port 8080 --output project-run
```

Open `http://127.0.0.1:8080/`. Opening, previewing, editing, and saving do not
execute the simulation. **Run** is explicit. The mock example pauses for one
second after each completed step to give a person time to use the controls;
`--step-delay 0` disables this presentation delay.

In coordinated sessions, display the exact shell command and obtain the required
launch acknowledgement before any actual simulation, including the validation
command below or pressing Run in a served editor.

## Build a project without source editing

New projects use the registered `scene-builder-v1` template (document schema 2).
Choose a minimal actor, basic actor, or dialogic GM prototype in **Registered
prefab prototype**, then **Add instance**. **Duplicate** copies the selected
instance's authored fields, gives it a new stable ID and unique name, and maps
self-references to the copy. Change its name, Instructions or Goal in the
inspector. An empty Goal omits that optional standard component on the next
build; basic actors retain their standard perception/reasoning architecture.

For a nontrivial project, duplicate Alice, name the new actor Charlie, and give
Charlie distinct instructions and a goal. Add a second dialogic GM and choose
its **Next game master** from the role-safe selector. Rename the original GM:
references retain IDs and resolve to its new runtime name when built. **Move
earlier/later** changes order within the selected role. Order is retained in
JSON and passed to standard Config; it is not merely visual organization.
**Remove** rejects referenced instances and the last actor or GM. Change incoming
references first. The project supports at most 100 instances.

**Undo/Redo** restores authored edits and structural operations, including IDs
and selection. Typing in one focused field forms one history entry. History
retains up to 50 entries in this browser tab; Save preserves it. Open JSON,
Reload saved, and adopting another tab's saved revision while clean start a new
history. Invalid drafts can be undone. Undo does not override stale revisions
or change runtime state. Search finds names, IDs, prefab names and built
component names. New components appear in the preview after **Save draft**;
preview construction does not run a simulation. **Export JSON** exports the
saved definition, so save pending changes first.

Component composition is through these registered prefab parameters, including
optional Goal. Arbitrary component constructors, scene/group object editing and
custom Python imports are not supported. Existing `conversation-v1` and
`conversation-v2` files retain their fixed instance schemas and are never
silently migrated. Structural controls are hidden for those legacy projects.

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
   imports it; try reopening it in a fresh editor process. Invalid edits retain
   the last saved document and the unsaved fields for correction.
5. Press **Run**, then **Pause**. The status first says `pausing`; `paused` means
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

`template.registry()` remains a Python-owned allowlist. JSON cannot import code
or construct arbitrary Python objects. Schema 1 retains the original fixed
instances. Opt-in schema 2 adds a trusted prototype key per instance, allows
stable IDs and authored ordering, and enforces the prototype's exact prefab,
role and scalar field types. Both formats preserve literal strings, booleans
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

An application can supply its already configured Concordia model/embedder by
adapting `build()`. Keep credentials out of project documents. The delivered
example and validation require no paid provider.

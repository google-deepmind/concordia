# Initial-project edit → save → reopen → run

This small example adds **one supported template** to Concordia's existing visual
interface and SimulationServer. Two university roommates discuss music, using the
library's `minimal.Entity` and `dialogic.GameMaster` prefabs. No new engine,
acting policy, component system, or web framework is involved.

The bundled **NoLanguageModel** is a development stub. Its output is not a
plausible conversation, a social-model validation, or evidence about changing
musical tastes. No paid account or model download is required.

## From a fresh source checkout

Use Python 3.12 or newer, in the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m examples.project_editor.run --port 8080 --output project-run
```

Run these commands from the source checkout: the repository packages the
library utilities, not its examples, in wheels.

Open http://127.0.0.1:8080/. The server is loopback-only by default. Opening the
page, editing, importing, or saving does **not** execute the simulation.

1. Select **Alice**. Edit `custom_instructions` or `goal`. Text supports quotes,
   line feeds, Unicode, angle brackets, ampersands, and empty strings.
2. Select **Conversation** (the GM). Change `acting_order` to `random` or
   `game_master_choice`, or change `can_terminate_simulation`. These are the real
   dialogic prefab parameters, not a scripted outcome.
3. Set the **Maximum steps** and initial premise.
4. **Save project** validates and stores this initial draft on the server, then
   downloads `concordia-project.json`. Failed validation leaves the last saved
   draft intact and identifies the invalid field. Unsaved browser fields are
   labelled; save them before Run. Download files are your durable project
   artifacts; this example does not create an automatic project database.
5. **Open project** selects that downloaded JSON. To prove portability, stop the
   server with Ctrl-C, start it again with the same command, then import the file.
   Actor/GM fields, stable IDs, and their supported JSON types survive reopening.
6. Click **Run saved project**. Only the saved revision runs. This builds a fresh
   standard `generic.Simulation`, with a fresh `Sequential`, from exactly the
   reopened initial configuration. Run status is shown in the project view.
   The **Runtime inspector** link opens the existing runtime visualization and
   play/pause/step controls separately.
7. On completion, inspect `project-run/initial-project.json`,
   `project-run/log.json`, and `project-run/log.html`. These use the standard
   SimulationLog APIs. Use a different `--output` directory to retain another
   run's artifacts; the example replaces files in its chosen output directory.

You cannot import, save over, or Run another project while the run callback is
active—even when paused. Finish/resume the active run first. Failed callbacks
are shown and retain the initial document so it can be corrected. Runtime
component edits do **not** change a saved initial project; Save is not a
checkpoint, continuation, or experiment branch. Pause/edit timing follows the
existing server's runtime contract, not a new locking protocol.

The initial view uses stable project IDs for selection. The runtime view retains
the existing runtime renderer and its separately maintained behavior.

## Same project without the editor

```sh
python -m examples.project_editor.run --project concordia-project.json \
  --headless --output headless-run
```

This uses the same `Registry.loads → Registry.to_config → build → Simulation.play`
path. JSON/config equality is not a promise of identical outputs: existing
components may randomize examples or decisions, and hosted models are not
generally deterministic under seeds. No continuation is implied.

To use an already configured real model and embedder, adapt the Python `build`
function; do not put credentials or importable Python names in project files.

## Explicit version-1 support boundary

`template.registry()` is a caller-owned allowlist. It names a trusted Python
factory, the fixed instance IDs `alice`, `bob`, `conversation`, and the supported
reference field. Imported text cannot import Python or instantiate components.

The document contains:

- `schema_version: 1`, `template: "conversation-v1"`;
- `premise` (text), `max_steps` (integer 1–1000);
- `instances`, each with stable `id`, registered `prefab`, fixed `role`, and
  the template's exact initial `params`.

The adapter gets scalar fields/default types from the factory's actual
Config/InstanceConfig values. It does not introspect or serialize arbitrary
constructors. This first contract supports text, booleans, and browser-safe
integers, not floats, nulls, lists, component objects, callables, memory snapshots,
or automatically discovered templates. Browser multiline controls use LF line
endings for edited text.

For `next_game_master_name`, the document value is the stable **target instance
ID** `conversation`, not its display name. The adapter resolves it to the GM's
current name only when constructing Config. Renaming and reordering imported
instances preserve this reference; dangling/wrong-role references are rejected.

Unknown versions/templates/IDs/prefabs/roles/fields, missing or duplicate
instances, duplicate runtime names, invalid types/enums/ranges, malformed JSON,
duplicate object keys, and unsupported object-bearing templates are rejected.
Imports are atomic. Revision checks prevent a stale tab from silently overwriting
another tab's saved draft. This is not full schema metadata, CRUD, undo, checkpoint
support, or a generic exporter for existing object-bearing examples.

## Verification

```sh
python -m pip install -e '.[dev]' pytest
python -m pytest -n 0 concordia/utils/project_config_test.py \
  concordia/utils/simulation_server_project_test.py
python -m pip install playwright
python -m playwright install chromium
python -m pytest -n 0 concordia/utils/project_browser_test.py
```

The optional Chromium tests use the real editor/server and bound standard
components. Their Run callback is **build-only**, guarded against model calls and
Simulation.play; those tests do not prove a simulation ran. The command above is
the separate bounded mock execution recipe.

This supported-template workflow is separate from broader authoring acceptance
for object-bearing philosophy/resource scenarios. It does not claim all
Concordia templates are exportable or that a test count establishes workflow or
scientific success.

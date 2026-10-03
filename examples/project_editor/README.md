# Roommate entity-component editor example

Start with the [GUI and CLI guide](../../concordia/command_line_interface/README.md)
for a complete exercise: inspect Alice's components, create Charlie, add a text
component, edit its state, save and run. The
[command reference](../../concordia/docs/editor-commands.md) covers all controls.

This example registers Alice's standard minimal prefab, Bob's basic prefab and
Conversation's dialogic-and-dramaturgic prefab. The simulation config assigns
Alice and Bob the player entity role and Conversation the game master entity
role. Their chosen components supply the behavior required for those roles.

## Launch

From the repository root in your Python environment:

```sh
python -m examples.project_editor.run --port 8080 --model-backend none
```

Open http://127.0.0.1:8080/ in a browser. The default NoLanguageModel produces
mock responses. Previewing, editing and saving build inspection data; Run starts
the standard simulation. The initial request is 10 steps, capped by the saved
simulation maximum. Adjust **Steps to run** for each run.

The saved maximum is 40 engine steps; **Steps to run** initially requests 10.
Use `--engine sequential` (default) or `--engine simultaneous` to choose the
example's engine. Its actual class is shown as **Engine** in the editor.
Applications can supply another standard `Engine` through `create_editor`'s
trusted `engine_factory` argument; no module names from project JSON are imported.

The engine sends actions and completion/failure state to Simulation log.
Completed runs write `initial-project.json`, standard `log.json`, and portable
`log.html` into a unique subdirectory of `project-run/`. Choose another output
location with `--output`. JSON exports from the editor contain the design;
logs contain execution records and may include prompts or memories.

## Configure a language model

Use the existing Concordia model factory by supplying both backend and model:

```sh
python -m examples.project_editor.run --port 8080 \
  --model-backend together_ai --model-name deepseek-ai/DeepSeek-V4.1-Flash
```

The selected model is constructed when Run or explicit `--headless` execution
begins. The same model serves player and game master entities. Preview continues
to use NoLanguageModel. Labels distinguish mock mode from a configured live
model; actual generated text appears after successful requests. Provider
availability, price and access depend on your account.

Together credential precedence is an explicitly prompted key, then
`TOGETHER_API_KEY`, then the adapter's `TOGETHER_AI_API_KEY` fallback. The launcher
passes the SDK environment variable through the standard factory's `api_key`
parameter. Diagnostics report the source name, never its value. Environment
credentials must be available to the process for its lifetime.

For a standalone local terminal, an optional masked prompt is available:

```sh
TOGETHER_BASE_URL=https://api.together.xyz/v1 \
python -m examples.project_editor.run --port 8088 \
  --model-backend together_ai --model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --prompt-api-key
```

Open http://127.0.0.1:8088/ for that command. Enter the key only in your local
terminal. Prompting requires a real TTY, accepts Together only, and fails before
startup on empty input, EOF or cancellation. The value stays in memory and is
excluded from dataclass repr/asdict, environment assignment and saved project
files. Environment-based launch works without this optional flag.

For this exact Flash model, the standard adapter disables internal reasoning so
short requests can return visible text. Permanent provider failures fail promptly;
known transient errors have bounded retries. Empty visible output raises a clear
error. Simulation log reports failure with sanitized provider guidance and a
credential source name. See the [adapter notes](../../concordia/contrib/language_models/together/README.md).

## Files, controls and integration

**Initial definition** edits the next run's configuration. **Current runtime**
shows the retained runtime. While paused, advertised dynamic component fields change
that runtime; Save draft changes the initial design. Each browser/CLI client owns
its draft, selection and undo history, and publishes through explicit Save.

Use `--project project.json` to open a saved design at launch. File version 4 includes
registered components and `dynamic_states`, the component fields to apply on a
fresh build. Version 3 files require explicit offline recovery; retain originals.
Earlier registered version 1/2 contracts remain supported. The `template` key
selects supported prefabs/fields; `schema_version` identifies the file layout.
The [reference](../../concordia/docs/editor-commands.md#file-versions-and-recovery)
explains validation and recovery.

`--headless` explicitly executes the saved design without a web editor, using the
same model selection and default requested step limit. `--step-delay` controls
example pacing after completed steps. `--public-origin` declares the browser's
origin when using an existing proxy. Choose it to match the actual browser URL;
origin validation remains enabled. Run `python -m examples.project_editor.run
--help` for all launcher arguments.

The example reuses `SimulationServer`, `ProjectEditor`, the registered component
catalog, `Simulation`, and its standard step controller. The GUI and
`concordia-session` dispatch the same operations. Node.js runs the shared draft
helpers for external CLI authoring; the browser executes those helpers directly.

## Verification

Tests construct registered prefabs and intercept operation/browser traffic while
guarding against model calls, simulation execution and listener startup. The guide
has a copied-command regression and a graphical authoring journey. Desktop and
mobile viewport checks are separate from physical Android testing. Use the actual
device to assess touch behavior before making device-specific claims.

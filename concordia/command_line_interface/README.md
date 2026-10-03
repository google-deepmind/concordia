# Build simulations with entities and components

A simulation is built from entities. Each entity has components that hold state
and contribute behavior. An entity is a container for these components, much like
a Unity GameObject. You can change its behavior by changing component settings or
adding a component, then see the result when the simulation runs.

This guide uses Alice and Bob discussing music in a shared kitchen. You can design
it in the graphical editor (GUI), in the editor's command input, or from an
operating-system terminal with `concordia-session`.

## Understand what you are editing

**Parameters** are settings used when constructing that component or entity.
A **prefab** is a reusable starting configuration: it chooses components and how
they work together.

| Example entity | Components chosen by its prefab | What they contribute |
|---|---|---|
| Alice | Instructions, Goal, recent observations, memory | Instructions describe Alice; Goal supplies her objective; observations and memory provide context for her next action. Goal is included when its initial text is nonempty. |
| Bob | Instructions, Goal, memory, plus SelfPerception and other perception components | Bob's basic prefab adds questions about himself and the situation when preparing an action. |
| Conversation | Instructions, MakeObservation, next-acting and event-resolution components | These prepare what players observe, choose whose turn comes next, and resolve proposed actions into events. |

**Observation** means information an entity receives about its situation. It can
be stored in memory and used to prepare a later action. A constant-text component
adds text to the context used for an action; it does not create a new turn.

The **engine** organizes the steps. Its actual class is shown prominently as
**Engine** in the toolbar; the host application chooses it when building the
simulation. The example uses Sequential; another configuration can use
Simultaneous or another implementation of Concordia’s engine contract. In this example it asks a **game master
entity** (Conversation) for observations and whose turn is next, asks a **player
entity** (Alice or Bob) to act, and asks the game master entity to resolve that
action. These roles are assignments for an engine's `run_loop` call, not different
intrinsic entity classes. “Player” here can be model-controlled; it does not mean
a person must type each action. The example's prefab configuration supplies the
appropriate components for each assigned role.

## Open the example

Install Concordia in a Python environment from a checkout containing the
[project editor example](../../examples/project_editor/README.md):

```sh
python -m pip install -e .
python -m examples.project_editor.run --port 8080 --model-backend none
```

Open **http://127.0.0.1:8080/** in a browser. Use another free port if needed.
The launcher opens an editor service; Run starts the simulation. The default
**NoLanguageModel** mode uses mock responses so you can explore the controls.
A launcher configured with a live language model generates provider text when
Run is pressed. [Example launch options](../../examples/project_editor/README.md)
explain how to select a model. The same design tools work in both modes.

The editor has three main views and a log:

- **Hierarchy** lists player entities, game master entities and their components.
- **Inspector** shows the selected entity's initial parameters and components.
- **Simulation** shows the entity diagram, actual engine, and current step.
  The **Central viewer** selector can show another HTML view of the same session.
- **Simulation log** shows actions, progress, errors and command results.

On narrow screens, use the panel tabs. On a desktop, drag the panel boundaries to
make space; keyboard users can focus a separator and use arrow keys.

## Commands in the browser and terminal

Lines in `text` blocks below go in **Simulation log command**, or at the
`concordia>` prompt described next. Lines in `sh` blocks go in your operating-system
terminal. Commands such as `set` take JSON values: `'"text"'` keeps the JSON string
as one argument, while `10`, `true`, `null` and `["alice","bob"]` are other JSON
values. Stable IDs such as `alice` identify records even if you change their names.

For a separate interactive terminal, install Node.js for the shared draft editing
helpers, then attach to the editor service:

```sh
node --version
concordia-session --url http://127.0.0.1:8080 interactive --draft my-design.json
```

`my-design.json` is this terminal client's working journal. GUI and terminal each
have their own draft and undo history. **Save draft** or `save` publishes that
client's design to the shared session; use **Reload saved** or `reload --discard`
in the other client when ready to adopt it. Follow either journey below in a fresh
draft rather than creating Charlie twice in the same draft.

## Journey 1: build a behavior in the GUI

1. Select **Alice** in Hierarchy. In **Initial definition**, expand **Instructions**
   under Components and read the text. Alice initially has no Goal component:
   her initial goal is empty.
2. Enter `Find music everyone enjoys` in **Goal · initial text (empty removes the
   optional component)** and press **Save draft**. Expand the newly added **Goal**
   component to see that text. The prefab has built a component from your setting.
3. Choose Alice's minimal prefab in **Registered prefab prototype** and press
   **Add instance**. The new entity is selected and has its own generated ID.
4. Set **Entity name** to `Charlie` and **Instructions · initial text** to
   `Charlie listens carefully.` in the Inspector.
5. Under **Component catalogue**, choose **constant** and press **Add component**.
   Select the new component in Hierarchy. Set its **Context text** to
   `Listen before replying.` and its **Context label** to `Reminder`.
6. Press **Save draft**. Select Charlie and expand the new component. Its saved
   text is now part of the context his prefab uses to prepare actions. A live
   model may use that reminder when deciding what Charlie says or does.
7. Set **Steps to run** to 10 and press **Run**. Read the Simulation log as steps
   arrive. Mock mode demonstrates the execution flow with mock output; live mode
   produces model-generated actions.
8. Press **Pause** and wait for the status to become **Paused**. Press **Step**
   to advance one engine step, or **Resume** to continue. If the run already
   finished, start another run when ready.

**Undo** and **Redo** work on draft edits. Select Charlie to inspect his record.

## Journey 2: build the same behavior from commands

Start with the example's original saved definition, which contains the stable IDs
`alice`, `bob` and `conversation`. Discover the
available starting configurations and component settings:

```text
catalog prefabs
catalog components
list instances
inspect alice
```

These commands create named IDs so you can copy the exercise. `params.FIELD` means a field inside that record's initial parameters.
`components:reminder` selects a component record; the prefix distinguishes it
from an entity.

```text
set alice params.goal '"Find music everyone enjoys"'
add instance alice --id charlie
set charlie params.name '"Charlie"'
set charlie params.custom_instructions '"Charlie listens carefully."'
add component constant charlie --id reminder
set components:reminder params.state '"Listen before replying."'
set components:reminder params.pre_act_label '"Reminder"'
set simulation max_steps 40
validate
save
inspect charlie
```

Use `run --steps 10` when ready, then `state` to see progress. `pause`, `step` and
`play` correspond to Pause, Step and Resume. Run reports acceptance first;
completion or failure appears in subsequent state/events. `watch` shows one live
event stream; Ctrl-C returns to the prompt. `reset` ends the current run through
its controller so you can start again from the saved definition.

## Change component state and choose a viewer

Components advertise editable fields in **DYNAMIC** rows. Text fields accept text;
structured fields accept JSON, such as a list in square brackets. **JSON value**
lets you enter a typed value such as `null` when the component supports it. In **Initial
definition**, the field's **Save** button places an override in your draft; then
**Save draft** validates it on a fresh preview. The same change from a command is:

```text
state-field alice Instructions state '"Alice asks before choosing a song."'
validate
save
inspect alice Instructions
```

This overrides the component's constructed state for future runs. Prefab
parameters still determine which components exist: giving Alice an initial goal
creates Goal, while editing a dynamic field changes an existing component.
Use **Reset to prefab** or `state-reset alice Instructions state` to remove an
override and use the prefab parameter again. Reset an optional component’s
overrides before clearing the parameter that creates it.

Choose **Central viewer** to switch between **Default visualization** and HTML
views registered by the application. **Open viewer HTML** loads your own local
HTML file; **Viewer URL** and **Open viewer URL** embed an HTTP(S) page. You can
switch back at any time; your draft, selection, Inspector and log remain in place.
**Refresh viewer** fetches the registered provider again or reloads the page.
HTML scripts run in an isolated frame and can provide their own controls.

`viewer` lists available views; `viewer NAME`, `viewer default` and
`viewer-refresh` operate this browser's central panel. In a separate terminal,
`viewer NAME` retrieves that registered view, and `--file view.html` exports its
HTML. Terminal clients inspect data and files; they do not render a diagram or
change another browser's selection. The [reference](../docs/editor-commands.md)
shows how an application registers any HTML provider, including the standard
social-media forum's output.

**Simulation maximum steps** caps a run; **Steps to run** chooses this run's length
within that saved maximum. A step is defined by the selected engine.

## Inspect, save and reopen your work

Use **Current runtime** to inspect the entities built for the current or retained
run. While paused, **Save** on an advertised dynamic field changes that runtime
through the component's own state setter. **Initial definition** returns to the design used to build the next run.
For example: `view runtime`, `inspect alice Instructions`, then
`edit alice Instructions state '"Alice asks before choosing a song."'` while paused.

**Export JSON** downloads your current design; **Open JSON** loads a design for
editing. In the interactive terminal use `export project.json` and
`load project.json`, followed by `save` when you want to publish it. Exported
project JSON is separate from the terminal's working journal.

After a run finishes, use `log overview --source current` or
`log actions --source current Alice` to explore its structured log. In the
terminal, `log export --source current --output run.json` saves a reusable log;
`log bundle --source current --output viewer.html` creates a portable viewer.
Browser versions download these files.

## When something needs attention

- **Invalid draft:** read the Simulation log and use **Show invalid field/item**;
  correct the field or Undo, then Save again. `validate` checks a CLI draft.
- **Another client saved:** export your edits, reload the saved definition, then
  reapply the changes you want. A stale revision protects both designs.
- **Pause pending:** an in-progress engine step finishes before editing is enabled.
- **Disconnected:** your draft stays local; reconnect and inspect state before retrying.
- **Older file version:** version 3 needs explicit offline conversion to version 4;
  keep the original file. See [file recovery and command reference](../docs/editor-commands.md).

Use `help` for commands, `files` for terminal file operations, and `exit` or EOF to
leave the prompt with your journal retained. History stays in memory during the
session. The [command reference and GUI/CLI coverage table](../docs/editor-commands.md)
cover discovery, replacement, ordering, log analysis and detailed client behavior.

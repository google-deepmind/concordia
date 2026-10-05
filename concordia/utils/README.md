<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Concordia utilities

## Portable structured log viewer

The repository includes an interactive HTML viewer for structured simulation
logs. To combine a structured JSON log and the viewer into one portable file,
run:

```shell
concordia-log bundle simulation_structured.json
```

This writes `simulation_structured_viewer.html` beside the input file. Open the
HTML file in any modern browser; it does not need a server or the original JSON
file.

Choose a different output path with `--output`:

```shell
concordia-log bundle simulation_structured.json \
  --output reports/simulation.html
```

The resulting file contains the complete structured log, including component
data, prompts, memories, content references, and inline images. Be careful when
sharing it: private model context contained in the JSON is also contained in the
HTML.

To browse a log without creating a new file, open `log_viewer.html` and select
the structured JSON file using the file picker.

## Editor session commands

The integrated editor and `concordia-session` share a command language,
operation validation and registered authoring actions. Use `concordia-session
--url URL interactive --draft design.json` for a persistent prompt, or `command
--line` for one command. In either prompt, `add instance minimal --id Charlie`
creates an entity when the host registers the minimal prefab. See
[Editor and session commands](../docs/editor-commands.md) for commands, local
draft/history semantics, structured-log analysis and client-side exports.

For a complete interactive authoring walkthrough and explicit editor/CLI
coverage, see the
[GUI and CLI entity-component guide](../command_line_interface/README.md).

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

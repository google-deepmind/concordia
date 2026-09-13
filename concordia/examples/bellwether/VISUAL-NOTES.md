# Optional local visual notes

This is **model image input**, separate from the public SVG presentation and
browser speech controls. The model call is an explicit local command-line aid, not a player
image-upload endpoint. The result is a draft for you to inspect and edit, never an
automatic observation, game action, consent decision, or source of inventory.

## Use an installed local vision model

From this contribution's source checkout, use the existing Ollama service.
Model setup is separate; the tool never downloads a model implicitly:

```sh
ollama pull qwen3-vl:2b
python -m concordia.examples.bellwether.visual_note \
  --model qwen3-vl:2b --image public-diagram.png \
  --question "Describe the visible objects and their positions." \
  --output visual-note.json
```

Review the JSON's `text` alongside the original image. The draft includes the
chosen model, source/transmitted hashes and dimensions, stop reason, output
limit, and private request timings. It is always labelled
`required_before_use`. A `length` stop reason indicates an incomplete answer.
An existing output file is not overwritten. To use any text in a game, explicitly
copy and edit it in your own pending command, check the interpretation, and send
it yourself. The tool has no game/session connection and cannot send it for you.

Only PNG, JPEG and WebP **still images** are supported. Input is limited to
8 MiB and four megapixels; Pillow applies orientation, composites transparency
on white, reduces the longest edge to at most 1024 pixels, and reencodes PNG
without EXIF, comments or other metadata. Small writing/details can be lost on
resize. SVG, animations, PDF, URLs, camera capture and audio are not accepted.
The existing Matplotlib dependency supplies Pillow; no new browser framework
or cloud service is needed.

The tool sends the normalized pixels and your question only to
`127.0.0.1:11434`, ignoring proxy settings. A model must report vision support
before pixels are sent. Requests use a 90-second HTTP timeout and 256 output-token
limit, do not retry automatically, and ask Ollama to unload this model afterward.
The standard profiler records counts/latency, not pixels or question text.
The JSON note **does** retain the question and model description: it may contain
private information and is not the sanitized public-account export. It omits
the image bytes, original filename and local file path.

## Evidence and interpretation

Unit checks use synthetic pixels and a mocked SDK. They verify decoding/limits,
EXIF orientation and metadata removal, actual normalized pixel payloads, capability
checks, request bounds, plain-text/literal JSON, and no output on transport or
validation failure. No simulation is executed by these checks.

A separate local probe uses two public synthetic diagrams with identical
questions and reversed positions of a red circle and blue square. Record the
actual descriptions, mismatches and timing when trying this on a model. Passing
those examples only establishes a small image-conditioned functional result,
not general recognition/OCR accuracy or reliability on scientific figures.
Do not infer perception quality merely from the model's capability flag.

One local `qwen3-vl:2b` run described both synthetic diagrams correctly with
the positions reversed, using the identical question. Each request took about
16.2 seconds on that machine, including model loading; the model was unloaded
between requests. These two observations do not establish OCR/diagram accuracy
on other images or a latency guarantee. No game operations or simulation steps
were executed.

Image text and model text are untrusted data. The note can be mistaken or contain
instructions; nothing here executes them. Review against the image before any
reuse. No broad image/audio understanding, physical Android image workflow,
automatic actor perception, or social validity follows from this tool.
For text-only play or unsupported models, keep using the normal player input.

## Review a note in the player page

On your own pending turn, open **Review an image note · optional** and choose
the generated visual-note JSON. The native file reader reads it in your browser:
the file, image, question and source metadata are not uploaded to the game.
The page verifies the bounded file structure, not the truth of its contents or
its claimed model/source. Source details are available separately from the
editable description.

Edit the description, then choose **Add reviewed text to action draft**. This
appends the literal text without replacing your existing draft or submitting it.
Review the complete command and, for a resident role, your selected decision.
Only the existing **Send action** submits anything. Long text may need trimming
to satisfy the existing action input limit; the importer never silently truncates.

Invalid imports preserve your draft and the previous edited review. Clear review
cancels outstanding file reads and leaves your action draft unchanged. A newer
file, role/turn change, connection loss or reload discards old review data and
late responses. Only text you explicitly added to the action draft uses the
existing role-scoped draft recovery. Spectators and browsers without a pending
controlled turn cannot use the review controls.

Chromium checks cover narrow portrait/landscape, unchanged-SSE focus/selection,
UTF-8/format/size errors, cancel/replacement of delayed reads, approved Nell's
private review, revocation, reconnect and explicit Send. These are emulated
browser checks with synthetic inboxes and no simulation or live image calls,
not physical Android validation. Image selection/inference from the phone is
not implemented by this JSON review step.

# A portable public account

In a full Bellwether night, **Download readable account** saves an offline HTML
page; **Download public JSON** saves the same public material in a versioned
machine-readable document. These controls never advance a turn or spend a choice.
They work during a night and at dawn. An interrupted run is labelled interrupted;
an unfinished night is not labelled complete.

The export deliberately has fewer fields than the developer trace or a player's
private journal. It is **not a saved game, checkpoint or executable replay**.
Do not use it to resume a simulation or infer hidden decisions.

## Public scope, not anonymization

The server reuses `StormNight.view('spectator')`, then selects an explicit
allowlist of public event text, watch, kind, material inventory, service
consequences and a small dawn summary. Public events are numbered locally in the
export; hidden event identifiers and recipients are omitted. Component state,
private messages, pending action IDs, actor contexts, service identity, cookies,
hostnames and timing metadata are not exported.

The same operation returns the same public bytes for a player, an approved
resident, an approved spectator or the local developer. Visitors and revoked
browser sessions cannot call it. There is no option to request another player's
private account. The public timeline can still include a participant's names or
anything they said publicly: review before sharing. Visibility filtering is not
anonymization, and the underlying model may choose to disclose information in
public speech. No automatic sharing/upload happens.

HTML is self-contained UTF-8 with escaped text and no JavaScript, external
resources, forms, frames or embedded live service data. It can be opened offline.
The UI retains your current draft when a download fails, and refuses a delayed
download after a connection/role change.

## Attached CLI and research tooling

Use the existing CLI against the **trusted local developer listener** of a
full-night service built from this checkout, not the one-turn slice. Do not expose
that listener to other players. With the service already running:

```sh
printf '%s\n' '{"operation":"game.public_account","arguments":{"format":"json"}}' |
  python -m concordia.command_line_interface.concordia_session --url http://127.0.0.1:8784 call > public-response.json
python -c 'import json,pathlib; x=json.loads(pathlib.Path("public-response.json").read_text()); pathlib.Path("bellwether-public-account.json").write_text(x["result"]["content"], encoding="utf-8")'
```

Choose `html` instead to obtain the readable artifact, or `svg` for the
public service figure. The shared operation's
`result` has `filename`, `media_type` and `content`; save **only content**.
The ordinary operation envelope contains process references for the attached
transport, and is not the shareable artifact. No mutation revision or retry key
is needed, and unknown formats are rejected with no effects.

The document schema is `bellwether-public-account/v1`. It records fixture/live
mode, in-progress/interrupted/completed status, public events and material
consequences. Export order is public event order, not wall-clock or causal
evidence. Multiple exports of an unchanged state are identical. Account numbers
are local display order, not references into private traces. New versions should
change the schema name before changing this contract.

## Evidence and limits

Tests use real components, the standard HTTP/operation/CLI path and actual
Chromium downloads, but **seeded records with simulation execution prohibited**.
They verify cross-role equality, private marker absence, error atomicity,
read-only state, offline literal markup and mobile width. They are not human
usability evidence or an additional live game. Prior fixture/live night evidence
remains separately documented.

The material ledger reports scenario rules; actor prose is not empirical
validation of real societies. An exported account neither certifies the truth of
the previous-storm dispute nor makes hosted model output deterministic. Full
checkpoints, intervention provenance and isolated continuation need separate
contracts before experimental branch comparison.


## A portable public figure

**Download public figure** produces a self-contained SVG through the same
`game.public_account` operation with `format: "svg"`. It uses the existing
Matplotlib dependency and the existing public document projection, not private
logs or screenshots of an authenticated page. No new service, provider upload,
model call or user-data import is involved.

The figure shows watch-by-facility **Served**, **Unserved** and **Not resolved**
states in words as well as colors, plus current recorded fuel stocks and spare
part holders. **Not resolved** is not a prediction; a request or promise does
not count as supplied service. Used stock remains cumulative, including any
explicit pre-play consumption. The image is an accounting figure, not a
geographical map or a model-generated depiction of historical events.

Its SVG title and description provide an accessible text account of every cell
and stock, with source schema, backend, run status and watch. The same provenance
is visible on the figure. Public dialogue is deliberately omitted from the
image—even though it is available in the HTML/JSON timeline. A snapshot marked
fixture or interrupted must retain that label when shared. Do not crop away
provenance or treat the figure as empirical warrant for a social conclusion.

SVG has no scripts, forms, external fonts/resources or live service link. Internal
clip/glyph references and any embedded PNG grid are self-contained. Open it in a
browser or a compatible image viewer; zoom preserves vector text/axes. Plotting
work uses owned Figure instances and a renderer lock, without changing unrelated
pyplot/global settings. Internal IDs are normalized and generation dates omitted
for repeatable exports from the same state/software/fonts; this does not promise
byte-identical rendering across plotting-library or font versions.

Real Chromium tests download through the player, compare the existing CLI result,
open the image offline in narrow portrait/landscape and verify descriptions,
private-data exclusion, unchanged drafts and no game effects. The screenshot
captures the SVG element itself: Chromium's full-document capture stalled for
standalone SVG in this environment, while native element capture and display
worked. No physical phone, assistive-technology study or played simulation is
claimed by these checks.

This is **presentation multimodality**, separate from image/audio understanding
by a model. The inspected installed local chat models report completion/tools
(and, for Qwen, thinking), not vision/audio input support. This feature neither
changes Concordia's text model contract nor silently supplies an image to one.

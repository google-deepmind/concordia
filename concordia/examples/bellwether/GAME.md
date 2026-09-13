# A night at Bellwether

You are the emergency coordinator of a small island community. The game runs
from **Dusk → High Tide → Before Dawn → dawn**, with four consequential choices
per watch. Looking at the coastal map and resident cards is free. A refused
request may cost a choice; an ambiguous command does not.

This is a fictional interactive scenario demonstrating standard Concordia
composition, not a model validated against human behavior. The original
one-turn `--mode slice` remains available; it is not the full game.

## Start from this contribution's source checkout

```sh
python -m pip install -e '.[dev]'
python -m concordia.examples.bellwether.run --mode fixture --editor-port 8786 --player-port 8787 --output runs/bellwether-fixture
```

Open <http://127.0.0.1:8787> to play and <http://127.0.0.1:8786> for the
developer editor. **Begin the night** (or the editor's `run.start`) starts
exactly one run. Reopening a tab does not restart it. The fixture banner denotes
deliberately cooperative, programmed responses, never a live model.

For live residents, first make an existing local Ollama model available:

```sh
python -m concordia.examples.bellwether.run --mode live --model llama3.2:3b --resident-prefab minimal --editor-port 8786 --player-port 8787 --output runs/bellwether-live
```

No paid backend is required. This command uses the existing Ollama adapter;
requests have a 90-second HTTP timeout, 256 output-token cap, and a 256-call
wrapper limit. The standard per-run profiler records timings and token
**estimates**, not measured billing. Exhausted/invalid output cannot grant
consent. Model failures end the run with the retained journal and developer
error; there is no hidden fixture fallback. Use Ctrl-C to close the owned
listeners. A new process starts a new night; checkpoint continuation is not
implemented by this example.

The standard `basic` prefab is an alternative via `--resident-prefab basic`.
Its perception components make additional model calls. Four residents share the
selected architecture but have separate standard memory, observations, goals,
accounts and affiliations. Developers can configure their individual instance
params in `game_prefab.configuration`. The human uses HumanAct, not an LLM.
The GM uses independently composed SwitchAct policies and explicit scenario
rules, not the resident decision logic.

## Optional local memory embeddings

The compatibility default is an eight-dimensional **constant placeholder**.
It permits fixture plumbing but does not provide meaningful semantic retrieval,
including when live residents use `basic`. To use real local embeddings, install
an embedding-capable model separately in your existing Ollama service:

```sh
ollama pull all-minilm
python -m concordia.examples.bellwether.run --mode live --model llama3.2:3b --resident-prefab basic --embedding-model all-minilm --output runs/bellwether-embeddings
```

The optional adapter feeds the standard `generic.Simulation` callable-embedder
interface and existing `AssociativeMemoryBank`; it does not replace retrieval.
It uses only `127.0.0.1:11434`, does not inherit proxy settings, and makes no
automatic model downloads. Capability checks happen before memory transmission.
Requests use a 30-second HTTP timeout and reject truncation. Vectors must be
finite, nonzero, and dimension-consistent; they are normalized because the
standard bank scores with a dot product. A failed request is surfaced, never
silently changed to a constant embedding.

Developer state and `outcome.json` identify constant, provided-callable, or
local-model embeddings. The existing private profiler records request counts,
failures, timings, and dimensions—not memory text or vectors. This metadata is
not added to player views. Model dimensions, vocabulary and latency vary;
`all-minilm` has a short input context and can reject long memories. Choose an
appropriate installed model for your text rather than silently truncating it.
This is not a quality benchmark or a guarantee that a chosen actor architecture
will use associative retrieval. Do not mix a saved bank's vectors with a different
model or placeholder; checkpoint compatibility/continuation is not implemented.

Trusted Python callers can instead pass `embedder=callable` to `Game` or
`SharedGame`, using their existing embedding library. No web configuration can
load arbitrary Python callables. Slice mode keeps its original fixture default.

## Playing

Select suggestions to fill the input, or enter a command and your own words.
There is deliberately a small, conservative physical-action vocabulary.
Unrecognized or ambiguous physical instructions ask for clarification **before**
waking the pending human turn. The parser is not a general natural-language
physical simulator.

| Intent | Example |
|---|---|
| Public conversation / open-ended proposal | `tell everyone: Can we find a plan that protects boats and families?` |
| Address one resident publicly | `tell Mara: What worries you about the cooperative?` |
| Private conversation | `message Nell: I want to hear your account.` |
| Ask for material consent | `ask Nell for fuel and part` |
| Ask for labor consent | `ask Ivo to repair` |
| Collect the agreed resources | `transfer reserve and part` |
| Ask Ivo to carry out accepted work | `order repair` |
| Allocate this watch's supply | `allocate all` or `allocate shelter and cold store` |
| Make your own promise | `promise shelter` |
| Withdraw your latest unfulfilled promise | `withdraw my promise` |
| Advance without agreement | `wait` |

`propose to everyone: …` preserves creative prose for resident discussion.
It cannot invent a new action mechanic or bypass resource/consent checks.
Residents respond in their own voices. An explicit structured `accept` records
only that resident's requested commitment; spoken claims alone are not consent.
Residents may decline, counter, or revoke their own unfulfilled commitment.
The next proposal can take their response into account; acceptance is not forced.

Nell's material commitment releases two fuel and the spare part. Collecting them
is a separate action. Ivo's labor commitment is distinct from performing the
repair: he may still refuse the work order. Repair requires accepted Ivo labor,
the delivered part, and actual performance **before Before Dawn**. A late repair
does not rewrite already-resolved services. General item trades and arbitrary
new engineering procedures are intentionally not silently adjudicated as valid.

## Material rules and two strategies

The generator has **6 fuel**, Nell has **2 reserve**, and there is **1 spare
part**. The beacon, shelter and cold store demand one fuel in each of three
watches: baseline demand 9. A completed repair removes only the final beacon
demand. The standard Inventory tracks Generator, Nell, Ivo and Used accounts;
fuel and the part never appear from narration. Requested allocation order
determines supply priority if there is insufficient fuel. Consumption occurs
at watch boundaries, after that choice's bounded resident responses.

Two useful plans, not guaranteed live social outcomes:

1. **Negotiate full service:** in Dusk ask Nell for fuel/part, collect them,
   ask Ivo to repair, then wait. In High Tide order the accepted repair;
   leave all facilities allocated and use the other choices for promises or
   discussion. Keep all allocated Before Dawn. With Nell's acceptance and
   Ivo's accepted **and performed** work, 8 fuel supplies all 9 facility-watches.
2. **Prioritize shelter and livelihoods:** allocate shelter and cold store in
   each watch. This consumes 6 fuel, leaves Nell's reserve untouched, but
   records three missed beacon watches and the associated harbor consequences.

The fixture demonstrates both. Live residents can defeat either social plan
through refusal or changed commitments. Waiting always allows progression to
dawn even without agreement. Dawn shows actual service, fuel, repair,
honored/broken/revoked/unfulfilled commitments, and each resident's response.
Missed beacon service delays safe arrivals; missed shelter service removes
heating; missed cold-store service risks stock. These are explicit fictional
scenario consequences, not predicted casualties or economic estimates.

At High Tide a **disputed** account of the previous storm is delivered to its
configured recipients. It does not establish objective truth or rewrite
memory. Use `--dispute-file account.json`:

```json
{"text": "Mara says the cooperative did not help; Nell disputes that account.", "recipients": ["Coordinator", "Mara", "Nell"]}
```

The object is validated before building. Institutions distinguish charter,
membership, enforcement and who knows about the disputed account. The default
delegates reserve custody to Nell and labor consent to Ivo; membership does not
grant someone else's consent. Interaction records describe changing positions
without imposing scalar trust or a single psychological theory.

## One service, three clients

The existing `OperationService`, `SimulationServer` and attached
`concordia-session` CLI remain the authority. The player GUI uses the same
`human.respond` handler available on its listener. Developer GUI/CLI edits use
the same designated `component.edit` as [the original slice](README.md).

Additional operations:

| Operation | Audience | Meaning |
|---|---|---|
| `game.begin` | player | Explicit one-time start |
| `game.preview` | player/developer | Parse an attempt without effects |
| `run.resume` | developer | Resume the existing paused worker; never replay |

```sh
concordia-session --url http://127.0.0.1:8786 discover
concordia-session --url http://127.0.0.1:8786 state
concordia-session --url http://127.0.0.1:8786 watch
concordia-session --url http://127.0.0.1:8787 state
```

A mutation uses the discovered arguments plus the current references/revision
and an exact retry key, as in README.md. Retry the identical request after a
lost response; stale edits fail atomically. Selecting Pause does **not** permit
editing while a worker or human request is still active. The designated edit is
safe before Run or after the worker has joined. Initial configuration remains
distinct from runtime. This is not a general mid-run transaction or checkpoint
interface.

Player HTML/JSON/SSE contain only public facts, the coordinator's delivered
observations and messages they participated in. The standard observation queue
separately delivers private events to the relevant residents. Developer
checkpoints and traces stay on the trusted editor listener; they are never
hidden in player HTML. Both listeners bind to loopback. This example changes no
routes, account configuration, lobbies or access controls.

## Verification and limits

```sh
python -m pip install pytest playwright
python -m playwright install chromium
python -m pytest -n 0 concordia/examples/bellwether concordia/contrib/language_models/ollama/ollama_model_test.py
```

Rule tests exercise conservation, both strategies, refusal, commitment
revocation, late/missing-part repairs, malformed resident output, atomic
clarification, recipient privacy and fresh ownership. Chromium tests use real
components, HTTP, SSE and attached CLI with a synthetic pending input and
explicitly forbid simulation execution.

Separate browser walkthroughs executed both complete fixture strategies and a
local live night through standard Simulation/Sequential: 12 human choices each,
four dawn responses, reload/reconnect recovery, matching CLI snapshots and no
JavaScript errors. The live run used 12 successful local llama3.2:3b calls
(about 2.9–6.0 seconds each); Nell declined and no reserve transfer or repair
was fabricated. Live timing is one machine/run observation, not a guarantee.
Automated browser submission timing is not a human play-duration measurement.

Logs are standard `simulation.json` and `log.html`; `outcome.json` adds
the explicit night result, per-step timings, model profile and backend label.
The 15–25 minute human experience is a target, **not yet measured**. The full
Bellwether editor/showcase A–H, isolated checkpoint branches, experiments,
authoring assets/undo/breakpoints, every editor-family parity and fresh
environment export remain subsequent work. No original workflow acceptance
baseline is upgraded merely because this bounded game now reaches dawn.

### Standard basic extensions

The optional basic residents retain their standard perception chain and normal
ConcatAct policy. `basic.Entity` now supports `extra_components` and
`extra_components_index` using the **same extracted assembly helper** as
`minimal.Entity`; basic defaults and minimal insertion semantics are unchanged.
This supplies each resident’s account, affiliations, and scenario instructions
without copying the basic prefab or silently ignoring configuration. Default
and injected-policy regressions accompany both advertised configurations.


## Two human roles

See [MULTIPLAYER.md](MULTIPLAYER.md) for optional host-approved Coordinator/Nell
play, private role journals, Android controls, and the HTTPS boundary. Omit
`--multiplayer` for the unchanged single-human game.

## Optional voice

[Local voice controls](VOICE.md) add opt-in spoken observations and editable
dictation on supported devices, retaining full text fallback. Android on-device
recognition is not assumed.

## Contributor recipes

The [researcher starter kit](RESEARCHER.md) maps the architecture and provides
tested mutual-aid, resource-governance and institutional-dispute variants,
with explicit assumptions and extension boundaries.


## First-play guidance and free interpretation

The role banner states whether this is single-player (human Coordinator plus
four computer-controlled residents) or a shared night (human Coordinator and
Nell plus three computer-controlled residents). Fixture residents are labelled
scripted, not live-model results. **How to play your role** explains your own
controls; spectators see public information and no action controls.

The Coordinator can type an attempt and choose **Check interpretation · free**.
This calls the same existing `game.preview` parser available to the attached
CLI. It does not send an action, reserve resources, spend a choice or predict
consent/success. Requests, accepted commitments, transfers, performed repairs,
allocations and service promises stay distinct. A message to everyone is public,
even if typed with the `message` prefix. Literal speech stays literal text.

Only **Send action** submits an attempt. Checking first is optional. Editing the
draft or changing turn/role invalidates old interpretations; a delayed reply
cannot overwrite the interpretation of a newer draft. Connection loss cancels
the pending preview, and saved drafts reappear only after the service identifies
the session/role on reconnect. Nell keeps the structured accept/decline/speak UI,
not the Coordinator's action parser.

Browser regression checks cover the standard service/HTTP parser, free-query
state equality, malformed/literal messages, stale delivery, reconnect and actual
host-approved Nell/spectator pages with simulation execution prohibited. They do
not establish physical Android usability or replace full played-night evidence.

## Reading service status without relying on color

Each map location now says **Not resolved yet**, or names the last resolved
watch and whether service was **Maintained** or **Unserved**. The generator yard
shows its current fuel stock. The glow and dashed border are supplementary;
keyboard and screen-reader names contain the same status in words. Map inspection
still has no turn cost.

**Service across the watches** shows a native table with watch row headers and
facility column headers. **Not resolved** means there is no recorded outcome
for that watch; it does not predict success or failure. Requested allocations and
promises are not delivered service. The table and map reuse only the existing
public service records, so spectators receive the same accounting without role
journals or private conversation data.

Browser checks use one watch resolved through the existing scenario component
as a deterministic fixture (no Simulation.play/entity.act/model invocation).
They verify the initial/served/unserved distinction, keyboard-only inspection
without API mutations, 360px portrait, 800px landscape, forced colors and enlarged
text. These are software/accessibility checks, not physical-phone or human
usability validation.


## Initial teaching presets

Use `--recipe mutual-aid`, `resource-governance`, or `institutional-dispute`
with `--mode fixture` or `live` to play a trusted teaching variation through
the same server and engine. The default is `bellwether`. See
[RESEARCHER.md](RESEARCHER.md#select-the-same-initial-case-in-the-browser) for
commands, exact initial-condition changes and interpretation limits. A preset
is selected only at launch, not applied to a running night. Existing game
processes are never replaced by this option.


## Reading and composing during live updates

Repeated snapshots no longer recreate unchanged resident buttons, action
suggestions or resident decision options. Keyboard focus stays on the same
native control. The journal appends new visible events without replacing its
unchanged prefix, and unchanged context text is retained, so copying a passage
does not lose the selection merely because another update arrives. Draft text
and its caret remain yours until an intentional edit or submission.

A changed location filter, completed input session or role boundary still
updates which controls and content are available. Revocation clears the retained
journal immediately; rejoining as a spectator never restores the previous
role's private text or action controls. Retention is local presentation state,
not a durable checkpoint or a guarantee that an active operation can be cancelled.

Real Chromium checks cover narrow portrait/landscape, native keyboard focus,
selection through repeated/appended journal snapshots, draft/caret retention,
changed filter/completion and host revocation/rejoin. These tests republish
scoped snapshots and component fixtures, without running a simulation or model.
They are not physical-device or screen-reader usability studies.

## Optional model-side image drafts

[Local visual notes](VISUAL-NOTES.md) provide an explicit image-to-text CLI with
an installed vision-capable model. On your own turn, the optional image-note
review control reads its JSON locally, lets you edit it, and explicitly append
the text to your existing draft before manual Send. It does not upload the note,
attach images to shared actor context, or automatically submit an action.


### Why a resolved service was supplied

In **Service across the watches**, select **Served** or **Unserved** for a
resolved facility. This free inspection displays the boundary's recorded
request priority, available fuel and fuel spent, or the completed-repair rule.
An omitted request is distinguished from a requested facility that lacked fuel.
Requests are considered in their submitted order; later allocations or stock
changes do not rewrite the prior explanation.

These are explanations of Bellwether's material accounting rules, not of a
resident's motives or the legitimacy of an institution. They do not establish a
scientific causal conclusion. Readable public accounts include the same notes;
public JSON adds an optional allowlisted service resolution object and demand.
Older service records without these fields say that boundary details were not
recorded, rather than inferring a reason from later stocks.


### Setup context in a public account

Saved public JSON, readable HTML and the public SVG now carry the trusted
recipe's declared setup: recipe name, standard actor prefab/engine, human role
names and initial fuel assumptions. The mutual-aid recipe declares two units
used before play and six initially available; cumulative Used must not be
confused with consumption during play.

This optional public projection is **not a complete run configuration**.
Private component edits, initial memories, prompts, model settings, browser
identities and session references are excluded. A recipe label neither captures
all interventions nor guarantees deterministic replay or empirical validity.
World-only/older records without setup provenance remain explicitly unknown;
the exporter never infers setup from later stock or narrative.


### Player names and watch labels

The player UI calls the Coordinator **Coordinator Alice** and labels the first
watch **Watch 1 · Early evening**, so a time-of-night badge is not mistaken for
a resident. High Tide and Before Dawn retain their existing names.

These are display aliases only. Authenticated role IDs, join request values,
draft recovery keys and entity names remain Coordinator; the first recorded
watch remains Dusk. Raw dialogue, the human component's assembled context,
public account records and API/CLI values are not rewritten. The role guide
explains this mapping. Existing hosted chapters are not reset by this change.

### Sending and uncertain connections

The action area distinguishes **Sending** (confirmation pending), **Action
received** (delivery confirmed, not a successful outcome), rejection, and an
unconfirmed network reply. A lost reply does not mean that nothing was sent:
check the journal before sending again. No automatic retry or fabricated model
progress is shown. If the same prompt is still open and the draft is unchanged,
a manual Send reuses the existing server idempotency key.

You can edit your draft while an acknowledgment is pending. A late success
clears only the unchanged draft that initiated it; re-entering identical text
counts as a new edit. Revocation or a changed browser role clears the old
submission's presentation, and its delayed success or failure cannot modify
the new role's draft. Submission feedback is page-local; reload obtains the
current scoped server state, while existing draft recovery remains in use.

Chromium regressions cover delayed success, pre-send network loss, a lost
successful reply, validation rejection, same-key manual retry, same-text edits,
role revocation/rejoin, and narrow portrait/landscape. They use synthetic human
input and real server/SSE delivery with simulated client network delays.
Simulation.play and resident-model calls are blocked; standard Inventory
construction still uses its NoLanguageModel helper. These are not measured
phone-network reliability or physical-device usability results.

### Live basic residents and action-only structured output

The live launcher uses two views of the same local model. The ordinary model
serves standard basic self-perception, situation-perception and
person-by-situation questions as prose. Only the final standard ConcatAct policy
uses Ollama's JSON-schema output for the existing decision/speech contract.
The schema permits every existing decision (including refusal and revocation);
it does not determine consent, truth or material consequences. The game still
validates actions and enforces the same resource and commitment rules.

Resident configuration selects the standard prefix_entity_name option as false
so JSON is not prefilled with a resident name. Minimal now honors that option
with its previous true default unchanged; basic already supports it. Human
readers take precedence over the model-backed policy. The action-model binding
is runtime-only, not a serialized object or an imported module from a project.

For already installed local models, select --mode live --resident-prefab basic
--model llama3.2:3b --embedding-model all-minilm in the normal source-checkout
launch command. Use unused private ports and a distinct output directory.
Sequential remains the engine. Each HTTP request is limited to 90 seconds and
256 generated tokens; the existing call-limit wrappers reserve 192 context
calls and 64 action calls, at most 256 total. The shared standard profiler counts
both paths. No provider/model download or silent embedding fallback is added.

Basic makes additional component-generation requests compared with minimal;
semantic embedding makes additional local requests. Inspect the profiler and
invalid_resident_response events, not just HTTP status or a terminal phase.
Changing prompts or actor/memory configuration can change live behavior, even
for minimal actors. The provider may reject a schema, time out or truncate an
answer; those failures are not permission to invent a resident's consent.

Two preliminary basic/local-embedding nights reached dawn but had ten and seven
malformed resident outputs respectively; successful transport was not valid
actor behavior. Their evidence is retained. The second private browser harness
also timed out on a large developer screenshot after reaching dawn; later
harnesses write machine-readable evidence before optional screenshots.
A subsequent basic/llama3.2:3b/all-minilm schema-routed night passed strict
format and Chromium checks: 12 automated human inputs, 24 Sequential steps,
48 successful local generation calls, 196 embedding requests, zero malformed
resident responses and no browser errors. It took about 305 seconds of
automated wall time. Six of nine facility-watches were supplied; no repair was
forced. This single run does not establish reliability across models or seeds.
Automated runs do not establish physical Android usability, human play duration,
deterministic replay or empirical social validity.


### Inspecting a large developer snapshot

The attached editor receives authoritative state through the existing service.
Its Snapshot preview is closed initially and is explicitly abbreviated. Select
a state section (for example, target or join_requests) to inspect it, and use
Refresh preview when a newer received revision is indicated. Incoming updates
do not rewrite the preview being read or reset an unsent operation draft. A
stale edit is still rejected by the existing server validation.

Download received JSON exports the original last-received event text, including
large integer identifiers; it does not parse and re-serialize them in JavaScript.
The preview marks unsafe integer values and limits nested values, text and lists.
It must not be used as a complete state or checkpoint. The downloaded developer
snapshot can contain private actor information; it is not the sanitized public
account, nor is it a complete resumable engine checkpoint. This change reduces
eager DOM rendering, not server/SSE payload size or
network cost. Full download occurs only when requested.

### Unavailable resident responses

A malformed resident response is a technical failure, not the resident's
refusal, uncertainty or speech. The game records a neutral response-unavailable
notice for the intended task audience without echoing the raw output. No new
decision, consent or material effect is inferred, and an existing commitment
is not silently revoked. Valid refusals and other valid decisions retain their
normal behavior. The turn still progresses; no automatic model retry or free
replacement player action is added.

At dawn, an unavailable reply is represented by null in dawn_responses and is
labelled “Response unavailable” in the player view. It is not added to dialogue
as invented first-person words. Public accounts include only publicly delivered
failure notices; a private exchange's failure does not become public. The
private invalid_resident_response diagnostic is retained for developers. Old
saved records are not rewritten, and schema-constrained models can still fail.

### Interrupted runs in the player view

A closed human-input session is not evidence that a run completed. The player
view uses the existing run phase to distinguish completion, unexpected stop and
input closure. After failure, every approved role (including spectators) sees
a neutral stopped-run notice, not a success or indefinite waiting message.
Private provider diagnostics are not included in that notice.

Recorded scoped journals, map inspection and the existing public-account
export remain available while the service is connected. Unsent local drafts
survive reload. The account export remains labelled interrupted and is not a
saved game. Reloading does not restart the run; no automatic retry, replay or
replacement worker is added. Human action/Begin controls stop accepting input,
and unrelated host-approval messages cannot overwrite the terminal status.

### Current-state delivery to slow clients

The attached SSE listener uses the standard operation service's optional
coalesced wakeups. It retains at most one pending notification per client,
instead of materializing and queueing full snapshots that the listener would
discard. The listener still builds the current authorized snapshot at delivery,
including rechecking the browser's role. Direct snapshot subscriptions retain
their existing default behavior.

This is a current-state feed, not a promise to deliver every intermediate
revision separately. The domain event ledger and latest revision retain all
published updates. JSON envelopes, reconnect snapshots, heartbeat comments,
player privacy and read-only exports are unchanged. Full delivered snapshots
can still be large; this change does not compress their contents or establish
any model-latency, bandwidth or physical-device performance result.

### Elapsed waiting after a confirmed action

After an action has been confirmed to this page and the game is still running,
the action area shows elapsed confirmation time after fifteen seconds, updated
in five-second increments. This uses the browser's monotonic clock. It is not
model execution time, a percentage or a prediction of when a response will
arrive. No extra request or automatic resend is made. The changing text is not
a live-region announcement every tick; the existing receipt remains accessible.

The timer is absent while confirmation is uncertain or rejected and clears on
a new prompt, role change, known disconnection or terminal run state. Reload
keeps the existing draft behavior but does not reconstruct a previous page's
confirmation time. This is not provider-cost or physical-device performance
measurement, and it does not add cancellation or restart support.

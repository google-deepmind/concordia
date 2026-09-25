# One More Song

You organise a neighbourhood benefit concert. Persuade a singer and a tired
neighbour to agree on one final song. Ask, negotiate, then make your final offer:
three human turns, four AI replies, two separately sampled AI votes. Neither
character sees the other’s ballot before voting. Both must
explicitly accept. A story saying "everyone agrees" cannot win the game.

This is an engine demonstration, not a model of real people's behaviour. There
is no claim that a pleasing conversation validates the agents' psychology.

## Dependencies and launch

This example builds on the existing human-play APIs in PRs #302, #304 and #306,
including Astral Canticle's reconnectable `HumanSession` and browser boundary.
Local Ollama ballots also require the choice-contract fix in PR #378.
It does **not** require the Bellwether or consolidated editor stacks. The example
branch integrates current main and that explicit dependency. Do not cherry-pick
just this example into a release without those APIs.

Run the following from the repository root in a Python 3.12+ environment.
First install Concordia and its optional browser/local-model clients:

```sh
python -m pip install -e . fastapi uvicorn ollama
```

For a quick **scripted UI preview**, no model server or download is needed:

```sh
python -m concordia.examples.one_more_song.web --fixture
```

For **live AI dialogue**, install and start [Ollama](https://ollama.com/download)
itself; `pip install ollama` installs only its Python client, not the server.
Check `ollama list` for models already installed. If needed, download the model,
then start the demo (stop the preview with Ctrl-C first if using the same port):

```sh
ollama pull qwen3:8b
python -m concordia.examples.one_more_song.web --port 8820 --editor-port 8821
```

Open `http://127.0.0.1:8820/`. The game waits for your first line. Reloading
reconnects to the same pending turn; it does not start another game. A new
process is a new game. The default creates a separate timestamped directory under
`runs/`, printed at startup, so trying again preserves earlier conversations.
For an explicit location use `--output runs/my-first-encore`; a directory that
already contains a run is rejected rather than overwritten.

`--fixture` uses plainly labelled scripted test replies. It is for UI/mechanics
validation, **not** evidence of generative play. Real mode uses the standard
local Ollama model wrapper, defaulting to `qwen3:8b`. Use `--model llama3.2:3b`
for a smaller/faster alternative with weaker dialogue in our observed samples. There are four free-text calls and two choice calls;
choice sampling may retry. Actual end-to-end time depends on the machine/model
and human deliberation. No paid model service is required.

The server binds loopback only. Phone sharing requires the host's separately
configured authenticated proxy; this example does not alter routes, ACLs or
other games. The designer port is private and must not be exposed with the player
page. Everyone who can reach the player endpoint shares one human controller.

## First-time player and host walkthrough

1. Open the player URL printed in the terminal (not the private designer URL).
   No model setup or account is needed on the player’s browser; the host runs the
   local model. A phone needs the host’s player URL, not the Mac’s `127.0.0.1` address.
   On a phone-sized screen, **Start the conversation** takes you to
   the reply field; keyboard users can use **Skip to your reply**.
2. Ask what Maya and Leon need, or write your own proposal. Suggestions fill a
   draft; **Say it** submits it. You have three turns, not an unlimited chat.
3. Read the replies before revising your offer. **Read the latest replies** next
   to the controls takes you to the new dialogue without moving you
   automatically while you read or type. A reply phase can take over a minute.
4. Make the third turn a concrete final offer. The result shows each recorded
   vote. Read the transcript to understand it; the page does not invent a reason
   for either vote. Save the public conversation if you want to compare attempts.
5. For a fresh attempt, the host stops the server with Ctrl-C and runs the same
   launch command again. Existing tabs reconnect to the host’s one shared game;
   reloading a tab does not restart the scenario. Keep private designer logs
   separate from the public journal when sharing evidence.

To inspect the UI before downloading/running a model, add `--fixture` to the
launch command. This is a labelled scripted preview, not an AI playthrough.

## If something does not work

- **The browser cannot connect:** keep the host terminal running and use the
  exact player port it prints. A loopback URL works on that host only.
- **The conversation is waiting:** the page shows elapsed time. Local inference
  can be slow; reloading reconnects rather than restarting or speeding up a turn.
- **The run stopped:** the host should inspect the terminal log, check that the
  Ollama server is running, and verify the selected model with `ollama list`.
  Save the partial public conversation before starting a fresh run.
- **A send was not confirmed:** use **Check last send**. It checks the original
  turn safely; do not paste the old reply into the next turn as a retry.
- **Someone else has taken a turn:** all browsers on this URL share one organiser.
  This is a single-controller demo, not separate per-visitor games.

## What this demonstrates

| Capability | Actual standard API |
| --- | --- |
| Replace an actor's action policy, not its memory/context | `minimal.Entity.build(act_component_factory=...)`, `HumanActComponent` |
| Independently motivated, observation-driven AI characters | `minimal.Entity`, `Constant` goals, `LastNObservations`, `ConcatActComponent` |
| Human/AI turn-taking with finite scenario rules | `generic.Simulation`, `Sequential`, `NextActingInFixedOrder`, `ActionSpec` |
| Public speech, private actor internals | `MakeObservation` queues, `EventResolution`, filtered `HumanSession.snapshot()` |
| Inspect and intervene, then continue | `SimulationServer`, `StepController`, `visualize_config_to_html`, `SimulationLog` |

`BallotPhase` is an example-specific rule component selecting speech or a ballot,
not a replacement engine. `PlayerSession` adds only the public vote tally and
presentation data to the existing inbox. No LLM-generated prose is parsed into
factual resource state. Public actions are utterances/proposals; the outcome
means agreement, not that a physical song actually happened.

## Designer intervention and continuation

Open the separate `http://127.0.0.1:8821/` standard designer. Select an entity to
inspect its real components. Private goals and model prompts are visible here,
not in the player view.

For a safe, reproducible boundary:

1. While the first human prompt is waiting, press **Pause** in the designer.
2. Submit the first human line in the player page. Wait for step 1 in the
   designer: that human action finishes, then the engine is paused before Maya.
3. Select **Leon**, expand **Goal**, change its dynamic `state` field and Save.
   For example make a quiet, firmly timed encore more acceptable. Record the old
   and new values as an explicit designer intervention, not an emergent change.
4. Press **Play**. Maya and Leon continue using their existing memories and the
   edited goal. Inspect the next action's context in the standard designer log.

The standard controller's paused flag alone is not proof that a currently
in-flight human/model action has finished; wait for the step boundary. The
example does not rewrite editor semantics. Goal edits do not erase memories,
force an acceptance vote or prove a causal effect; compare multiple runs if
investigating that question.

## Output and limitations

- `simulation.json` and self-contained `log.html`: standard designer logs,
  including private model context. Keep private.
- `checkpoints/`: standard snapshots after completed steps, including phase and
  queued public observations. They are for inspection; this CLI does not claim
  complete restart/fork restoration of a live human session.
- `public.json`: only the player-visible conversation, explicit votes and status.
- `timing.json`: observed wall time including human waits; per-step durations in
  public state also include waits and persistence overhead, not pure model time.

The page keeps turn/wait instructions beside the reply controls as the
conversation grows, and keeps unsent drafts through disconnection and reload in the same tab.
Drafts use browser session storage, isolated by the host run; a fresh game does
not reuse an earlier offer. When browser policy blocks storage, the open tab
still keeps its draft, but reloading cannot recover it.
Stalled network requests time out and reconnect; a timed-out submission keeps
the draft and offers **Check last send**, because the server may already have
accepted it. That button rechecks the original request ID using the existing
inbox’s idempotency; it cannot consume the next speaking turn. The pending
confirmation survives a same-tab reload when session storage is available.
Submissions are never automatically replayed.
Tab-scoped storage is not a permanent backup; losing the browser session loses unsent drafts. Host/provider errors
end the session with a visible message; partial standard logs are saved.
Changing a local model or re-running does not promise the same conversation.
Mobile viewport emulation is not physical-phone verification.

## Repeatable player-journey checks

The example-specific `playtest` driver uses maintained Playwright primitives;
there is no separate journey engine. It attaches to an **already running fresh
game**, submits all three human turns, captures screenshots and public state,
checks offline draft recovery and the downloaded transcript, and records elapsed
time from each submission until the next human turn or ending. These timings
include browser polling and persistence, not just inference. Install the optional
test dependency with `pip install playwright` and `playwright install chromium`.

For a fixture started with both ports above:

```sh
python -m concordia.examples.one_more_song.playtest --port 8820 --editor-port 8821 --output runs/desktop-evidence
```

`--editor-port` also performs the documented paused Leon goal intervention using
the actual designer controls at a separate 1280px desktop viewport, even when
the player viewport is mobile-sized. This does not claim a phone-sized designer
journey. Its evidence contains the old private goal: keep
that output private. Omit it for an unintervened baseline. For a fresh mobile-size
fixture, use `--width 390` (emulation, not a physical phone). For a fresh local-model
game, add `--real`; `--scenario compromise`, `demand`, `revision`, or `ambiguous`
selects the human utterances. Real runs record the votes **without asserting a
preferred outcome**. A single playthrough cannot establish model reliability or
the causal effect of an intervention.

`--network-faults` additionally holds a real polling request until the page
reports disconnection, then restores it. It also forwards the first submission
but withholds its acknowledgment, verifying that the draft survives, only one
human action is recorded, and the journey can continue. These fault checks add
around 30 seconds to the journey; do not compare their timings with normal
model-latency runs. They use Playwright routing, not a replacement transport.

For a different imagined player journey, pass `--actions-file my-offers.json`
containing exactly three nonempty strings (up to 8000 characters each). This
replaces the named scenario without editing driver code. The evidence records
the actual utterances; it does not judge their quality or assert winning votes.
Use this for clearly labelled simulated-user hypotheses, not as a substitute
for observing human players.

The driver requires a new game for each complete journey; it does not reset,
start, stop or silently substitute models. The fixture is deliberately scripted
and always accepts. Its votes cannot validate negotiation quality.

Every journey captures a Playwright `trace.zip` (open with `playwright show-trace`).
On failure, `failure.json` preserves the error and available public state, with
`failure.png` when capture is possible; the command still exits unsuccessfully.
Treat traces as private: designer journeys can include component state, and all
journeys include the entered conversation. Failure capture never starts a server
or silently retries a simulation.

## Live playtest observations (25 September 2026)

Four automated Chromium journeys against local `llama3.2:3b`, with the neutral
shared scene and first-person speech configuration, all completed nine steps.
Each used a fresh server; no outcome was forced by the driver:

| Human approach | Maya | Leon | Submission-to-ending time |
| --- | --- | --- | --- |
| Quiet two-minute compromise | ACCEPT | DECLINE | 12.220 s |
| One-hour amplified demand | DECLINE | DECLINE | 8.581 s |
| Withdraw demand, offer quiet farewell | ACCEPT | DECLINE | 9.176 s |
| Ambiguous/off-topic offer | ACCEPT | DECLINE | 10.429 s |

These timings exclude human reading, typing and deliberation; they **do not**
prove a 5–10-minute human playthrough. The machine/model may behave differently.
The adapter fix eliminates the reproduced invalid-choice failure in these
samples, not every possible provider failure.

Small-model dialogue still sometimes invents details, refers to itself in the
third person, or adds conditions the player did not propose. A reasonable offer
is not guaranteed acceptance, and variation between runs is not evidence of
psychological realism. The shared scene deliberately does not tell every actor
that they are the organiser. The speaker prefab reuses standard
`ConcatActComponent(prefix_entity_name=False)` so Sequential's existing actor
label is not duplicated. Neither change rewrites generated speech or forces a
vote. Inspect the actual transcript, not just the final score.

The default `qwen3:8b` was selected after comparison with already-installed local
models, not from a scripted successful outcome. With the same prompts, its
revised-offer journey took 131.253 s and recorded ACCEPT/ACCEPT; the one-sided
demand took 95.400 s and recorded DECLINE/DECLINE. The ambiguous journey took
235.609 s and recorded ACCEPT/ACCEPT after the characters themselves proposed a
specific quiet encore. That does not prove the final vague wording was prudent:
interpret the transcript, not a predetermined pass/fail label. A larger local
model can take over a minute per reply phase on CPU; the page shows elapsed wait
without claiming a completion percentage. Choose the smaller model explicitly
when responsiveness matters more than the observed quality tradeoff.

If a run fails, `public.json` now marks it stopped, with no fabricated ending or
votes; raw provider diagnostics remain in private host logs. The page likewise
distinguishes an interrupted conversation from a completed ballot.

A further 390px Qwen revised-offer journey deliberately stalled polling and
withheld the first accepted submission’s response. The page detected the stalled
poll after 12.917s, kept the unconfirmed draft, did not replay it automatically,
and completed all nine steps with no page errors (ACCEPT/ACCEPT). Submission-to-
ending time was 130.691s, excluding the initial polling fault and human thought.
This tests recovery with one observed live model run, not network reliability.

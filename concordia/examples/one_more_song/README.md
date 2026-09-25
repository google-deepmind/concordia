# One More Song

You organise a neighbourhood benefit concert. Persuade a singer and a tired
neighbour to agree on one final song. Ask, negotiate, then make your final offer:
three human turns, four AI replies, two independently sampled AI votes. Both must
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

From a Python 3.12+ checkout, install Concordia plus the existing optional browser
and local-model dependencies:

```sh
python -m pip install -e . fastapi uvicorn ollama
ollama pull qwen3:8b
python -m concordia.examples.one_more_song.web --port 8820 --editor-port 8821 --output runs/one-more-song
```

Open `http://127.0.0.1:8820/`. The game waits for your first line. Reloading
reconnects to the same pending turn; it does not start another game. A new
process is a new game. Use a new output directory for each run.

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

The page reports waiting/disconnection and keeps unsent drafts in the open tab.
It does not store drafts across browser/process restarts. Host/provider errors
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

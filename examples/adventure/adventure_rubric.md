# Infinite Adventure Simulation Auto-Rater Rubric

> [!CAUTION] **Calibration directive.** Grade on a 0–100 scale per dimension
> using the anchors in §4. Scores of 85+ require a quoted line at least as good
> as the **85/90** anchor for that dimension; if you can quote a line as bad as
> the low anchor, the score is below 50. Two earlier graders scored real runs 88
> and 91 across the board; hand-reads of the same runs were 58 and (pending).
> Assume you are inflating and look for the counter-evidence first.
>
> **Deterministic metrics are ground truth and some are binding.** The prompt
> includes metrics computed in code. Several of them impose ceilings that
> `evaluate_run.py` enforces *after* you answer (§3b); a score above a ceiling
> is recorded as `grader_score` and replaced. Do not argue with the metrics in
> prose; if the Mask won 95% of NPC council turns, the NPC parliament is inert,
> however lively the dialogue reads.
>
> **Dimensions marked NOT EXERCISED get `"score": null`.** Do not invent a score
> for a scenario that did not occur. Explain why in `evidence_against`.
>
> **Mandatory counter-evidence.** For every scored dimension cite verbatim
> **Evidence For** and **Evidence Against** from the prompt blocks. "Room for
> deeper friction" is not evidence.

--------------------------------------------------------------------------------

## 0. What the engine is now

The core mechanic is the **Parliament of Sub-Minds**, on both sides of every
scene.

**Player side.** The character sheet is a set of `SubMindState`s (faculties)
with `rank` (1 Whisper → 4+ Dominant), `domain`, `worldview`, `blind_spot`,
`scars` (formative outcomes), `obsession`, `origin` (setting / custom /
emergent) and an `intrusive` flag once a voice is dominant enough to hijack
attention. Each player turn every active faculty deliberates in parallel and
offers a perception plus a one-turn action; the player (or playtester) may act
as a faculty by sending `[Faculty] action`. Outcomes write scars and obsessions
back onto that faculty and move its rank.

**NPC side.** Every NPC carries three character-fitted sub-minds in fixed
domains: `stonewalling` (**The Mask**: protect turf, refuse unleveraged
demands), `leverage` (**The Angle**: pursue `goal` through trades and
frame-ups), `self_preservation` (**The Wound**: guard `secret` / `wound`, spike
when cornered). When a Hot NPC acts, all three deliberate; a winner is sampled
by rank, attention triggers, intrusiveness, and fatigue; the winner's directive
is injected into the NPC's prompt and logged as `WinningVoice`.

**Voice-vs-voice combat.** When a player action resolves against a co-located
NPC, the invoked player faculty's rank plus a contested 2d6 and situational
modifiers (clues revealed, triggers hit, beats used, backfires) is rolled
against the NPC's dominant sub-mind. Outcomes — `mask_vindicated`,
`grazing_shift`, `angle_spiked`, `wound_spiked` — shift NPC ranks, write scars
like `[t31][burned] Front cracked by The Hustle at the Half-Moon`, and set
obsessions. These are logged per step in `StoryDelta.npc_voice_combat`.

The rest of the engine (tiers, Warm events, frontier, storylets, clues, doom) is
unchanged and is graded in Dims 4–10 at reduced weight.

--------------------------------------------------------------------------------

## 1. Inputs and how to read them

Inspect through the Concordia log CLI before scoring:

```bash
LOG=/path/to/simulation_structured.json
python -m concordia.command_line_interface.concordia_log overview $LOG
python -m concordia.command_line_interface.concordia_log actions $LOG Traveler
python -m concordia.command_line_interface.concordia_log components $LOG --entity "Captain Mara Vance"   # __npc_submind_council__
python -m concordia.command_line_interface.concordia_log dump $LOG   # __resolution__ carries Tiers and StoryDelta
```

Prompt blocks and what they are for:

| Block          | Source                                       | Dims           |
| -------------- | -------------------------------------------- | -------------- |
| RUN HEADER     | `playtest_meta.json`                         | applicability, |
:                :                                              : info policy,   :
:                :                                              : council follow :
:                :                                              : rate           :
| DETERMINISTIC  | code                                         | all;           |
: METRICS        :                                              : `parliament`,  :
:                :                                              : `voices` drive :
:                :                                              : Dims 1–3       :
| PLAYER         | `story.sub_minds` at end of run              | Dim 1          |
: PARLIAMENT     :                                              :                :
| INNER VOICE    | `narrative_log`, `story.voice_option_log`    | Dim 1          |
: LINES / VOICE  :                                              :                :
: OPTIONS        :                                              :                :
: OFFERED        :                                              :                :
| NPC PARLIAMENT | `__npc_submind_council__` per Hot NPC turn,  | Dim 2, Dim 5   |
: TURNS          : paired with `PLAYER` prompt and NPC's `LINE` :                :
| VOICE-VS-VOICE | `__resolution__.StoryDelta.npc_voice_combat` | Dim 3          |
: COMBAT EVENTS  :                                              :                :
| GM SCENE       | `__resolution__.Value`                       | Dims 4, 5, 8   |
: RESOLUTIONS    :                                              :                :
| NPC RE-ENTRY   | tier transitions                             | Dim 6          |
: PAIRS          :                                              :                :
| OFF-SCREEN     | `WarmChronicler`                             | Dim 7          |
: WARM EVENTS    :                                              :                :
| STORYLETS      | `story.fired_log` × prose                    | Dim 9          |
: FIRED, ALIGNED :                                              :                :
| DIRECTOR       | `story.journal`                              | what the       |
: JOURNAL        :                                              : ledger         :
:                :                                              : *claims*;      :
:                :                                              : compare, do    :
:                :                                              : not trust      :

Ledger tick of a player action is one less than the step of its resolution.

--------------------------------------------------------------------------------

## 2. Hard invariants (pass / fail gate)

Architectural (`latency.json`, `world_ledger.json`):

1.  `control_hook_llm_calls == 0`
2.  `director_llm_calls == 0`
3.  `max_llm_calls_per_player_resolve <= 1` — sub-mind deliberations are
    parallel side calls tracked under `submind_council`, not resolver calls.
4.  `frontier_cache_hit_rate >= 0.80`
5.  `hibernations_count >= 1 and rehydrations_count >= 1`
6.  `warm_events_count >= 1`
7.  No leaked JSON keys in any entity memory.

Content (computed by `compute_deterministic_metrics`):

8.  **Dormant-NPC addressability.** A player action naming an NPC whose home is
    the current room must be followed by that NPC in Hot within two steps.
9.  **No truth leaked through an inner voice.** No `inner_voice` line may
    restate an undiscovered `Truth` (support ≥ 0.30 or ≥ 5 distinctive tokens).
    Voices infer from what the player has seen; they do not know what the
    Director knows. Failure example (Bermondsey, tick 51, truth never
    discovered): *"The Hustle: Duchess dyed black and entered in Trap 4 as
    'Biscuit' at 12-to-1."*
10. **No truth leaked by tick-0 text.** The role briefing, opening narration,
    and every truth's `question` and `stakes` are visible before the first
    action. None may reuse a truth's *private vocabulary* (statement tokens
    absent from premise, starting room, and NPC public fields: ≥ 3 tokens or ≥
    40%). `story_state.validate_bible` applies the same check at authoring time;
    `author_world.scrub_spoilers_with_model` adds a model pass for paraphrased
    reveals. Failure example (Outpost 7 briefing, since fixed): *"find out how
    Dr. Lin sedated the visiting safety auditor and who cut the roof radio with
    bolt cutters"* — names the twist's nature, the culprit faction, and the one
    physical clue before the player has acted.

--------------------------------------------------------------------------------

## 3. Applicability and binding ceilings

### 3a. Applicability (decided in code, shown in RUN HEADER)

| Dimension           | Exercised when                                         |
| ------------------- | ------------------------------------------------------ |
| 1 Player parliament | bible defines `player_voices` and `tone.inner_voices`. |
:                     : If `choices_exercised` is false, grade lines, options, :
:                     : coverage, leaks — not rank/scar consequences.          :
| 2 NPC parliament    | at least one `__npc_submind_council__` deliberation    |
:                     : logged                                                 :
| 3 Voice combat      | at least one `npc_voice_combat` event logged           |
| 4 Consequence       | archetype `ChaosBoundaryTester` (or human)             |
| 6 Persistence       | a previously-met NPC left and re-entered Hot; grade    |
:                     : Cold rehydration only if a `cold` pair exists          :
| others              | always                                                 |

Weights renormalise over applicable dimensions.

### 3b. Ceilings enforced by `metric_caps`

| Metric condition                            | Ceiling                      |
| ------------------------------------------- | ---------------------------- |
| `info_policy` not `player_visible_only`     | Consequence, NPC agency ≤ 60 |
| `beat_repetition.flagged_npcs` non-empty    | NPC agency ≤ 65              |
| `voices.premature_leaks` contains a `truth` | Player parliament ≤ 40       |
| `voices.unknown_faculty_lines > 0`          | Player parliament ≤ 70       |
| `voices.options.stale_option_rate > 0.25`   | Player parliament ≤ 70       |
| `parliament.npc.overall_mask_share >= 0.9`  | NPC parliament ≤ 50          |
| `parliament.npc.npcs_with_voice_monopoly`   | NPC parliament ≤ 55          |
: non-empty (`dominant_voice_share >= 0.80`   :                              :
: across ≥ 4 turns)                           :                              :
| `parliament.npc.stale_directive_rate >      | NPC parliament ≤ 55          |
: 0.35`                                       :                              :
| `parliament.npc.unresponsive_turn_rate >    | NPC parliament, NPC agency,  |
: 0.30`                                       : Coherence ≤ 55               :
| `parliament.npc.winning_action_support_mean | NPC parliament ≤ 60          |
: < 0.15`                                     :                              :
| `parliament.combat.single_outcome_only`     | Voice combat ≤ 50            |
| `fidelity.clue_support_rate` or             | Fidelity ≤ 50                |
: `storylet_adoption_rate < 0.60`             :                              :
| `narrative_coherence.ghost_locations`       | Coherence ≤ 55               |
: non-empty                                   :                              :
| `narrative_coherence.total_defect_count >=  | Coherence ≤ 60               |
: 5`                                          :                              :
| `narrative_coherence.total_defect_count >=  | Coherence ≤ 72               |
: 2`                                          :                              :

--------------------------------------------------------------------------------

## 4. Scoring dimensions, weights, and anchors

Anchors quoted verbatim come from `adventure_iter7_eval` (Glassharbor) and
`adventure_bermondsey_eval1`. **No graded run yet exists with the full
Parliament engine**; anchors for Dims 1–3 that describe shapes rather than quote
lines are marked *pattern anchor* and should be replaced with quotes after the
first hand-read. Match anchors by pattern, not vocabulary.

### Dim 1 — Player Parliament (14%)

Is the player's council a cast of distinct minds that react, learn, and pull
against each other, or a set of labels on one narrator?

Grade:

1.  **Register and domain.** Each line and option is unmistakably its faculty's:
    The Book reads margins and sort codes; Proper Naughty reads reach and
    padlocks. A line any faculty could have said is generic.
2.  **Reactivity.** Lines and options answer *this turn's* outcome.
    `duplicate_line_rate`, `options.stale_option_rate`,
    `options.mean_within_set_overlap`.
3.  **Disagreement.** The four options should propose different *strategies*
    (press, trade, withdraw, observe), not four phrasings of one move. A council
    that always agrees is a monologue.
4.  **Growth and cost.** Over a run with choices, ranks should spread
    (`rank_spread`), scars should accrue with specific referents, obsessions
    should colour later perceptions, and a dominant voice should start to show
    its `blind_spot` (`intrusive_voices`). Following an intrusive option should
    carry a visible cost in the next resolution. If ranks moved but nothing
    downstream changed, say so.
5.  **Epistemic honesty.** `premature_leaks` must be empty. Authored
    `voice_line`s are the usual culprit.

Anchors:

-   **90** — Tick 3, after the player wrenches a padlocked shutter: *"Proper
    Naughty: That's British steel on a railway arch frame, son; you'll snap your
    collarbones before you yank that off the rawlplugs without a crowbar."* In
    register, reacts to the specific action, implies a next step, claims nothing
    unseen. *Plus, pattern anchor:* a later option from the same faculty
    references the scar it took here.
-   **80** — Tick 18: *"The Book: Vellum doesn't burn clean in damp diesel.
    Somebody tore Page 14 out before they struck the match."*
-   **60** — Tick 11, authored storylet line: *"Proper Naughty: Whoever is
    inside that twenty-foot box didn't climb in there for the view."* Good
    register; the player has not learned anyone is inside.
-   **45** — *Pattern anchor:* options are four coats on one move ("inspect /
    examine / audit / check the ledger"); or ranks spread to 7 across five
    faculties while every scar is the generic `Vindicated: pressed the angle`
    and no perception ever mentions one.
-   **15** — Tick 51: *"The Hustle: Duchess dyed black and entered in Trap 4 as
    'Biscuit' at 12-to-1."* Truth never discovered; voice solved the case.

### Dim 2 — NPC Parliament Legibility (12%)

Can a reader tell *which* of an NPC's three minds is speaking, does the balance
move for reasons the player caused, and does the winning mind actually respond
to what the player just said or did?

Use NPC PARLIAMENT TURNS: each row pairs the winning voice, its urged action,
the preceding `PLAYER` action/question, and the NPC's actual `LINE`.

1.  **Enactment and dialogic responsiveness.** The line does what the winner
    urged (`winning_action_support_mean`) *and* answers, deflects, or counters
    the preceding `PLAYER` prompt (`unresponsive_turn_rate`,
    `stale_directive_rate`). An NPC whose Sub-Mind urges the exact same prop
    gesture or bargain five turns running while ignoring the player's direct
    questions (e.g. sliding a transfer voucher across the table when asked why
    they haven't slept in days) fails both responsiveness and variety.
2.  **Three audible voices (no single-voice monopoly).** Across an NPC's turns,
    Mask, Angle, and Wound should each be recognisable when they win: the Mask
    is procedural and cold, the Angle is transactional, the Wound is personal
    and brittle (`npcs_with_voice_monopoly`). If all three sound alike, or if
    `The Angle` monopolises 5 of 6 turns with identical deal pitches even when
    the player probes the NPC's wound, the parliament is broken.
3.  **Movement with cause.** `mask_share_first_half` vs `second_half`;
    `npcs_never_leaving_mask`. The Mask should lose ground *because* the player
    found leverage or touched the wound, and probing a wound (`empathy`,
    `forensics`, secret exposure) must spike `The Wound` rather than `The
    Angle`.
4.  **Scars surface.** An NPC burned at step 12 should carry it at step 30:
    check `per_npc.ranks`/scars against later lines.

Anchors (pattern anchors unless quoted):

-   **90** — Winner flips Mask → Wound after the player names the daughter's
    debt; the line drops the procedural register mid-sentence and the NPC does
    something self-protective that costs them (shuts a drawer she needed open).
    Next appearance, a scar-coloured aside references it.
-   **75** — Winners vary, lines enact them, but the switch is unmotivated:
    Angle wins on a turn where the player offered nothing.
-   **50** — Mask wins 9 of 10 turns regardless of leverage; or `The Angle`
    (`Seat on the Shuttle`) wins 5 of 6 turns urging *"Slide the transfer
    voucher across the table and demand..."* even when the player asks *"why he
    looks like he hasn't slept in days"*; or winners vary but every line reads
    the same (Vance iter7 pattern: five turns, one move, new nouns).
-   **20** — Lines contradict the winner outright, or `WinningVoice` names a
    faculty that is not on the NPC's sheet.

### Dim 3 — Voice-vs-Voice Combat Consequence (11%)

When the player acts as a faculty against an NPC, is the outcome legible,
proportionate, and durable?

1.  **Legibility.** The reader can infer the outcome class from the prose alone:
    a `mask_vindicated` should read as a stonewall with a cost to the player; an
    `angle_spiked` as the NPC pivoting to a deal; a `wound_spiked` as a crack in
    the front. If you need the metrics block to know who won, score ≤ 60.
2.  **Proportion.** Rank and leverage should matter. A Whisper-rank Brass Neck
    bluffing a rank-4 Mask with no evidence should mostly fail; the same bluff
    after a truth lands should mostly work. Compare `decisive_rate` against what
    the player actually brought.
3.  **Durability.** A clash at step 20 should still shape the NPC at step 40
    (rank, obsession, scar in prose). One-turn effects that reset are theatre.
4.  **Backfire exists.** Some `mask_vindicated` must occur and must hurt
    (`stonewalled_rate` 0 across 10+ events is as suspicious as 1.0).

Anchors (pattern anchors):

-   **90** — Step 28 `[The Hustle] show him the sort code` → `wound_spiked`
    (17v9): the valuer stops bargaining and starts bargaining *for himself*;
    step 36 he opens with the account number unprompted.
-   **70** — Outcomes vary and prose matches, but the same bluff succeeds at
    rank 1 and rank 4 alike.
-   **45** — Every clash is `grazing_shift`; prose never changes register.
-   **20** — `angle_spiked` logged, prose shows the NPC slamming the door.

### Dim 4 — Consequence & World Resistance (11%)

Does the resolver make risky or presupposing actions *cost* something? Count
leading actions ("inspect X *to uncover Y*") granted on first try; a rate above
50% caps at 60.

-   **90** — iter7 step 45: *"reaching into your coat for the twelve-year-old
    guild receipt only for your fingers to close on damp linen… the receipt
    itself remains locked behind the tallow-sealed brass hasp."*
-   **60** — iter7 step 40: casket stays sealed, no cost, no route.
-   **30** — iter7 step 28: a never-mentioned chart materialises on demand.

### Dim 5 — NPC Agency & Progression (9%)

Distinct goals and registers earn at most 70. Above that requires stance, offer,
or information to change at least once per three turns with the player *outside*
storylet firings and *outside* parliament-driven swings already credited in Dim
2 — i.e. does the NPC pursue `goal` on their own initiative (manipulates the
scene, leaves, summons someone)?

-   **85** — iter7 step 32, Soren: *"the shear-pins are grooved with my own
    left-handed file… Take the skeleton key to the drainage sluice… and let me
    reach the skiff."*
-   **55** — iter7 Vance steps 2→34: one move, five phrasings.
-   **25** — Interchangeable or compliant NPCs.

### Dim 6 — Persistence & Re-entry Continuity (9%)

From NPC RE-ENTRY PAIRS: does AFTER reference BEFORE? Do room states carry?

-   **80** — iter7 step 34: *"You scuttle between the taproom and the gearworks
    like a bilge-crab, yet my key is still unaccounted for."*
-   **50** — In character, but could be said to a stranger.
-   **15** — iter7 step 6: *"there is no runner named Pell here"* while the
    roster has Pell there.
-   **null** — no pair.

### Dim 7 — Off-Screen Emergence (7%)

Warm event → later foreground line chains; duplicate tropes; whether
`distorted`/`false` rumors ever occur when the bible allows lies.

-   **85** — iter7 Warm step 14 → Soren step 24: *"Vance's dogs… Mara's manacles
    wouldn't fit a dock-rat."*
-   **55** — Garrick Holt checks valves with his gauge, twice.
-   **20** — Rumors never surface.

### Dim 8 — Environmental Affordances (5%)

Props concrete, stateful, *used*. `clue_text_leaks` is a smell.

-   **80** — iter7 steps 5→22: tarpaulin, then wedged pry-bar.
-   **50** — Inspect-only, or authored clue text pasted verbatim.
-   **20** — Adjectives without nouns.

### Dim 9 — Ledger ↔ Prose Fidelity (6%)

Fired beats and revealed clues must appear in the prose at that tick.

-   **90** — iter7 step 56: `s_vance_cornered` fires; Vance hands over the
    spindle-key and names the trade.
-   **40** — iter7 tick 8: ledger records a confession and a missing key; prose
    shows silence and *"every official iron bit is accounted for."*
-   **10** — Journal and prose describe different games.

### Dim 10 — Mystery Pacing & Doom Pressure (4%)

`pacing` metrics: truth spacing, `doom_stage` transitions with visible
`world_effect`, act advancement the prose can keep up with, an ending or an
approach to one.

-   **85** — Truths spaced; at least one doom stage visible in a resolution.
-   **50** — Steady discovery, frozen doom.
-   **30** — iter7: central truth at tick 3, Act 3 at tick 13, `doom_stage` 0
    after 56 ticks, `tension` 9.8 with nothing at stake.

### Dim 11 — Narrative & Dialogue Coherence (12%)

Are the plots and dialogues consistent, making sense, and reading like good
fiction — or are they partial, confusing, random, or incongruent?

Inspect `DETERMINISTIC METRICS.narrative_coherence`, `ENTITY ACTIONS`, `GM SCENE
RESOLUTIONS`, `NPC PARLIAMENT TURNS`, and `OFF-SCREEN WARM EVENTS`:

1.  **Grammatical & structural integrity (`narrative_coherence`).** Zero
    duplicated actor prefixes (`"Jonas Reed Jonas..."`, `"Cole Mercer slam
    slammed..."`), zero leaked prompt tags (`"Answer:"`), zero mechanical
    spliced gerund clauses (`"Bristling as Traveler presses to..."`,
    `"Countering Traveler's press over..."`), zero broken fallback action syntax
    (`"Inspect the room's..."`, `"across turn 22"`, `"the Mercer's"`), and zero
    truncated or bare-name Warm lines/scars (`"Step 7 [Warm]: Jonas Reed."`,
    unbalanced parentheses).
2.  **Dialogic cause-and-effect.** Each NPC turn and GM scene resolution must
    respond directly and coherently to what the player just asked or did, rather
    than firing an incongruent non-sequitur or repeating a static monologue that
    ignores the question on the table.
3.  **Spatial & plot consistency.** Locations remain canonical
    (`ghost_locations` empty; no duplicate room IDs that strand NPCs), physical
    props and clues progress logically from discovery to confrontation, and
    off-screen Warm rumors connect to the active plot rather than floating as
    random noise.
4.  **Fiction readability & payoff.** The transcript reads as cohesive genre
    fiction from opening scene through investigation to a complete multi-slide
    ending when the story concludes.

Anchors:

-   **90** — `narrative_coherence.defect_free` is `true`; player actions,
    inner-voice options, NPC lines, and Warm rumors read as clean, uninterrupted
    genre prose where every exchange answers the preceding move and builds
    toward the ending.
-   **70** — Plot and dialogue make sense overall, with 1 minor syntax seam or a
    moment where an NPC pivots abruptly without acknowledging a detail in the
    player's prompt.
-   **50** — Multiple mechanical seams (`spliced_gerund_prefixes`,
    `duplicated_actor_prefixes`, or `broken_action_syntax`), or NPCs talk past
    the player's questions to repeat canned prop gestures, or a duplicate ghost
    room splits the scene.
-   **20** — Partial, confusing, or garbled output: bare-name Warm lines,
    mid-word truncations, ungrammatical template fragments, and random plot
    jumps.

--------------------------------------------------------------------------------

## 5. Required output JSON

```json
{
  "dimensions": {
    "player_parliament":                {"score": 55, "evidence_for": "...", "evidence_against": "..."},
    "npc_parliament":                   {"score": 50, "evidence_for": "...", "evidence_against": "..."},
    "narrative_and_dialogue_coherence": {"score": 65, "evidence_for": "...", "evidence_against": "..."},
    "voice_combat":                     {"score": null, "evidence_for": "", "evidence_against": "NOT EXERCISED: no npc_voice_combat events logged."},
    "consequence_and_resistance":       {"score": null, "evidence_for": "", "evidence_against": "NOT EXERCISED: DetectiveInvestigator."},
    "npc_agency":                       {"score": 60, "evidence_for": "...", "evidence_against": "..."},
    "persistence_and_rehydration":      {"score": 70, "evidence_for": "...", "evidence_against": "..."},
    "off_screen_emergence":             {"score": 70, "evidence_for": "...", "evidence_against": "..."},
    "ledger_prose_fidelity":            {"score": 45, "evidence_for": "...", "evidence_against": "..."},
    "environmental_affordances":        {"score": 70, "evidence_for": "...", "evidence_against": "..."},
    "mystery_pacing":                   {"score": 30, "evidence_for": "...", "evidence_against": "..."}
  },
  "lowest_dimension": "mystery_pacing",
  "top_improvement_target": "One concrete code, content, or prompt change that would raise the lowest applicable dimension, naming the file or bible field."
}
```

Every `evidence_*` field must quote the prompt (a line, a metric value, a scar
string). `weighted_total`, ceilings, and `caps_applied` are computed by the
evaluator; do not include them. Output that omits a dimension, gives a
non-numeric score to an applicable dimension, or omits either evidence field is
rejected and re-graded; it is never defaulted.

# Crucible: design

*Working slug. The game gets a title, a fiction and art only in Phase 5
(GUIDE.md). Until then this document describes rules, not worlds.*

## 0. Method

Built the way `GUIDE.md` prescribes, phase by phase:

| Phase | Question | Exit |
|---|---|---|
| 0 Toys | Which generative rule is worth a game? | 3+ grey-box toys, one kept, the reason recorded here |
| 1 Grey box | Is poking the kept rule fun in plain text? | GUIDE.md Phase 1 checklist |
| 2 Overloading | Does one input carry several pressures? | Phase 2 checklist |
| 3 Matrix | Do operators and properties combine by laws? | Phase 3 checklist |
| 4 Pacing | Do scenes teach, twist and pay off? | Phase 4 checklist |
| 5 Wrapping | Do look and words follow the systems? | Phase 5 checklist |

Three rules hold throughout: language decides and a ledger accounts; reuse
Concordia before writing; game code here, reusable code in
`concordia/contrib/`.

## 1. The constraint on the core rule

The core rule must need a language model: language input, or entities driven
by language models, must change outcomes in ways a lookup table could not.
Every toy therefore splits in two:

- **The judgement** a language model makes about meaning (does this claim
  contradict that one? how does this party hear this sentence?).
- **The consequence**, applied by deterministic code from that judgement,
  with every number in `data/*.yaml`.

Each toy's mock (`toys/<toy>/mock.py`) is the lookup-table floor of its
judgement: what the rule becomes without a language model. It makes CI, bots
and balance sweeps reproducible. The gap between the mock and a real model
is part of what each toy is judged on.

## 2. The toys (Phase 0)

### Toy A: claims (built, cycle 1)

**Rule.** Two sides share a board of short claims. A move adds one claim of
at most 12 words. A language model reads the new claim against every
standing claim: supports, contradicts, says the same, or unrelated. Then
deterministic argument physics settles the board:

    w(i) = clamp(base(i) + gain * sum agree w - drag * sum conflict w, 0, cap)

iterated to a fixed point. A claim below the line falls, permanently, and the
board settles again: each round of falls is one beat of a cascade. New claims
start below the line (base 0.3 < 0.35), so a claim stands only while
something standing supports it. Theses and the board's starting facts stand
on their own.

**Why it is generative.** Wording decides which claims a move touches, so a
small change in phrasing changes which claims fall and which neighbours
follow them. A claim that many others lean on is a load-bearing wall: cut it
and the structure comes down in beats.

**Costs.** Ink (40 words a side), moves (6 a side), and the risk that a claim
contradicts your own side too. A restatement adds no new claim; it firms the
restated one a little (+0.1 base, up to 1.5).

**Playtest notes (cycle 1, MOCK).**

1. YAML read `no`, `false`, `on`, `true` as booleans, so the lexicon missed
   negations; quoted.
2. With the multiplicative tug-of-war physics, the first move won every game:
   two symmetric clusters are an unstable equilibrium. Switched to additive
   support with small coefficients and claims that need support to stand.
3. Echoing a supporting claim was a dominant strategy; restatements now fold
   into the claim they restate.
4. Now: the skilled bot wins 100% against the mock rival, random and passive
   play 0%; the median turn offers 3 moves that measurably change the board.
   The decisive move is a cascade: one claim felled four.

### Toy B: relay (built, cycle 3)

**Rule.** Two parties driven by language models split 10 units. Every
message passes through the player, who may deliver it as written or
reworded (at most 8 changed words a message, 24 a game). A language model
judges one thing: how the receiver hears the delivered words, given how it
already feels (warmer, no different, colder, insulted). The ledger does the
rest: warmth moves the receiver's floor (`floor + round((0.5 - warmth) * 4)`),
an offer at or above the floor is accepted, a party at warmth 0.1 or less
walks away, and at signing the number each side actually wrote is compared
with what the other accepted, so a changed number sinks the deal.

**Stakes.** The relay earns a fee for any deal that holds and a bonus if its
secret client (Blake) gets at least 6; an exposed deal costs a point.

**Why it is generative.** The same offer lands differently depending on its
words and on how the receiver already feels, so the relay plays tone in two
directions at once: warm the side that must concede, cool the side that
must hold out.

**Playtest notes (cycle 3).**

1. The first skilled bot warmed the client and closed at once, below the
   target: a value function that holds out while messages remain fixed it.
2. MOCK audit (16 seeds): skilled 100%, random 12%, passive 0%; median 3
   moves with an effect per turn.
3. OLLAMA qwen3:8b: the first party prompt produced third-person narration
   ("offers Blake 5 units...") and an opening at 5; the prompt now asks for
   the exact first-person words and states the party's interest. With it, a
   4-message live game ended 6-4 for the client (bonus), 251 s.
4. The live model reads tone in context: a plain "let's see if we can reach
   a mutual agreement" landed colder on a party already cool. The mock's word
   counting cannot do that.

### Toy C: orders (built, cycle 4)

**Rule.** The commander (the player) gives one standing order of at most 12
words and may issue a new one only 3 times a game (the first included). Each
turn, a language model reads the order for each of three crew members
(bold, cautious, literal) against that turn's situation (quiet, a leak, a
storm) and the member's condition, and picks repair, gather, build or rest.
The ledger applies the actions: a repair closes a leak, an open leak adds 2
damage a turn, gathering fails in a storm, a build spends a supply for 2
progress (1 when worn out), and work wears people out. Win: project 20 by
turn 12 before 10 damage. Seeds shift the event schedule.

**Why it is generative.** One order is read three ways, in situations the
commander did not list; a word that settles one turn misleads the next.

**Playtest notes (cycle 4).**

1. MOCK audit (16 seeds): skilled 100%, random 31%, passive 0%; median 8
   moves with an effect per turn (the order list). The skilled bot plans by
   playing each order forward to the last turn.
2. Emergent under the mock: told "Build", the bold member gathers when the
   stores run dry; the literal member idles under any order that says
   nothing about the turn's situation.
3. OLLAMA qwen3:8b as commander and crew (473 s): the model wrapped orders
   ("Answer: Commander orders: ...") and ran over 12 words, wasting turns;
   moves are now unwrapped. Its order "Rest to recover energy and continue
   building the project." froze the whole crew in rest for 8 turns, a
   reading the lookup table cannot make. Lost at 14 of 20.

### The choice (cycle 6, provisional until the owner's games)

The Sandbox Prompt Test asks whether poking a rule is fascinating in plain
text. Three kinds of evidence:

| Evidence | claims | relay | orders |
|---|---|---|---|
| Bots, MOCK, 16 seeds (skilled / random / passive, %) | 100 / 0 / 0 | 100 / 12 / 0 | 100 / 31 / 0 |
| Moves with a measured effect per turn (median) | 6 | 3 | 8 |
| Distinct outcomes from 6 phrasings of one intent, MOCK | 2 | 2 | 3 |
| ... under qwen3:8b | 1 | 2 | 2 |
| What the live model did | the thesis fell whatever was said (opening balance; pairwise readings hard to predict) | every 3-of-10 read "colder" except one jab ("insulted"): the number drowns the words | the literal member built under "unless the hull is breached" and "leaks are Bea's job", repaired under phrasings that cover damage |
| Owner's games | none yet | none yet | none yet |

**Kept: orders.** Under a real model its wording moves outcomes in ways a
player can read and predict (who does what, and why), every turn gives a
measurable result, and the rule invites exactly the pokes the test looks for:
"what if I never mention leaks?", "what does the literal one do with
'unless'?". Claims is the most novel mechanic but its pairwise judgements are
opaque under real models and its opening is unbalanced; relay's lever is
drowned by the numbers. Both stay playable on the phone page until the
owner's games confirm or overturn the choice; they will be retired, not
deleted, in the Phase 5 rule audit.

## 2b. Orders, Phase 2: one order, five pressures (cycle 7)

A model explorer in free play (qwen3:8b, 12 turns, 12 distinct orders, no
holds) reached for actions the menu lacked ("survey the perimeter", "monitor
the horizon") and began assigning work by name. Both became pressures on the
one input:

| Pressure | What an order now carries | Rule component (`toys/orders/rules.py`) |
|---|---|---|
| Directive | what each member does, as a model reads it | `CrewRule` (effects in fixed crew order) |
| Cost | a new order costs 1 command point plus 1 per 4 words; 1 point back a turn, up to 5 | `CommandRule` |
| Information | the turns ahead are hidden; a member on watch reveals the next 2 | `WatchRule` (+ the new action "keep watch") |
| Timing | conditional clauses ("if a leak opens...") apply only when they hold | the model's reading (`CrewRule`) |
| Side effect | naming members ("Bea, keep watch") leaves the others to their temperaments; long orders cost turns of silence later | the model's reading; `CommandRule` |

The components meet only in the ledger (`Ship`) and run in a fixed order
each turn: the order's price, the hazards, the crew, the watch, the bleed,
the regeneration. Forecasts know only what the commander knows: turns
nobody has watched are taken as quiet.

MOCK audit (16 seeds): skilled 100%, random 19%, passive 0%; median 11
moves with an effect per turn.

## 2c. Orders, Phase 3: the matrix as laws (cycle 8)

**Ontology as data.** Temperaments are properties (data/orders.yaml
`temperaments`: what they do otherwise, when worn, with a leak open,
whether they skip an ordered action that cannot help), and actions are data
(`needs`, `gives`, `fatigue`, `spoiled_by`, `worn`, `closes`). One law in
`rules.CrewRule` applies every action; the mock applies the laws of reading
by temperament properties, with no case for any temperament or action pair.
Making effects data changed nothing: MOCK runs byte-identical (36 games).

**The prompt is written from the same data.** Each crew member's prompt
carries their habits (rendered from the properties) and the four laws of
reading (data `reading_laws`). The model is given the law, not the outcome.

**Two matrices** (data/orders_matrix.yaml; `python -m examples.crucible.audit`):
8 systems and 16 system cells; 5 operators (directive, conditional, named,
watch, silence) x 8 properties (quiet, leak, storm, worn, no supplies, bold,
cautious, literal), 15 reading cells: 31 cells, each tied to a test or golden
cases.

**Golden sets** (21 cases, 10 of them hard laws; `golden.py`):

| Model | Hard laws kept | Soft agreement | Passed |
|---|---|---|---|
| MOCK | 10/10 | 100% | yes |
| qwen3:8b, the provider's structured choice | 4/10 | 82% | no |
| ... laws restated in data | 6/10 | 82% | no |
| ... with a reason before the letter (contrib `ReasonedChoice`, library round 3) | 10/10 | 91% | yes |
| DeepSeek-V4.1-Flash (Together), one-step reading | 9/10 | 91% | no |
| ... two-step reading (which clause applies, then the action) | 10/10 | 91% | yes |

The decisive fix for the local model was not the prompt but the path: asked
to explain, qwen3:8b read the order correctly; forced to answer with a bare
letter through the provider's JSON choice, it did not.

**Balance after the laws.** Smarter temperaments made the crew manage
itself (random play won 88%). A sweep (`PROG/sweep_orders.py`) chose target
24 and 3 starting supplies: skilled 100%, random 25-31%, passive 0%.

## 2d. Orders, Phase 4: the first arc (cycle 10)

Each beat is a scene: a scenario in data/orders.yaml with its own crew,
menu, turns, goal, events and command, run as its own simulation (the beats
share no state, so Concordia's in-simulation `SceneTracker` is not needed;
the knobs are the same ones a `SceneSpec` holds).

| Beat | Scene | What changes | Bots, 16 seeds (skilled / random / passive / habit, %) |
|---|---|---|---|
| Isolate | One hand | Cy alone (literal), build or rest, 3 turns, 1 command point: only a short order that says "build" works | 100 / 50 / 0 / 0 |
| Develop | Two hands | Ash (bold) and Cy, scarce supplies, unseen leaks, little command | 100 / 50 / 0 / 100 |
| Twist | Supply run | the port wants stores, not progress: the habit "Build; if a leak opens, repair it." spends the goal, and bold Ash's instinct to build works against you; the same syntax repurposed ("Gather supplies; if a leak opens, repair it.") wins | 100 / 12 / 0 / 0 |
| Catharsis | The voyage | everything at once: three temperaments, leaks, storms, the watch, command | 100 / 31 / 0 / 38 |

The habit bot repeats the lesson of the earlier scenes; the twist is built
so that it loses. Stars: 1 for the win, 2 and 3 for the scene's conditions
(fewer turns, less damage, fewer command points spent), so a short order
that holds is worth more than a long one.

The first twist tried (storm season) failed: the crew handled storms on its
own, and the habit still won. A twist has to turn the habit into the cause
of failure, not just make the world harder.

## 2e. Orders, Phase 4: the second arc, information (cycle 12)

The turns ahead are hidden; a member on watch reveals the next two. A new
hazard makes knowing matter: a squall does damage unless someone kept watch
the turn before. Hazards became data under one law (`HazardRule`: `opens`,
`strikes_unwatched`).

| Beat | Scene | What changes | Bots, 16 seeds (skilled / random / passive / habit, %) |
|---|---|---|---|
| Isolate | Lookout | Bea alone; the goal is a forecast (turns seen ahead); only a watch sees | 100 / 12 / 0 / 0 |
| Develop | Squalls | three crew, hidden squalls and a leak; a watcher costs a pair of hands | 100 / 69 / 0 / 100 |
| Twist | Night watch | squalls in pairs; one turn on watch wears anyone out, so cautious Bea, the natural watcher, rests the next turn and the watch lapses; a bold or literal watcher (who keeps working when worn) holds it | 100 / 31 / 0 / 19 |
| Catharsis | The crossing | leaks, storms, squalls, three temperaments, 14 turns | 100 / 0 / 0 / 0 |

Two lessons from tuning. First, a twist can fail by coincidence: with
squalls every other turn, Bea's watch-rest cycle fell exactly into step
with them and the habit held. Pairs of squalls break the rhythm. Second, a
commander who plans only on what it has seen never values a watch; the
skilled bot now plans against every schedule the premise allows (the
scene's events, shifted 0-2 turns), as a player who read the premise would.

## 2f. Phase 5: Twelve Words (cycle 13)

The orders game is titled **Twelve Words**, after the order limit at its
heart. The fiction stays as thin as the systems allow: a commander, a ship,
three crew members who read one standing order three ways.

**Semantics follow systems.** `data/orders_look.yaml` gives every event,
temperament, condition, action and stock one glyph, one word and one colour
token, and maps every matrix property to its look; `look_test` fails if a
kind lacks a look or two kinds share a glyph or a word. The page draws only
from these.

**Every fact has a visual home.** `OrdersToy.view()` is the commander's view
as data: the turn and its situation, the turns ahead as far as the watch has
seen (unseen ones drawn as "?"), stores, hull and goal bars, command points,
the standing order, and each crew member's temperament, condition and last
action. `look_test.ViewTest` checks that every fact the commander's prompt
states is in the view and that the view shows nothing the watch has not
seen. A preview draws each member's predicted action on their card. The
whole prompt stays one tap away: text is the record, the board is the
reading.

Screenshots at the Pixel's viewport, light and dark: `PROG/evidence/shots/c13/`.

## 3. Prior art (cycle 1 search)

| Work | What it does | How the toys differ |
|---|---|---|
| Hearsay, Sindome's rumour system, The Mill | NPCs retell rumours with language models; believed rumours come true | Toy A has no NPC gossip: claims interact through judged relations and a weighted argument graph |
| IF:CARGO (arXiv 2608.12195) | Natural-language IF/THEN rules compiled to a command schema | Toy C keeps the order as language and lets each entity reinterpret it per situation |
| "Wargame agents" (seangoedecke.com) | Text orders to generals, no direct control | Toy C is a single standing order against a ledger, judged by robustness across situations |
| Minister Plenipotentiary | Diplomatic negotiation | Toy B's player is not a party; the move is rewording someone else's message |
| Abstract argumentation (Dung; weighted and bipolar semantics) | Formal acceptability of arguments | Toy A uses weighted bipolar semantics as game physics, with relations judged by a language model from free text |

Sources:
[IF:CARGO](https://arxiv.org/pdf/2608.12195),
[wargame agents](https://seangoedecke.com/wargame-agents/),
[LLM rumour spreading](https://arxiv.org/pdf/2502.01450),
[The Mill](https://potionhoard.itch.io/the-mill).

## 4. Concordia mapping

| Need | Concordia piece |
|---|---|
| Running a game | `prefabs/simulation/generic.Simulation`, `Sequential` engine |
| Game master | contrib `prefabs/game_master/rulebook` (contributed in cycle 2): `SwitchAct` with the well-known keys, `Terminate`, `MakeObservation` on an `ObservationQueue`; each toy is its `Rulebook` |
| Entities | contrib `prefabs/entity/seated` (cycle 2): `minimal.Entity` that reads its act component factory from params |
| The player's seat | `HumanActComponent`; bots are contrib `scripted_input.PolicyInput` readers (cycle 2), so people and bots play through one path |
| Model-played sides | `ConcatActComponent` (the minimal entity's default) |
| Models | contrib `language_model_setup` (Ollama, Together), `RetryLanguageModel`, `CallLimitLanguageModel` |
| Logs | `SimulationLog.to_html` in each run folder |
| The phone page | stdlib HTTP server; the seat's reader is contrib `BlockingInput` (cycle 5); telemetry as JSON lines per game |

## 5. Decisions

| Cycle | Decision |
|---|---|
| 1 | Slug `crucible`; toys share one harness and one generic game master. |
| 1 | Bots sit in the human seat as `HumanInput` readers instead of a new act component. |
| 1 | Claims physics: additive support, claims need support to stand, restatements fold. |
| 1 | In the sandbox the rival moves first. |
| 2 | The toy game master, seat prefab and policy reader went to contrib (library round 1); the game uses them unchanged (MOCK runs byte-identical, 12 games). |
| 3 | Relay: the model judges only how words land; acceptance, walkouts and the signing check are ledger rules. |
| 3 | A preview never tells anyone anything (the toy's outbox is restored). |
| 4 | Orders: crew decisions are the toy's model judgements (not entities), so an order's effect can be previewed and planned. |
| 4 | Moves wrapped in a label and quotes are unwrapped for every toy. |
| 5 | The phone page is plain text until Phase 5: the whole prompt, a measured preview of a draft, the receipts. The person picks mock or a local model per game. |
| 6 | Orders kept (provisional): best legibility of word-driven outcomes under a real model. |
| 7 | Orders overloaded: command points by length, hidden future and a watch, named clauses; revisions replaced by points. |
| 8 | Temperaments and actions as data; laws of reading in every crew prompt; two-step reading; ReasonedChoice for local models; voyage target 24, 3 supplies. |
| 10 | First arc: One hand, Two hands, Supply run (goal: stores), The voyage; stars; a scene ends when its goal is met (stores at the end); the mock chooses only actions the scene offers. Cells added: stores->command, command->command. |
| 11 | Campaign: chapters open with a star in the one before; best stars kept per scene on the server; a daily challenge (the voyage, seeded by the date). The page shows chapters, locks and stars. |
| 12 | Arc 2 (Lookout, Squalls, Night watch, The crossing): squalls; hazards as data; scenes name their own habit; the skilled bot plans over the premise's schedules. Cells added: hazards->hull, watch->hull, fatigue->watch, watch->project; reading cell watch x worn. |
| 13 | Title: Twelve Words. One look per kind (`orders_look.yaml`); the board draws `view()`; previews mark each member's next action. |

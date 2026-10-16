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

## 2g. The rule audit (cycle 14)

An independent pass walked every data key, lexicon word, rule, function and
page field of Twelve Words. Every data key was read; the findings were
values never exercised, dead helpers and four real defects.

| Kind | Finding | Done |
|---|---|---|
| Defect | Lookout told a model "project X of 4"; its goal is turns seen | the prompt states the forecast goal; a forecast look (`◔`) |
| Defect | condition words doubled as verbs: "If tired, build" made a member rest | a conditional clause's condition words are not orders; golden g24, g25 |
| Defect | "if a squall..." could never apply | squall is a condition; one helper gives the conditions now |
| Defect | One hand's and Lookout's star conditions could never fail | stars now count words spoken (the game's currency): one word for three stars in One hand |
| Orphan | `two_step_reading` (always on), `max_turns`, condition `low`, keyword `once` (not in the laws the prompt states) | cut |
| Orphan | habits equal to the default, or never checked (Supply run, Lookout, Squalls, Crossing) | cut |
| Orphan | `split_putative_event`, `current_actor`, `rollout`, `play(quiet=False)`, rule defaults never reached, preview `steps`, view fields nobody reads | cut |
| Unused but useful | the mock's call count; the open-leak flag; the habit bot | the run summary reports model calls; the board marks an open leak; `play_audit --policies` |
| Kept | single-instance laws (`needs`, `spoiled_by`, `opens`, ...), empty hazard kinds, verb synonyms for typed orders | laws are general even with one instance; every ledger kind has a look; people type what they like |

The claims and relay toys stay as the record of the toy stage (Phase 0);
they are not rules of Twelve Words.

## 2h. Judgements must be repeatable (cycle 15)

Re-running the golden set (now 25 cases, 11 hard) on DeepSeek-V4.1-Flash gave
11/11 hard laws on one run and 9/11 on the next: the crew's readings were
sampled at the provider's default temperature (1.0). A reading of an order
is a judgement, not a story, so it is now sampled cold: the which-clause
question at `reading_temperature` (0.0) and the choice through the contrib
`ReasonedChoice` at temperature 0 (library round 4). Asking the model to
explain also exposed a slip between reason and letter ("so he builds",
then the gather option's letter); the wrapper now trusts the option's words
over a slipped letter.

| Model (25 cases) | Hard laws | Soft agreement | Passed |
|---|---|---|---|
| Flash, default path | 11/11 | 71% | no |
| Flash, reason first, temperature 1 (two runs) | 11/11, 9/11 | 93%, 79% | yes, no |
| Flash, reason first, temperature 0 (two runs) | 11/11, 11/11 | 71%, 71% | no, no |
| Flash, + words over letters (two runs) | 11/11, 11/11 | 86%, 79% | yes, yes |
| qwen3:8b, reason first, temperature 1 | 10/11 | 100% | no |

The provider is not fully deterministic even at temperature 0 (the failing
soft cases change between runs), so the golden set is run at least twice
per model and both runs must pass.

## 2i. Arc 3: Dex, and the word "not" (cycle 15)

A fourth crew member, Dex, is **contrary**: he does the opposite of a
clause that applies (told to work, he rests; told to rest, he builds; told
not to do something, he does it). Contrariness is a temperament property
(`contrary`, `works`), the prompt states it, and the mock reads it with the
same law as everyone else. To give him something to reverse, orders gained
a new operator: **negation** ("not", "never", "don't" forbid the action they
name), which every temperament reads differently: the bold do what they do
otherwise, the literal are left with nothing to do, the contrary do exactly
the forbidden thing.

| Beat | Scene | What changes | Bots, 16 seeds (skilled / random / passive / habit, %) |
|---|---|---|---|
| Isolate | Contrary | Dex alone, 1 command point: "Build." makes him rest; "Rest." or "Don't build." makes him build | 100 / 31 / 0 / 0 |
| Develop | Mixed hands | Ash and Dex read one order: what moves one stops the other | 100 / 44 / 0 / 0 |
| Twist | The mender | the habit "Dex, rest; everyone else build." gets Dex building while leaks run; he must be told "don't repair" to mend | 100 / 6 / 0 / 0 |
| Catharsis | All hands | all four, squalls, leaks, storms, 14 turns | 100 / 6 / 0 / 0 |

Matrix: +1 operator (negation), +1 property (contrary), +6 reading cells;
golden set 33 cases (17 hard), the mock 33/33. Adding Dex's orders to the
bots' repertoire made random play stronger in Squalls (81%), so Squalls was
retuned (target 18 from 2 supplies: 100 / 38 / 0).

## 2j. The reading on the card (cycle 17)

Each crew member's reading has two steps: which clause of the order applies
to them, then what they do. The first answer used to be thrown away; it is
now kept and drawn on the member's card under what they did ("read
“everyone else build”", or "by temperament" when no clause applied). The
commander sees *why* the crew acted, so a misreading can be answered with
the next order instead of guessed at. The mock quotes the clause it applied
by the same laws, so the cards are deterministic in CI.

Showing the answers exposed a fault: on qwen3:8b every clause answer had
been empty. A thinking model's reply opens with its thinking, and Ollama
applies stop sequences to the thinking too, so the newline that ends an open
question ended the reply before the answer. Library round 5 gives the
contrib Ollama wrapper a `think` option: with `True` the thinking is kept
apart and the terminators apply to the answer.

| qwen3:8b, golden set (33 cases, 17 hard) | Hard laws | Soft agreement | Time per crew turn |
|---|---|---|---|
| thinking off (one run) | 11/17 | 69% | ~9 s |
| thinking kept apart (run 1; run 2 hung on one request after 32 cases) | 17/17 | 94% | ~86 s |

Matrix: crew -> command added (the commander reads the crew's readings);
conditional x leak and named x bold are now checked by a rule test as well
as golden cases (the audit counts such cells: 2).

## 2k. Arc 4: the word "someone" (cycle 18)

Naming a member costs no more words than "someone" ("Bea, keep watch" and
"Someone keep watch" are both three), but it binds the order to one person
and their temperament. A clause for **someone** asks one member only: of
those willing to take it up, the freshest (lowest fatigue, the first in crew
order on a tie); a member named in another clause does not take it up.
Willingness is a temperament property (`volunteers`): the bold always, the
cautious only when fresh, the literal and the contrary never ("someone"
names no one).

Every member reads alone, at the same moment, so the readings must agree
without talking. The first version gave each reader the crew's fatigue and
the law, and asked them to work out whether *they* were the freshest
willing member. DeepSeek-V4.1-Flash could not do it reliably (hard laws
23/23 then 21/23; written as three explicit steps, 20/23 and 19/23): two
members volunteered, or none. Counting fatigue is arithmetic over the
ledger, like an order's price, so the ledger does it (`CrewRule.volunteer`)
and every reading is told who takes it up this turn. What is left to the
model is the reading: recognising a clause for someone ("one of you",
"whoever is fresh") and that it does not apply to them. With that, Flash
kept 22/23 hard laws twice; the one miss was Bea reading "build" right and
then judging that building with 3 supplies "cannot help", a temperament
misreading the golden set had never isolated. The case became soft, a
literal member (who has no such judgement) took its hard place, and the
crew prompt now says what "cannot help" means (building with no supply,
repairing a sound hull, gathering in a storm): 23/23 twice, soft agreement
72% and 89% (the bar is 75%; Flash's soft answers lean to "gather"). The
law and the name go into a reading only when the order says someone, so
every other prompt is unchanged apart from that sentence.

What it changes in play: under "Someone keep watch; everyone else build."
the watch passes from Ash to Bea and back as they tire, where "Bea, keep
watch" lapses whenever Bea is worn out (Night watch: the named watch kept 3
turns of 6, the someone watch 6). The price is the builder: whoever is
freshest is taken off the project, and the bold keep volunteering while worn
out, building at half when they return. With no willing member (only the
literal and the contrary aboard) nobody does it at all.

Matrix: +1 operator (someone), +5 reading cells, +2 system cells (fatigue ->
crew, crew -> watch); golden set 41 cases (23 hard), the mock 41/41. The
bots' repertoire gained three someone orders; at 16 seeds every scene still
gives skilled 100, random 0-44, passive 0 (evidence/audit-c18.jsonl).

## 2l. Arc 4: the rota (cycle 19)

The first draft of an isolate scene showed that "someone" was a weaker
order than a name: the bold keep watching when worn out, so "Ash, keep
watch" never lapses and "someone" only cost a builder (scene lab:
`PROG/scene_lab.py`, `PROG/lab/*.yaml`). A fixed assignment fails only when
the person is not there, so arc 4 adds a **rota**: a scene can put members
off duty on given turns (`off_duty`), when they rest and hear no orders. A
named duty lapses then; a clause for someone passes to whoever is on. The
situation says who is off duty, the board marks it on the turn cells and the
crew cards ("heard no order"), and a scene may open just after a watch
(`start.watched`), so a squall on turn 1 is fair.

| Beat | Scene | What changes | Bots, 16 seeds (skilled / random / passive / habit, %) |
|---|---|---|---|
| Isolate | The rota | Ash and Bea, squalls every turn, Bea off on 2-3 and Ash on 4-5, command for one order: a watch named on either lapses; "Someone keep watch; everyone else build." holds it all night (a rush that ignores the watch wins only if the squalls come late, at 1 star) | 100 / 6 / 0 / 0 |
| Develop | Watches | three crew, squalls and leaks, a rota for each; orders can be changed | 100 / 25 / 0 / 19 |
| Twist | No volunteers | Ash, Cy and Dex; Ash off on 3-5: the habit "Someone keep watch" finds no one willing (the literal and the contrary never volunteer); name Cy to the watch, and tell Dex to rest so that he builds | 100 / 6 / 0 / 0 |
| Catharsis | The long night | all four, each off duty for two turns, squalls, leaks and storms over 12 turns; from cycle 21 an unwatched squall also hurts the most worn member on duty | 100 / 0 / 0 / 0 (before injuries: 100 / 19 / 0 / 56) |

The campaign is 16 chapters. Matrix: +1 system (rota) with 3 cells, +1
property (off duty) with 2 reading cells; golden set 43 cases (25 hard).
Every earlier scene's MOCK games are byte-identical to cycle 18.

## 2m. Deep pass (cycle 20)

- **Play audit**, 16 seeds, all 16 scenes: skilled 100%, random 0-44%,
  passive 0% (`PROG/evidence/audit-c19.jsonl`).
- **Moves with a measured effect per turn** (median, 4 seeds): 23 in ten
  scenes (every affordable order changes the outcome), 20 in Two hands, 10
  in No volunteers, and 0 in the four isolate puzzles (One hand, Lookout,
  Contrary, The rota), which give command for exactly one order: one
  decision, made well, is the lesson. Over the campaign's scenes the median
  is 23.
- **Matrix audit**: 58 cells, every one evidenced; 3 by both a test and
  golden cases.
- **Golden sets**: on 43 cases DeepSeek-V4.1-Flash kept 24/25 hard laws on
  both runs; g43 (Bea under "Ash, keep watch; everyone else build" with Ash
  off duty) gathered once each run and built when probed: the cautious
  "cannot help" misjudgement seen in g35, not the law under test. As with
  g35, the hard case now uses the literal Cy and Bea's case is soft (g44).
  On 44 cases (25 hard): the mock 44/44; Flash 25/25 hard twice, soft 79%
  both, PASS both. qwen3:8b: (running; a golden run on this CPU-only host
  takes about 30 minutes, and one crashed when a reading thought for over 3
  minutes three times: the Ollama timeout is now 600 s and a failed call
  fails its case instead of the run).
- **Live games**, Flash as commander and crew, in The rota, Watches and No
  volunteers: all lost; $0.009, $0.024 and $0.010 a game. The model
  commander never used "someone" (it named members and re-issued the same
  named order), and twice its "order" was narration over the word limit,
  refused by the rules.
- **Simplify**: three helpers nothing used were cut.
- **Cost**: ≈ $1.17 of $10 so far; a golden pair on Flash ≈ $0.12; a live
  game ≈ $0.01-0.02 (crew readings are two calls per member per turn).

## 2n. Injuries: a rota that moves (cycle 21)

A scripted rota can be planned around with names; a rota that moves cannot.
In a scene whose squall `injures`, a squall no one watched for also hurts
the most worn member on duty (the tired are careless), who is off duty for
that turn and the next. A watch now protects the crew as well as the hull,
fatigue decides who is lost, and a named duty can break on a turn no one
planned, while a clause for someone passes to whoever is left. Only The
long night has it: there the best fixed order of the scene lab ("Cy, keep
watch; Dex, rest; everyone else build.") now loses one schedule in three,
random play never wins and the habit never wins (16 seeds: skilled 100,
random 0, passive 0, habit 0). Matrix: hazards -> rota, fatigue -> rota,
watch -> rota (61 cells).

## 2o. Orders are heard (cycle 23)

An order reaches only the crew on duty when it is given. A member off duty
(on the rota or hurt) keeps the last order they heard and comes back to it;
the receipt says who missed it ("Bea is off duty and misses it."), their
card says what they are still on, and the commander's prompt lists them.
Saying an order again reaches those who missed it, at its price again. A
clause for someone is taken up only by a member who heard it. The timing of
an order now matters on a rota: change it when everyone is aboard, or pay
twice. The bots' rates in the rota scenes did not move (they rarely change
orders while someone is off), so this is a lever for people more than a
balance change. Matrix: rota -> command, command -> crew, command -> watch
(64 cells).

## 2p. Every scene names what it teaches (cycle 24)

Each campaign scene now lists the matrix cells it exercises (`teaches`:
`from->to` for a system cell, `operator x property` for a reading cell) and
one sentence it teaches (`lesson`), shown on the page when the scene is
over, won or lost. The audit checks every reference and reports the cells
no scene teaches: all 59 cell references (system pairs and reading cells)
are taught by at least one of the 16 scenes. A cell no scene reaches would
be a rule the campaign never asks a player to learn.

## 2q. Deep pass (cycle 25)

- Play audit, 16 seeds, all 16 scenes (`evidence/audit-c25.jsonl`):
  skilled 100%, random 0-44%, passive 0%; only The long night moved since
  cycle 19 (injuries: random 19% -> 0%, habit 56% -> 0%).
- Matrix 64 cells, 5 checked both ways; 59 of 59 cell references taught by
  some scene.
- Golden prompts are unchanged since cycle 20, so DeepSeek-V4.1-Flash's
  two passes on the 44-case set stand; the mock passes; qwen3:8b: running.
- Simplify: the rota logic (who is on duty, which order each member heard)
  moved from the toy into `rules.RotaRule`, with its own tests; the MOCK
  snapshot is byte-identical.
- Cost: no paid calls this cycle; ≈ $1.29 of $10 in all.

## 2r. Golden sets on two kinds of model (cycles 28-30)

- qwen3:8b (Ollama, thinking kept apart) on a CPU-only host: a "someone"
  reading thought for ~4,700 tokens (4 minutes; 10+ when the CPU was
  shared). Told plainly whether the clause was theirs, ~2,800. On 44 cases
  it then kept 23/25 hard laws and every soft case; both misses were
  someone cases, and one showed a real gap in the law text: a clause for
  someone was never said to come before the plain clauses for its taker.
  It now says so (the mock always read it that way), and on the fixed law
  qwen3 kept 24/25 hard laws with every soft case agreeing; g36 now
  passes. The one miss was g38: told a clause for someone was not his, the
  literal Cy still repaired under "Someone repair the hull.", because by
  the first law (no names, no condition) the clause looked plain. In orders
  that say someone, the someone law now follows the plain-clause law and
  says it is not one, and each reader's habits say whether they
  volunteer ("Cy never takes up a clause for someone."). One probe then
  passed g38, but full runs did not: qwen3:8b reads the literal as obeying
  "repair the hull" whatever the laws say (it calls the clause "plain" two
  lines after a law saying it is not), 24/25 hard in every run. The law
  stays; the miss is recorded. (That volunteer sentence was later removed:
  with it qwen3 fell to 22/25.) On the 49-case set with the bosun's cases
  (cycle 45 wording), qwen3:8b kept all 28 hard laws with every soft case
  agreeing in one run, and 27/28 (g38 again) in the other: g38 is flaky for
  it, not a fixed misreading. Flash on the same wording passes both runs
  (28/28, soft 81%).
- DeepSeek-V4.1-Flash keeps all 25 hard laws in every run since cycle 20.
  Its soft agreement (temperament judgements, bar 75%) is 74-94% run to
  run, on prompts that did not change: 79/79, 79/79, then 74/74. A probe
  showed the slip: the reason says "Ash builds ... there are 3 supplies",
  the answer is "(b) gather supplies"; every soft miss of that run was
  option (b). Asking the reason to name the option (84, 79, 74 over three
  runs) was within the spread and was not adopted. The bar stays where it
  is; the record keeps every run.

## 2s. A storm hides the horizon (cycle 33)

An event can shorten the watch (`hazards.<event>.sight`): in The crossing
and The long night a watch kept in a storm sees one turn ahead, not two, so
the turn after a storm is the one a commander is most likely to meet
blind. The bots' rates did not move (crossing 100 / 6 / 0 / 0, long night
100 / 0 / 0 / 0 at 16 seeds); what changes is what a person can plan
around. Matrix: hazards -> watch, and watch x storm now checked by a test
as well as a golden case (65 cells; 60 of 60 references taught).

## 2t. Arc 5: the bosun (cycles 41-43)

In a scene with a bosun, the commander's order goes to one member, who
passes it on to the others in their own words, an open question to the
model (cached, held to 16 words). The bosun follows the order as given; the
rest of the crew on duty hear only the bosun's version, and the receipt,
the cards ("heard “…”") and the commander's prompt say what was said. Each
temperament relays by a law the prompt states and the mock follows: the
literal word for word, the bold without clauses to rest or keep watch, the
cautious adding "repair leaks", the contrary with "don't" after whom each
clause addresses ("Cy, don't keep watch"). Probed, DeepSeek-V4.1-Flash
relayed exactly by these laws for all four.

| Beat | Scene | What changes | Bots, 16 seeds (skilled / random / passive / habit, %) |
|---|---|---|---|
| Isolate | The bosun | Ash (bold) relays to Cy; squalls all night; one order: "Build; Cy, keep watch." reaches Cy as "Build."; the watch must go to the bosun himself (or to "someone", which only he heard) | 100 / 25 / 0 / 0 |
| Develop | Word of mouth | Bea (cautious) relays to Ash and Cy and adds "repair leaks": leaks get mended without the words for it | 100 / 12 / 0 / 44 |
| Twist | The contrary bosun | Dex relays with "don't" and himself does the opposite: the habit "Dex, keep watch" makes him rest; "Don't keep watch." makes him watch and reaches the crew unchanged | 100 / 6 / 0 / 0 |
| Catharsis | Chain of command | all four, Cy (literal) as bosun, off duty on turns 5-7 (orders then go out as given), squalls, leaks, storms | 100 / 25 / 0 / 0 |

The campaign is 20 chapters. Matrix 71 cells (+relay operator, 4 reading
cells, 2 system cells), 64 of 64 references taught; golden set 49 cases (28
hard): the mock passes, Flash passes twice.

## 2u. The wind takes words (cycle 51)

In a scene whose storm `garbles`, an order given during a storm carries
only its first clause: everyone on duty, the bosun too, hears only the
words before the first ';' or '.', and the receipt says what carried. The
vital clause must come first, or the order must wait for the calm, which a
watch can see coming (and in these scenes a storm also shortens the
watch). On in The crossing, The long night and Chain of command; at 16
seeds they stay skilled 100, random 0-12, passive 0. Matrix: hazards ->
command and hazards -> crew (73 cells, 65 of 65 references taught).

## 3. Prior art (cycle 1 search)

| Work | What it does | How the toys differ |
|---|---|---|
| Hearsay, Sindome's rumour system, The Mill | NPCs retell rumours with language models; believed rumours come true | Toy A has no NPC gossip: claims interact through judged relations and a weighted argument graph |
| IF:CARGO (arXiv 2608.12195) | Natural-language IF/THEN rules compiled to a command schema | Toy C keeps the order as language and lets each entity reinterpret it per situation |
| "Wargame agents" (seangoedecke.com) | Text orders to generals, no direct control | Toy C is a single standing order against a ledger, judged by robustness across situations |
| Minister Plenipotentiary | Diplomatic negotiation | Toy B's player is not a party; the move is rewording someone else's message |
| Abstract argumentation (Dung; weighted and bipolar semantics) | Formal acceptability of arguments | Toy A uses weighted bipolar semantics as game physics, with relations judged by a language model from free text |

Update (cycle 27), for arcs 3 and 4:

| Work | What it does | How Twelve Words differs |
|---|---|---|
| "Los SIMS LLM" (Universidad Complutense de Madrid) | A space simulation in which the player speaks to the crew in natural language (Gemini) to run research, exploration and resources | Here the speech is the scarce resource: one standing order of at most 12 words, priced by length, read by four fixed temperaments under stated laws, with the consequences kept by a deterministic ledger and scenes built to teach one law at a time |
| UBOAT (Deep Water Studio) | A submarine simulation with crew shifts, fatigue and duty stations assigned by the player | Duties are not assigned but worded: a clause for "someone" goes to the freshest willing member on duty, an order reaches only those on duty, and a squall no one watched for hurts the most worn |

Sources:
[Los SIMS LLM](https://docta.ucm.es/entities/publication/29e1d8bf-43b7-4e6a-8bdf-8c43ba0638a1/full),
[UBOAT crew shifts](https://steamcommunity.com/app/494840/discussions/0/4331979876124626282),
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
| 14 | Live games on DeepSeek-V4.1-Flash (3 scenes, est. $0.05): the commander prompt states order prices, narrated orders are unwrapped. The rule audit (DESIGN §2g). Stars count words spoken. Cell added: conditional x worn. |
| 15 | Readings sampled at temperature 0; golden sets run twice per model (DESIGN §2h). Library round 4: ReasonedChoice temperature and words over letters. Arc 3: Dex (contrary) and negation; 12 campaign chapters. Cells added: 6 reading cells. |
| 17 | Crew cards show the clause each member acted on (DESIGN §2j). Ollama readers keep thinking apart (library round 5). Cell added: crew -> command; 2 cells doubly evidenced. |
| 18 | Arc 4 mechanics: a clause for "someone" goes to the freshest willing member, counted by the ledger and named in each reading (DESIGN §2k); "cannot help" spelled out in the crew prompt. Cells added: 2 system, 5 reading (operator someone). |
| 19 | Arc 4 scenes on a rota (off duty: rest, no orders): The rota, Watches, No volunteers, The long night; 16 campaign chapters (DESIGN §2l). Cells added: rota -> crew, rota -> fatigue, rota -> watch; someone x off duty, named x off duty. |
| 20 | Deep pass (DESIGN §2m): golden cases isolate laws from temperament judgements (g43 literal, g44 soft); a provider failure fails its case, not the run. |
| 21 | Injuries: in The long night an unwatched squall hurts the most worn member on duty, off duty for two turns (DESIGN §2n). Cells added: hazards -> rota, fatigue -> rota, watch -> rota. |
| 22 | The preview marks each crew card with the predicted action and the clause behind it. |
| 23 | Orders are heard: an order given while a member is off duty does not reach them (DESIGN §2o). Cells added: rota -> command, command -> crew, command -> watch. |
| 24 | Scenes name the cells they teach and a lesson, shown at the end; the audit checks coverage: 59/59 (DESIGN §2p). |

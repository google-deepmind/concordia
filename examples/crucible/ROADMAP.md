# Crucible: roadmap

Boxes mirror GUIDE.md's checklists. Each ticked box names its evidence.

## Pass it on (the rethink, cycle 98): the boxes, re-earned

The owner's verdict on the orders game ("I don't really understand what to
do at all"; "reconfigure it into a very different, and much more legible,
game") reopened the guide's checklists for the new game. Evidence below is
for Pass it on; the orders game's boxes follow as the record.

Phase 0, toys
- [x] Candidates compared and one kept, the reason in DESIGN.md (DESIGN 2x: the orders game failed the owner's Sandbox Prompt Test; Pass it on reuses its relay and reading as the whole game)
- [x] Runs end to end on the mock (`passiton/toy_test.LineTest.test_a_whole_game_through_the_harness`)

Phase 1, grey box
- [x] A mock run finishes in seconds and reproduces from a seed (`toy_test`, 28 tests in 3 s)
- [x] Every move shows a measured preview of its effect before it is sent (Rehearse with the first in line: how the first relayer would pass the words on, with the changes marked, no try used; the top's reading stays the puzzle. `toy_test.test_a_rehearsal_shows_the_first_hop_only`, `server_test.test_pass_it_on_rehearses_with_the_first_in_line`; checked in the browser pane, cycle 104)
- [x] Every move gets a receipt saying what it did (each try: every hop's words, what the last in line did, the verdict; `toy_test.test_dex_flips_each_part_...`, page cards, evidence/shots/c99)
- [ ] A tester with no win condition starts poking the system unprompted (the owner's Pass it on games)
- [x] The skilled bot beats the random bot (16 seeds, evidence/audit-c101-passiton.jsonl: 100 vs 0-62 in every level)

Phase 2, overloading
- [x] Every move carries at least two of directive, cost, side effect, information and timing (a message is a directive read at the top, golden p1-p13; a cost in words and tries, `test_the_skilled_bot_earns_three_stars_on_the_first_try`, `test_three_tries_then_over_...`; a side effect, each relayer reshaping it for everyone above, `test_the_last_in_line_hear_only_the_relayers_words`, golden r1-r10; and information, every hop shown and the rehearsal, `test_a_rehearsal_shows_the_first_hop_only`)
- [x] Each dimension is its own component with its own test (directive: `OrdersToy.reading`; side effect: `OrdersToy.relay` and the mock's relay laws, `orders_test`/`someone_test` relay tests; cost and verdict: `PassItOnToy` ledger, `toy_test`; information: `view`/`rehearse`, `toy_test`, `server_test`)
- [x] Simultaneous acts resolve in a fixed order (the last in line read in line order; deterministic receipts)
- [x] The median turn offers three or more moves with a measured effect (audit-c101: 120 to 14640 candidate messages with distinct outcomes)

Phase 3, interaction matrix
- [x] Properties live in YAML and component state, not only in prompts (relay laws and temperaments in `orders.yaml`/`passiton.yaml`, the line and goals in `passiton.yaml`; prompts are written from them)
- [x] Matrix cells live in YAML and an audit script counts them (25+) (`data/passiton_matrix.yaml`, `python -m examples.crucible.audit passiton`: 31 cells, no problems; `audit_test.test_pass_it_on_matrix_has_no_problems`)
- [ ] Every new feature adds or deepens two cells
- [ ] Golden sets pass on the mock, an Ollama model and a Together model (mock 22/22 x2 `audit_test`; DeepSeek-V4.1-Flash PASS x2, 20/20 hard, soft 100%, evidence/golden-passiton-flash-c103b-x2.json; qwen3:8b running)

Phase 4, pacing
- [x] Each arc has Isolate, Develop, Twist and Catharsis levels, gated by stars (`data/passiton.yaml`: arcs Mouth to ear and Dex in the middle; `campaign.chapters`)
- [x] Bots at 16 seeds show the skill gaps in every level (audit-c101: skilled - random 38 to 100, skilled - passive 100)
- [x] Telemetry keeps the move, what the player saw and the hidden truth (the server's move and end records; the end record's truth holds every try's hops)

Phase 5, wrapping
- [x] Every fact in a role's prompt has a visual home (the player's prompt: task, each relayer's temperament and law, who acts, the goal, the try, the word limit, all in the view the page draws; `toy_test.LookTest.test_every_fact_in_the_players_prompt_is_in_the_view`)
- [x] Each property has one look and one word, enforced by a test (each way of passing on has one phrase, each temperament a glyph, a word and a meaning, each action a glyph; `toy_test.LookTest.test_every_way_of_passing_on_and_temperament_has_one_look`)
- [x] The rule audit is done and orphan rules are cut (cycle 105, read-only audit of laws, mock, toy and page: 12 findings, fixed in 1a7460a; the situation cell and its golden case cut in cycle 103, the two Bea-habit levels in cycle 100; DESIGN 2x)

## Phase 0: toys

- [x] Shared harness: generic game master, seats, mock, CLI, play audit (cycle 1)
- [x] Toy A, claims, runs end to end on the mock (cycle 1: `crucible_e2e_test`)
- [x] Toy B, relay (cycle 3: `relay_test`, `crucible_e2e_test`; audit skilled 100 / random 12 / passive 0)
- [x] Toy C, orders (cycle 4: `orders_test`, `crucible_e2e_test`; audit skilled 100 / random 31 / passive 0)
- [x] A plain-text phone page with a human seat and telemetry (cycle 5: `web/`, `server_test`; https://openclaw-mac.taild21fee.ts.net/crucible/)
- [x] Sandbox Prompt Test of every toy: bots, a live model in the seat, the owner's games (cycle 6: bots and qwen3:8b wording probe, `PROG/evidence/wording-qwen3.json`; owner's games pending)
- [x] One toy kept; the reason in DESIGN.md (cycle 6: orders, provisional until the owner's games)

## Phase 1: grey box

- [x] A mock run of the core loop finishes in seconds and reproduces exactly from a seed. (orders: 0.1 s; `crucible_e2e_test.test_same_seed_same_game`)
- [x] Every move shows a measured preview of its effect before it is sent. (page Preview: the crew's actions this turn, measured on a copy; with the mock, a forecast to the end: `server_test.test_free_play_and_forecast`)
- [x] Every move gets a receipt saying what it did and when it will show. (each turn: who did what, and the ledger after; effects land the same turn)
- [ ] A tester with no win condition starts poking the system unprompted.
- [x] The skilled bot beats the random bot. (orders, 16 seeds: 100% vs 31%)

## Phase 2: overloading

- [x] Every action carries at least two of directive, cost, side effect, information and timing. (cycle 7: every order has a directive and a price; watch, conditions and names add information, timing and side effects; DESIGN §2b)
- [x] Each dimension is its own component with its own test; components meet only in the ledger. (`toys/orders/rules.py`, `rules_test.py`)
- [x] Simultaneous acts resolve in a fixed order. (`rules_test.CrewRuleTest.test_effects_apply_in_the_fixed_crew_order`)
- [x] The median turn offers three or more moves with a measured effect. (orders, 16 seeds: 11; cycle 20, all 16 scenes at 4 seeds: 23 in ten, 20, 10, and 0 only in the four one-decision isolate puzzles; DESIGN §2m)

## Phase 3: interaction matrix

- [x] Properties live in YAML and component state, not only in prompts. (cycle 8: temperaments and actions as data; prompts rendered from them)
- [x] No resolver branches on a specific operator and target pair. (cycle 8: one law in `CrewRule` over action data; the mock reads temperament properties; MOCK byte-identical)
- [x] Matrix cells live in YAML and an audit script counts them. (`data/orders_matrix.yaml`, `audit.py`, `audit_test`: 31 cells)
- [x] Every new feature added or deepened at least two cells. (cycle 8 laws of reading deepened hazards->crew, fatigue->crew, stores->crew; recorded per feature in DESIGN from now on)
- [x] Golden sets pass on every model you ship. (cycle 8: MOCK 10/10 hard, 100%; qwen3:8b 10/10, 91%; DeepSeek-V4.1-Flash 10/10, 91%)

## Phase 4: pacing

- [x] Each arc has Isolate, Develop, Twist and Catharsis scenes. (cycle 10, arc 1: One hand, Two hands, Supply run, The voyage; cycle 12, arc 2: Lookout, Squalls, Night watch, The crossing; cycle 15, arc 3: Contrary, Mixed hands, The mender, All hands; `beats_test`)
- [x] Bots at many seeds show the skill gaps you target in every scenario. (cycle 11, 16 seeds, skilled - random / skilled - passive: One hand +50/+100, Two hands +50/+100, Supply run +88/+100, voyage +69/+100, claims +100/+100, relay +88/+100; free play has no goal; `PROG/evidence/audit-c11.jsonl`)
- [x] Human-game telemetry keeps the move, the reading the player saw and the hidden truth. (`web/app.py`: each move with the question shown, answer time and previews; at the end the outcome and the whole state; `server_test`)

## Phase 5: wrapping

- [x] Every fact in a role's prompt has a visual home. (cycle 13: the board draws `OrdersToy.view()`; `look_test.ViewTest` checks every prompt fact is in the view; shots `PROG/evidence/shots/c13/`)
- [x] Each matrix property has one look and one word, and a test enforces it. (`data/orders_look.yaml`, `look_test.LookTest`)
- [x] The rule audit is done and orphan rules are cut. (cycle 14: DESIGN §2g; four defects fixed, orphans cut, MOCK audit unchanged: `PROG/evidence/audit-c14.jsonl`)

## Each upstreaming round

Round 1 (cycle 2; issue google-deepmind/concordia#418, PR google-deepmind/concordia#419):

- [x] Reuse check logged; an issue tagged `contribution` opened before the first component PR. (JOURNAL c2; #418)
- [x] No game names in the API; state checkpointable. (`rulebook_test` checkpoint round trip)
- [x] absltest tests beside each module, docstrings and a package README. (three `*_test.py`; the modules join existing packages, so no new README)
- [x] `pytest`, `pylint --errors-only` and `pyrefly check` pass; files formatted with `pyink`. (`PROG/check_lib.sh`, 1402 passed)
- [x] The game's mock runs are byte-identical before and after the switch. (`PROG/evidence/snapshot-c2-{before,after}.json`, 12 games)
- [x] Commits are CLA-clean, carry no `Co-Authored-By` trailer and touch library or game paths, not both. (3477499 library, f2f5367 game)
- [x] The PR's status note says what landed and what this round adds. (#419 body)

## Library rounds

| Round | Modules | State |
|---|---|---|
| 1 (cycle 2) | contrib `rulebook` game master (component + prefab), `seated` entity prefab, `scripted_input` readers | draft PR google-deepmind/concordia#419 |
| 2 (cycle 5) | contrib `blocking_input` (a `HumanInput` reader answered from another thread) | same PR, second commit |
| 3 (cycle 9) | contrib `language_models/reasoned_choice` (multiple choice through a short reason) | same PR, third commit |
| 4 (cycle 15) | `ReasonedChoice`: a temperature for the reasoned reply; the option's words win over a slipped letter | same PR, two commits |
| 5 (cycle 17) | `OllamaLanguageModel(think=...)`: a thinking model's terminators apply to its answer, not its thinking (open questions on qwen3 came back empty) | same PR |
| 5b (cycles 19, 26) | `OllamaLanguageModel(timeout=...)` for the client; a README for the Ollama adapter (thinking models, timeouts) | same PR |
| 6 (cycle 34) | `contrib/testing/golden_set`: golden sets for a model's judgements (hard/soft cases, repeats, flaky answers, provider failures) | same PR; CI green at 3ff3bce |

## Next (Phase 1, orders)

1. Evidence that a tester pokes the free scenario unprompted (the owner's games).
2. Done (cycle 73): the 53-case golden set passes twice on the mock, qwen3:8b (Ollama) and DeepSeek-V4.1-Flash (Together).
3. Keep deepening: the commander learning from the readings shown (a scene where the order must be re-worded after a misreading); a bosun who is also on a rota mid-scene; the owner's games.

## Backlog

- Pass it on, arcs 3 and 4 (Ash and Bea relaying; cycle 108). On the mock
  the search found good levels: Past Ash ("Don't rest." gets Cy to keep
  watch), Dex then Ash ("Rest."), All four ("Rest." through Bea, Dex and Ash).
  Live (Flash, evidence/live-flash-c108-arcs34.json) 5 of 11 disagreed, each
  on a gap in the laws: Ash drops "Cy, don't rest" as a clause about resting
  (the page already says "drops anything about resting or watching"; make
  the law and the mock say so too), once passing "Don't rest." as "Don't.";
  Cy ignores "Repair leaks." with no leak open (set Bea's levels in a leak);
  Bea gathered when told "Build.". Redesign on the clarified laws, then
  check live before adding.
  Idea: under "drops anything about resting or watching" Ash makes thin
  puzzles (nothing about watch ever gets past him); an impatient Ash who
  passes on only the first clause would make the order of clauses the
  puzzle. Check live first.

- From the cycle 76 rule audit (fixed in cycle 77: one someone-taker through
  a bosun; the bosun's own name kept in relays; no stale relay when the
  bosun is off duty; storm-cut words not shown as an earlier order; YAML
  'no'; cycle 78: a forbidden repair beats the leak habit in the mock;
  cycle 79: a span's age per heard order; cycle 80: the crew are told
  the scene's own goal; cycle 81: the mock reads a span per clause;
  cycle 82: mock negation scope, negated conditions, a damaged hull;
  cycle 83: the law-text batch: precedence of a conditional that holds,
  the fallback habit in a storm, the relay laws and the mock's relay;
  cycle 84: the commander's someone law and a span's turn; comma-joined
  named clauses and verb forms in the mock). All 13 items done; golden
  set PASS on Flash after the law batch (32/32 hard x2, soft 85.7%).

- Done (cycle 65): model commanders are told the laws and to answer with
  the order's words alone; a live Flash commander then used the mechanics
  ("Ash, watch all night, Cy rest.").

- Orders: the skilled bot plans as if unseen turns are quiet; teach it the
  value of a watch (it lost to random in some sweep settings).
- Done in the library: the Ollama wrapper's `timeout` (cycle 19)
  and `max_tokens` as `num_predict` unless thinking is kept apart (cycle 86,
  with a test).
- Ollama wrapper: an explanation-first JSON template raised qwen3:8b's hard
  laws 6 -> 8 of 10 but lowered soft agreement 82% -> 64% (one run each);
  measure with repeats before proposing.

- Library: `Simulation.play(verbose=...)` (it always prints today).
- Claims: the mock rival only negates and echoes; give it support-building.
- Claims: vary the starting board by seed (the mock's games are identical
  across seeds except for the random bot).
- Claims: who moves first is a lever worth measuring.

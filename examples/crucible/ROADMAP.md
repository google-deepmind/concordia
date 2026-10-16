# Crucible: roadmap

Boxes mirror GUIDE.md's checklists. Each ticked box names its evidence.

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

- From the cycle 76 rule audit (fixed in cycle 77: one someone-taker through
  a bosun; the bosun's own name kept in relays; no stale relay when the
  bosun is off duty; storm-cut words not shown as an earlier order; YAML
  'no'). Still to do:
  - A span's age per heard order (members on an older order are told the
    newest order's turn); repeating an order restarts the span for everyone.
  - The mock lets a cautious member repair a forbidden repair (the leak
    habit is checked before the forbidden list).
  - Mock negation: "not" reaches every later verb in a clause; negated
    conditions ("if there is no leak") and damage without a leak.
  - Mock spans: only a trailing "for N turns" is parsed; "For two turns, ..."
    and numbers above six; a span swallows earlier clauses.
  - Relay law text: the contrary bosun's exception for clauses that already
    forbid is not in the law; the cautious "repairs" test is by vocabulary.
  - The crew's stakes line says "project N of target" in supply_run and
    lookout.
  - The fallback habit text vs the mock's "cannot help" test on fallbacks.
  - No law ranks a conditional clause that holds above a plain one (g5).
  - Smaller: the commander's someone law omits "only those who heard it";
    the commander's prompt lacks a span's turn; "Ash builds, Bea keeps
    watch." is one clause to the mock.

- Model commanders: a live Flash commander never used "someone" and twice
  "ordered" in narration over the word limit (cycle 20); the human seat is
  the target, but a model commander could be prompted to answer with the
  order's words only.

- Orders: the skilled bot plans as if unseen turns are quiet; teach it the
  value of a watch (it lost to random in some sweep settings).
- Library: the Ollama wrapper ignores `max_tokens` and `timeout`; with
  thinking on, a reply has no bound and a hung request waits forever (cycle
  17: one golden request never returned). Bound both (num_predict counts the
  thinking too, so it needs care) and propose it with a test.
- Ollama wrapper: an explanation-first JSON template raised qwen3:8b's hard
  laws 6 -> 8 of 10 but lowered soft agreement 82% -> 64% (one run each);
  measure with repeats before proposing.

- Library: `Simulation.play(verbose=...)` (it always prints today).
- Claims: the mock rival only negates and echoes; give it support-building.
- Claims: vary the starting board by seed (the mock's games are identical
  across seeds except for the random bot).
- Claims: who moves first is a lever worth measuring.

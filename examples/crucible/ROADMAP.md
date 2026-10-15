# Crucible: roadmap

Boxes mirror GUIDE.md's checklists. Each ticked box names its evidence.

## Phase 0: toys

- [x] Shared harness: generic game master, seats, mock, CLI, play audit (cycle 1)
- [x] Toy A, claims, runs end to end on the mock (cycle 1: `crucible_e2e_test`)
- [ ] Toy B, relay
- [ ] Toy C, orders
- [ ] A plain-text phone page with a human seat and telemetry
- [ ] Sandbox Prompt Test of every toy: bots, a live model in the seat, the owner's games
- [ ] One toy kept; the reason in DESIGN.md

## Phase 1: grey box

- [ ] A mock run of the core loop finishes in seconds and reproduces exactly from a seed.
- [ ] Every move shows a measured preview of its effect before it is sent.
- [ ] Every move gets a receipt saying what it did and when it will show.
- [ ] A tester with no win condition starts poking the system unprompted.
- [ ] The skilled bot beats the random bot.

## Phase 2: overloading

- [ ] Every action carries at least two of directive, cost, side effect, information and timing.
- [ ] Each dimension is its own component with its own test; components meet only in the ledger.
- [ ] Simultaneous acts resolve in a fixed order.
- [ ] The median turn offers three or more moves with a measured effect.

## Phase 3: interaction matrix

- [ ] Properties live in YAML and component state, not only in prompts.
- [ ] No resolver branches on a specific operator and target pair.
- [ ] Matrix cells live in YAML and an audit script counts them.
- [ ] Every new feature added or deepened at least two cells.
- [ ] Golden sets pass on every model you ship.

## Phase 4: pacing

- [ ] Each arc has Isolate, Develop, Twist and Catharsis scenes.
- [ ] Bots at many seeds show the skill gaps you target in every scenario.
- [ ] Human-game telemetry keeps the move, the reading the player saw and the hidden truth.

## Phase 5: wrapping

- [ ] Every fact in a role's prompt has a visual home.
- [ ] Each matrix property has one look and one word, and a test enforces it.
- [ ] The rule audit is done and orphan rules are cut.

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

## Next

1. Toy B (relay), then Toy C (orders).
2. The phone page (plain text) with a human seat and telemetry.

## Backlog

- Library: `Simulation.play(verbose=...)` (it always prints today).
- Claims: the mock rival only negates and echoes; give it support-building.
- Claims: vary the starting board by seed (the mock's games are identical
  across seeds except for the random bot).
- Claims: who moves first is a lever worth measuring.

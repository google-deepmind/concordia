---
name: rulebook-games
description: >
  Build turn-based games on Concordia where deterministic rules keep the
  score and language models make the judgements. Use when the user wants a
  game master that follows written rules instead of improvising, a seat a
  person, bot or model can take, a web or chat front end for a human
  player, or a way to check that a model reads the rules the way they were
  written.
---

# Rulebook Games on Concordia

Some games want a referee that never improvises: the rules are code, the
score is a ledger, and a language model is asked only for the judgements
that need language (what does this player's free-text move mean, how does
this character read that order). The contrib pieces below let such a game
run inside Concordia's generic `Simulation` and `Sequential` engine, with
any seat played by a model, a bot or a person.

## The pieces

| Piece | Module | What it does |
|---|---|---|
| `Rulebook` protocol | `contrib/components/game_master/rulebook.py` | Your rules object: `actors()`, `next_actor()`, `action_spec(actor)`, `resolve(actor, action)`, `drain_observations()`, `done()`, `get_state()` / `set_state()` |
| `rulebook__GameMaster` | `contrib/prefabs/game_master/rulebook.py` | A game master that delegates turn order, action specs and resolution to the rulebook (`params={'name': 'Referee', 'rulebook': rules}`); no LLM calls of its own |
| `seated__Entity` | `contrib/prefabs/entity/seated.py` | `minimal.Entity` that reads `act_component_factory` from its params, so one config can seat a model (no factory), a person (`HumanActComponent`) or a bot |
| `PolicyInput`, `ScriptedInput` | `contrib/components/agent/scripted_input.py` | `HumanInput` readers answered by a policy (bots) or a script (tests, replays) |
| `BlockingInput` | `contrib/components/agent/blocking_input.py` | A reader answered from another thread: a web or chat transport reads `pending()` and calls `submit(request_id, answer)`; `close()` releases a waiting simulation |
| `ReasonedChoice` | `contrib/language_models/reasoned_choice.py` | Multiple choice through one sentence of reasoning and a "(letter)", optionally at temperature 0, trusting the option's words over a slipped letter |
| `golden_set` | `contrib/testing/golden_set.py` | Fixed situations with the answers your rules allow, run several times against a model: hard cases must all pass, soft cases need a share, changed answers are flagged as flaky |

## How a game is put together

1. Write the rules as plain Python implementing `Rulebook`. Keep every
   number in data and every consequence in code. When a judgement needs
   language, ask a model inside `resolve` (for example with an
   `InteractiveDocument`) and keep the answer in the ledger.
2. Build a `Simulation` whose game master is `rulebook__GameMaster` with
   your rulebook, and whose players are `seated__Entity` instances.
3. Seat players: no factory for a model; a factory returning
   `HumanActComponent(reader, component_order=order)` for a person or a bot,
   with `PolicyInput` (a bot), `ScriptedInput` (a test) or `BlockingInput`
   (a page) as the reader.
4. Write a mock language model that answers your prompts by the same rules
   (a lookup table is enough). It is the CI: bots, balance sweeps and tests
   run without a provider and stay byte-identical across refactors.
5. Write golden cases for every judgement a model makes, and run them with
   `golden_set.run(cases, answer, repeats=2)` against each model you mean to
   use.

The crucible example (`examples/crucible/`) is a complete game built this
way, with a phone page on `BlockingInput`, bots on `PolicyInput`, and a
golden set of 49 cases.

## Pitfalls seen in practice

- **Let models judge, not count.** Asked to work out from fatigue numbers
  which crew member should volunteer, a strong model got several of those
  cases wrong in every run, and worse when the rule was spelled out as
  steps; told the result by the ledger and asked only whether the clause
  applied to it, it got them right.
- **Golden sets find gaps in your own rule text.** A mock reads the rules
  the way you meant them; a model reads them the way you wrote them. Two
  missing sentences in the crucible's laws of reading were found only by
  a small model failing golden cases.
- **Run judgements cold, and still run them twice.** Temperature 0 made
  readings repeatable enough to test, but providers still varied between
  runs; hard rules held while soft judgements moved by several points.
- **Thinking models and stop sequences.** A thinking model's reply opens with
  its thinking, and Ollama applies stop sequences to it, so a newline
  terminator returns an empty answer. Use the Ollama adapter's
  `think=True` (terminators applied to the answer) and a `timeout`.
- **Isolate the law in hard cases.** If a hard case fails because of a
  separate judgement (a temperament's taste, say), move that variant to the
  soft cases and test the law with a member who has no such judgement.

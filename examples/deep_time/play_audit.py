# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Play audit: does skill matter, and is every turn a decision?

Plays each scenario's main role (the first a person may pick: the Steward, or
Mission Control) with three bots in mock mode, over many seeds:

* `default`, the skilled bot: the mock's game-aware policies;
* `random`: the main role picks any of its moves;
* `passive`: the main role never acts.

Only the main role changes: every other role (species, herds, peoples,
colonists) keeps the skilled policies.

It reports win rates and the skill gaps between the bots, how often the
skilled bot acts rather than holds, and how many of the moves offered each
turn carry a measured push (`GameContext.pushes`): those it can make now,
and with the moves it is saving for, which may be planned for a later turn.

  python -m examples.deep_time.play_audit --scenarios=daisyworld,herds \
      --seeds=16
"""

# pylint: disable=g-complex-comprehension,g-long-ternary

from collections.abc import Sequence
import contextlib
import functools
import io
import json
import statistics
from typing import Any

from absl import app
from absl import flags
from examples.deep_time import model_setup
from examples.deep_time import scenarios as scenarios_lib
from examples.deep_time import simulation
from examples.deep_time.web import session as session_lib

BOTS = ('default', 'random', 'passive')


@functools.cache
def main_role(scenario: str) -> str:
  """The role a person plays first in a scenario (its entity name)."""
  return session_lib.roles(scenario)[0][0]


def variant(bot: str, role: str) -> str:
  """The mock policy set in which only `role` is played by `bot`.

  Every other role keeps the skilled policies, as when a person plays the
  main role against the computer: a random Steward still faces skilled
  peoples.

  Args:
    bot: One of `BOTS`, or another `--mock_variant` (used as it is).
    role: The main role's entity name.

  Returns:
    A key of `mock_language_model.VARIANTS`.
  """
  if bot not in ('random', 'passive'):
    return bot
  return f'{bot}_{"mission" if role == "Mission Control" else "steward"}'


def play_one(
    scenario: str, bot: str, seed: int, ticks: int | None = None
) -> dict[str, Any]:
  """One mock game; what the main role was offered and did, and the outcome.

  Args:
    scenario: Scenario key.
    bot: Mock policy set (one of `BOTS`, or another `--mock_variant`).
    seed: Seed of the planet.
    ticks: Overrides the scenario's length.

  Returns:
    The outcome, the turns the main role played, how many of them it acted
    on (a move other than its neutral one), and the moves with a measured
    push it was offered each turn.
  """
  role = main_role(scenario)
  sim, context = simulation.build_simulation(
      scenario,
      model_setup.make_models('mock', mock_variant=variant(bot, role)),
      max_ticks=ticks,
      seed=seed,
  )
  actor = context.actors[role]
  pushes: list[int] = []
  plannable: list[int] = []
  spec_of = actor.action_spec

  def recording():
    spec = spec_of()
    # A Steward that may speak answers in words; its moves are still offered.
    offered = list(spec.options) or list(getattr(actor, 'options', list)())
    pushes.append(len(context.pushes(offered, role)))
    # With the moves being saved for, which may be planned.
    later = list(getattr(actor, 'out_of_reach', lambda: [])())
    plannable.append(len(context.pushes([*offered, *later], role)))
    return spec

  actor.action_spec = recording  # This game only: what was on the menu.
  with contextlib.redirect_stdout(io.StringIO()):  # The engine's log.
    sim.play(premise='', max_steps=ticks or context.planet.max_ticks)
  # A model's free answer starts with its name.
  moves = [
      action.strip().lower().removeprefix(f'{role.lower()} ')
      for _, action, _ in actor.log
  ]
  neutral = {o.lower() for o in session_lib.NEUTRAL_OPTIONS}
  return {
      'scenario': scenario,
      'bot': bot,
      'seed': seed,
      'role': role,
      'outcome': context.planet.outcome or 'unfinished',
      'stars': context.planet.star_count(),
      # Side objectives met, of those the scenario sets.
      'side': [
          sum(1 for v in context.planet.side.values() if v == 'met'),
          len(context.planet.side),
      ],
      'reason': context.planet.outcome_reason,
      'ticks': context.planet.tick,
      'turns': len(moves),
      'acted': sum(
          1 for m in moves if not any(m.startswith(n) for n in neutral)
      ),
      'pushes': pushes,
      'plannable': plannable,
  }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Win rates per bot, skill gaps, the skilled bot's acting share and pushes."""
  wins = {}
  for bot in dict.fromkeys(r['bot'] for r in rows):
    runs = [r for r in rows if r['bot'] == bot]
    wins[bot] = round(
        sum(r['outcome'] == 'victory' for r in runs) / len(runs), 3
    )
  skilled = [r for r in rows if r['bot'] == 'default']
  turns = sum(r['turns'] for r in skilled)
  pushes = [p for r in skilled for p in r['pushes']]
  plannable = [p for r in skilled for p in r.get('plannable', ())]
  return {
      'scenario': rows[0]['scenario'],
      'role': rows[0]['role'],
      'seeds': len(skilled),
      'wins': wins,
      'gap_random': round(wins.get('default', 0) - wins.get('random', 0), 3),
      'gap_passive': round(wins.get('default', 0) - wins.get('passive', 0), 3),
      'acted_share': round(sum(r['acted'] for r in skilled) / max(1, turns), 3),
      'pushes_median': statistics.median(pushes) if pushes else 0,
      # Counting the moves saved for, plannable for a later turn.
      'plannable_median': statistics.median(plannable) if plannable else 0,
      # Each bot's share of side objectives met, if any are set.
      'side': (
          {
              bot: round(
                  sum(r['side'][0] for r in rows if r['bot'] == bot)
                  / max(1, sum(r['side'][1] for r in rows if r['bot'] == bot)),
                  3,
              )
              for bot in wins
          }
          if any(r.get('side', [0, 0])[1] for r in rows)
          else None
      ),
      # The skilled bot's stars: how many games earned 0, 1, 2 and 3.
      'stars': [sum(1 for r in skilled if r['stars'] == n) for n in range(4)],
  }


def audit(
    scenario: str,
    seeds: int,
    bots: Sequence[str] = BOTS,
    ticks: int | None = None,
) -> dict[str, Any]:
  """Plays a scenario's main role with each bot over seeds 0..seeds-1."""
  rows = [
      play_one(scenario, bot, seed, ticks)
      for bot in bots
      for seed in range(seeds)
  ]
  return summarize(rows)


def main(argv: Sequence[str]) -> None:
  del argv
  values = flags.FLAGS
  for scenario in values.scenarios or list(scenarios_lib.SCENARIOS):
    print(json.dumps(audit(scenario, values.seeds, values.bots, values.ticks)))


if __name__ == '__main__':
  flags.DEFINE_list('scenarios', [], 'Scenario keys (default: all).')
  flags.DEFINE_integer('seeds', 16, 'Seeds per bot.')
  flags.DEFINE_list('bots', list(BOTS), 'Mock policy sets to compare.')
  flags.DEFINE_integer('ticks', None, "Overrides the scenarios' length.")
  app.run(main)

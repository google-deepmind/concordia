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

"""Measures play: bots at many seeds, and the moves each turn offers."""

import statistics
from typing import Any

from examples.crucible import harness
from examples.crucible import model_setup
from examples.crucible.toys import base


class _MeasuringReader:
  """Wraps a bot reader; counts the moves on offer that change the world.

  A move counts when its measured result differs from passing.
  """

  def __init__(self, game: harness.Game, candidates_fn: Any):
    self._game = game
    self._inner = game.reader
    self._candidates = candidates_fn
    self.offered: list[int] = []

  def __call__(self, request: Any) -> str:
    if self._candidates is not None:
      toy = self._game.toy
      actor = request.entity_name
      _, held = toy.simulate(actor, base.PASS)
      moving = 0
      for move in self._candidates(toy, actor):
        if move == base.PASS:
          continue
        _, after = toy.simulate(actor, move)
        if repr(after) != repr(held):
          moving += 1
      self.offered.append(moving)
    return self._inner(request)  # pyrefly: ignore[not-callable]


def audit(toy: str, scenario: str, seeds: int) -> dict[str, Any]:
  """Win rates per bot, the skill gaps, and moves with an effect per turn."""
  wins: dict[str, float] = {}
  offered: list[int] = []
  for bot in harness.BOTS:
    won = 0
    reader = None
    for seed in range(seeds):
      game = harness.build_game(
          toy, model_setup.mock_model(), bot, seed, scenario
      )
      if bot == 'skilled':
        module = __import__(
            f'examples.crucible.toys.{toy}.bots', fromlist=['candidates']
        )
        reader = _MeasuringReader(game, getattr(module, 'candidates', None))
        # Swap the reader the seat's act component calls.
        seat = next(e for e in game.sim.entities if e.name == game.kind.seat)
        seat._act_component._input_reader = reader  # pylint: disable=protected-access  # pyrefly: ignore[missing-attribute]
      game.play()
      if game.outcome()['winner'] == game.kind.seat:
        won += 1
      if reader is not None:
        offered.extend(reader.offered)
    wins[bot] = round(100.0 * won / seeds)
  return {
      'toy': toy,
      'scenario': scenario,
      'seeds': seeds,
      'win_pct': wins,
      'skilled_minus_random': wins['skilled'] - wins['random'],
      'skilled_minus_passive': wins['skilled'] - wins['passive'],
      'moves_with_effect_median': statistics.median(offered) if offered else 0,
  }

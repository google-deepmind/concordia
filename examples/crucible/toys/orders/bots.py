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

"""Bots for the orders toy: skilled, random and passive.

All three choose among the same orders (or hold). The skilled bot plays each
candidate forward to the last turn on a copy of the ship, holding it, and
issues the one that ends best; it only spends a revision when that beats
holding.
"""

import random

from examples.crucible.toys.orders import toy as toy_lib

ORDERS = (
    'Build.',
    'Build the project.',
    'Gather supplies and build.',
    'Build; if a leak opens, repair it.',
    'If there is a leak, fix it; otherwise build.',
    'Build; repair leaks; rest when tired.',
    'Bea, keep watch; everyone else build.',
    'Ash, keep watch; everyone else build.',
    'Cy keeps watch; Ash and Bea build.',
    'Keep watch.',
    'Bea repairs leaks; Ash and Cy build.',
    'Cy, gather supplies; Ash builds; Bea fixes leaks.',
    'Gather supplies; during storms, build.',
    'Everyone rest.',
    'Rest.',
    "Don't build.",
    'Dex, rest; everyone else build.',
    "Dex, don't repair; Ash and Cy build.",
    "Dex, don't repair; everyone else build.",
    "Ash builds; Dex, don't build.",
    'Someone keep watch; everyone else build.',
    'Someone repair leaks; everyone else build.',
    'If a leak opens, someone repair it; everyone else build.',
)


def candidates(toy: toy_lib.OrdersToy, actor: str) -> list[str]:
  """Hold, and every listed order the commander can afford now."""
  del actor
  points = toy.world.points
  return [toy_lib.HOLD] + [o for o in ORDERS if toy.command.cost(o) <= points]


def value(toy: toy_lib.OrdersToy, ship: toy_lib.Ship) -> float:
  goal = float(getattr(ship, toy.goal()))
  return (100.0 if toy.won(ship) else 0.0) + goal - 2.0 * ship.damage


def expected_value(toy: toy_lib.OrdersToy, order: str) -> float:
  """The order's value averaged over the scene's possible event schedules.

  A skilled commander knows what the premise says can happen (the scene's
  events, shifted by up to two turns) but not which turns; it plans against
  every schedule the scene allows, not only against what it has seen.

  Args:
    toy: The toy.
    order: The order to give now.

  Returns:
    The mean value of the ship at the end over the schedules.
  """
  shifts = range(3)
  return sum(value(toy, toy.play_out(order, shift)) for shift in shifts) / 3


def skilled(toy: toy_lib.OrdersToy, actor: str, rng: random.Random) -> str:
  del rng
  best, best_value = toy_lib.HOLD, expected_value(toy, toy_lib.HOLD)
  for order in candidates(toy, actor):
    if order == toy_lib.HOLD:
      continue
    v = expected_value(toy, order) - 0.5  # A revision has a price.
    if v > best_value:
      best, best_value = order, v
  return best


def random_move(toy: toy_lib.OrdersToy, actor: str, rng: random.Random) -> str:
  return rng.choice(candidates(toy, actor))


def passive(toy: toy_lib.OrdersToy, actor: str, rng: random.Random) -> str:
  del toy, actor, rng
  return toy_lib.HOLD


HABIT = 'Build; if a leak opens, repair it.'


def habit(toy: toy_lib.OrdersToy, actor: str, rng: random.Random) -> str:
  """Repeats the lesson of the arc's earlier scenes: the habit a twist breaks."""
  del actor, rng
  order = toy._data.get('habit', HABIT)  # pylint: disable=protected-access
  if toy.world.order != order and toy.command.cost(order) <= toy.world.points:
    return order
  return toy_lib.HOLD


POLICIES = {
    'skilled': skilled,
    'random': random_move,
    'passive': passive,
    'habit': habit,
}

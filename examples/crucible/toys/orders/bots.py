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
    'Bea repairs leaks; Ash and Cy build.',
    'Cy, gather supplies; Ash builds; Bea fixes leaks.',
    'Gather supplies; during storms, build.',
    'Everyone rest.',
)


def candidates(toy: toy_lib.OrdersToy, actor: str) -> list[str]:
  """Hold, and every listed order the commander can afford now."""
  del actor
  points = toy.world.points
  return [toy_lib.HOLD] + [o for o in ORDERS if toy.command.cost(o) <= points]


def rollout(toy: toy_lib.OrdersToy, order: str) -> toy_lib.Ship:
  """The ship at the end of the game if `order` is given now and held."""
  return toy.play_out(order)


def value(toy: toy_lib.OrdersToy, ship: toy_lib.Ship) -> float:
  goal = float(getattr(ship, toy.goal()))
  return (100.0 if toy.won(ship) else 0.0) + goal - 2.0 * ship.damage


def skilled(toy: toy_lib.OrdersToy, actor: str, rng: random.Random) -> str:
  del rng
  best, best_value = toy_lib.HOLD, value(toy, rollout(toy, toy_lib.HOLD))
  for order in candidates(toy, actor):
    if order == toy_lib.HOLD:
      continue
    v = value(toy, rollout(toy, order)) - 0.5  # A revision has a price.
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
  """Repeats the lesson of the earlier scenes: the habit a twist breaks."""
  del actor, rng
  if toy.world.order != HABIT and toy.command.cost(HABIT) <= toy.world.points:
    return HABIT
  return toy_lib.HOLD


POLICIES = {
    'skilled': skilled,
    'random': random_move,
    'passive': passive,
    'habit': habit,
}

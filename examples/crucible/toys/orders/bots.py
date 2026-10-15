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

import copy
import random

from examples.crucible.toys.orders import toy as toy_lib

ORDERS = (
    'Build the project.',
    'Gather supplies and build.',
    'Build; if a leak opens, repair it.',
    'If there is a leak, fix it; otherwise build.',
    'Build; repair leaks; rest when tired.',
    'Repair the hull first, then build.',
    'Gather supplies; during storms, build.',
    'Everyone rest.',
)


def candidates(toy: toy_lib.OrdersToy, actor: str) -> list[str]:
  del actor
  if toy.world.revisions_left <= 0:
    return [toy_lib.HOLD]
  return [toy_lib.HOLD, *ORDERS]


def rollout(toy: toy_lib.OrdersToy, order: str) -> toy_lib.Ship:
  """The ship at the end of the game if `order` is given now and held."""
  ship = copy.deepcopy(toy.world)
  toy.apply(ship, toy.commander, order)
  while ship.turn < int(toy._data['turns']) and not toy.wrecked(ship):  # pylint: disable=protected-access
    toy.apply(ship, toy.commander, toy_lib.HOLD)
  return ship


def value(toy: toy_lib.OrdersToy, ship: toy_lib.Ship) -> float:
  data = toy._data  # pylint: disable=protected-access
  won = not toy.wrecked(ship) and ship.progress >= int(data['target'])
  return (100.0 if won else 0.0) + ship.progress - 2.0 * ship.damage


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


POLICIES = {'skilled': skilled, 'random': random_move, 'passive': passive}

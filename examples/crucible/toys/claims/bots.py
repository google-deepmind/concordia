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

"""Bots for the claims toy: skilled, random and passive.

All three choose from the same candidate moves, so the gap between skilled
and random play measures how much choosing matters. The skilled bot measures
every candidate on a copy of the board (the same preview a person sees) and
keeps the best.
"""

import random

from examples.crucible.toys import base
from examples.crucible.toys.claims import toy as toy_lib
from examples.crucible.toys.claims import world as world_lib


def negate(text: str) -> str:
  body = text.strip().rstrip('.')
  return f'It is false that {body[:1].lower()}{body[1:]}.'


def affirm(text: str) -> str:
  body = text.strip().rstrip('.')
  return f'Indeed, {body[:1].lower()}{body[1:]}.'


def candidates(toy: toy_lib.ClaimsToy, actor: str) -> list[str]:
  """Moves worth considering: attack what opposes you, echo what helps."""
  board = toy.world
  opponent = toy.opponent(actor)
  mine = board.thesis_of(actor)
  out = [base.PASS]
  for c in board.standing():
    against_me = c.author == opponent or (
        mine is not None
        and c.author == ''
        and board.relation(c.key, mine.key) == world_lib.CONFLICT
    )
    for_me = c.author == actor or (
        mine is not None
        and c.author == ''
        and board.relation(c.key, mine.key) == world_lib.AGREE
    )
    if against_me:
      out.append(negate(c.text))
    if for_me:
      out.append(affirm(c.text))
  limit = min(int(toy.rules['max_words']), board.ink[actor])
  seen = set()
  fitting = []
  for move in out:
    if move not in seen and (move == base.PASS or len(move.split()) <= limit):
      seen.add(move)
      fitting.append(move)
  return fitting


def value(toy: toy_lib.ClaimsToy, board: world_lib.Board, actor: str) -> float:
  """How good `board` is for `actor`: theses first, then weight."""
  opponent = toy.opponent(actor)
  mine = board.thesis_of(actor)
  theirs = board.thesis_of(opponent)
  v = board.side_weight(actor) - board.side_weight(opponent)
  if mine is not None:
    v += 3.0 if mine.standing else -3.0
  if theirs is not None:
    v += -3.0 if theirs.standing else 3.0
    v -= theirs.weight
  return v


def skilled(toy: toy_lib.ClaimsToy, actor: str, rng: random.Random) -> str:
  del rng
  best, best_value = base.PASS, value(toy, toy.world, actor) - 0.01
  for move in candidates(toy, actor):
    if move == base.PASS:
      continue
    _, after = toy.simulate(actor, move)
    # Ink is scarce: a word spent should buy something.
    v = value(toy, after, actor) - 0.02 * len(move.split())
    if v > best_value:
      best, best_value = move, v
  return best


def random_move(toy: toy_lib.ClaimsToy, actor: str, rng: random.Random) -> str:
  return rng.choice(candidates(toy, actor))


def passive(toy: toy_lib.ClaimsToy, actor: str, rng: random.Random) -> str:
  del toy, actor, rng
  return base.PASS


POLICIES = {'skilled': skilled, 'random': random_move, 'passive': passive}

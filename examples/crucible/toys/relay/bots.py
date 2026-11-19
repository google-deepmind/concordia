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

"""Bots for the relay toy: skilled, random and passive.

All three choose among the same deliveries: as written, warmed, cooled, or
with the offer nudged toward the client. The skilled bot measures each on a
copy of the talks (the same preview a person sees) and keeps the best.
"""

import random
import re

from examples.crucible.toys import base
from examples.crucible.toys.relay import toy as toy_lib

_COLD_SWAPS = {
    'insist': 'hope',
    'final': 'my thought',
    'demand': 'ask',
    'must': 'would like to',
}


def warmed(text: str) -> str:
  out = text
  for cold, mild in _COLD_SWAPS.items():
    out = re.sub(rf'\b{cold}\b', mild, out, flags=re.IGNORECASE)
  return f'Please, {out.rstrip(".")}. Thank you.'


def cooled(text: str) -> str:
  return f'I insist: {text.rstrip(".")}. That is final.'


def nudged(toy: toy_lib.RelayToy, text: str) -> str | None:
  """The offer moved two units toward the relay's client (risky)."""
  share = toy_lib.offered_share(text, toy.pot)
  if share is None:
    return None
  client = toy._data['client']  # pylint: disable=protected-access
  receiver = toy.other(toy.world.speaker)
  new = share + 2 if receiver == client else share - 2
  new = max(0, min(toy.pot, new))
  return re.sub(rf'\b{share}\b', str(new), text, count=1)


def candidates(toy: toy_lib.RelayToy, actor: str) -> list[str]:
  del actor
  original = toy.world.original
  moves = [base.PASS, warmed(original), cooled(original)]
  nudge = nudged(toy, original)
  if nudge and nudge != original:
    moves.append(nudge)
  return moves


def value(toy: toy_lib.RelayToy, talks: toy_lib.Talks) -> float:
  """Payout when over; otherwise how promising the talks are for the client.

  A held deal without the bonus is worth less than talks that can still
  reach it, until the messages run short.

  Args:
    toy: The toy.
    talks: The talks after a candidate delivery.

  Returns:
    The value of `talks` for the relay.
  """
  if talks.phase == 'over':
    return 10.0 * toy.payout(talks)
  client = toy._data['client']  # pylint: disable=protected-access
  other = toy.other(client)
  left = int(toy._data['messages']) - talks.messages  # pylint: disable=protected-access
  promise = talks.warmth[other] - 0.5 * talks.warmth[client]
  base_value = 15.0 if left > 2 else 5.0
  return base_value + 5.0 * promise + 0.01 * talks.edits_left


def skilled(toy: toy_lib.RelayToy, actor: str, rng: random.Random) -> str:
  del rng
  best, best_value = base.PASS, None
  for move in candidates(toy, actor):
    _, after = toy.simulate(actor, move)
    v = value(toy, after)
    if best_value is None or v > best_value:
      best, best_value = move, v
  return best


def random_move(toy: toy_lib.RelayToy, actor: str, rng: random.Random) -> str:
  return rng.choice(candidates(toy, actor))


def passive(toy: toy_lib.RelayToy, actor: str, rng: random.Random) -> str:
  del toy, actor, rng
  return base.PASS


POLICIES = {'skilled': skilled, 'random': random_move, 'passive': passive}

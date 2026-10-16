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

"""Bots for Pass it on: a search, a guess and silence.

Candidates are messages built from the level's actions (the data's `say`)
and the last in line's names: an optional plain clause and clauses for any
of the names, each part asked for or forbidden.
The skilled bot sends the fewest-word candidate that solves the level (it
tries them on a copy of the world); the random bot sends any candidate; the
passive bot says nothing.
"""

import itertools
import random
from typing import Any

from examples.crucible.toys.passiton import toy as toy_lib


def _clause(
    say: dict[str, str], action: str, no: bool, name: str | None
) -> str:
  words = ("don't " if no else '') + say[action]
  return f'{name}, {words}' if name else words


def candidates(toy: toy_lib.PassItOnToy, actor: str = '') -> list[str]:
  """Messages a player might try, fewest words first."""
  del actor  # one seat
  say = toy.view()['say']
  options = [(a, no) for a in say for no in (False, True)]
  names = toy.receivers
  out = set()
  plains = [None] + options
  for k in range(len(names) + 1):
    for named in itertools.combinations(names, k):
      for combo in itertools.product(options, repeat=k):
        clauses = [_clause(say, a, no, n) for n, (a, no) in zip(named, combo)]
        for plain in plains:
          if plain is None and not clauses:
            continue
          parts = ([_clause(say, *plain, None)] if plain else []) + clauses
          out.add('; '.join(parts))
  messages = [m[0].upper() + m[1:] + '.' for m in out]
  return sorted(messages, key=lambda m: (len(m.split()), m))


def skilled(toy: Any, prompt: str, rng: random.Random) -> str:
  del prompt, rng
  for message in candidates(toy):
    receipt, _ = toy.simulate(toy_lib.SEAT, message)
    if receipt.delta.get('solved'):
      return message
  return 'hold'


def random_policy(toy: Any, prompt: str, rng: random.Random) -> str:
  del prompt
  return rng.choice(candidates(toy))


def passive(toy: Any, prompt: str, rng: random.Random) -> str:
  del toy, prompt, rng
  return 'hold'


POLICIES = {'skilled': skilled, 'random': random_policy, 'passive': passive}

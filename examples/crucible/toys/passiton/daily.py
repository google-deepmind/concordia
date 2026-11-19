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

"""Today's puzzle for Pass it on: a level made from the date (cycle 112).

A seed picks a line (no relayer, Cy, Dex, or both) and goals for the people
at the top; the level is kept only if, on the mock, the skilled bot solves
it in at most `MAX_BEST` words, the obvious message (each goal said plainly
to its person) fails, and the random bot solves it less than half the time.
Only what held on a live model is used: Cy and Dex relay; Ash, Cy and Dex
act (Bea's readings of plain clauses varied live, cycle 108).
"""

import copy
import functools
import random
from typing import Any

from examples.crucible import model_setup
from examples.crucible import toys

MAX_BEST = 8
_RELAYS = ([], ['Cy'], ['Dex'], ['Cy', 'Dex'], ['Dex', 'Cy'])
_ACTORS = ('Ash', 'Cy', 'Dex')
_GOALS = ('build', 'watch', 'rest')
_PASSES = {'Cy': 'word for word', 'Dex': 'saying the opposite of each part'}


def _task(
    relays: list[str],
    receivers: list[str],
    goal: dict[str, str],
    say: dict[str, str],
) -> str:
  line = ''.join(f'{who} ({_PASSES[who]}), then ' for who in relays)
  wants = ' and '.join(f'{who} to {say[goal[who]]}' for who in receivers)
  return (
      f"Today's line: your words go through {line}to "
      f'{" and ".join(receivers)} at the top. Get {wants}.'
  )


@functools.lru_cache(maxsize=32)
def level(seed: int) -> dict[str, Any]:
  """The day's level for `seed` (the same seed, the same level)."""
  # pylint: disable=g-import-not-at-top
  from examples.crucible.toys.passiton import bots
  from examples.crucible.toys.passiton import toy as toy_lib
  # pylint: enable=g-import-not-at-top
  data = copy.deepcopy(toys.load_data('passiton'))
  mock = model_setup.mock_model()
  rng = random.Random(seed)
  for _ in range(300):
    relays = list(rng.choice(_RELAYS))
    pool = [who for who in _ACTORS if who not in relays]
    receivers = rng.sample(pool, rng.choice([1, 2]) if len(pool) > 1 else 1)
    goal = {who: rng.choice(_GOALS) for who in receivers}
    candidate = {
        'beat': 'daily',
        'title': "Today's puzzle",
        'line': relays + ([receivers] if len(receivers) > 1 else receivers),
        'goal': goal,
        'event': 'quiet',
        'task': _task(relays, receivers, goal, data['say']),
        'stars': {},
    }
    data['scenarios']['daily'] = candidate
    toy = toy_lib.PassItOnToy(mock, data, 'daily')
    best = bots.skilled(toy, '', rng)
    if best == 'hold' or toy_lib.words_in(best) > MAX_BEST:
      continue
    plain = '; '.join(f'{who}, {data["say"][goal[who]]}' for who in receivers)
    if toy.preview(toy_lib.SEAT, plain + '.').delta.get('solved'):
      continue
    tries = [
        bots.random_policy(toy, '', random.Random(seed * 31 + i))
        for i in range(16)
    ]
    wins = sum(
        bool(toy.preview(toy_lib.SEAT, m).delta.get('solved')) for m in tries
    )
    if wins >= 8:
      continue
    words = toy_lib.words_in(best)
    candidate['stars'] = {2: {'max_words': words + 2}, 3: {'max_words': words}}
    candidate['best'] = best
    return candidate
  raise ValueError(f'No daily level found for seed {seed}.')

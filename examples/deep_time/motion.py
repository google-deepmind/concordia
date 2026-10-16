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

"""How a move shows on the map the turn it acts.

A Steward's move by its effect's kind (`data/motion.yaml`), refined by its
key; every other role's moves by the role: a people's labor at its towns, a
species' or the colonists' as a ring where they are, the herds' over the
herds. Holding course shows nothing: the map rests.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib


def look(
    planet: planet_lib.Planet,
    role: str,
    move: str,
    neutral: Sequence[str] = (),
) -> dict[str, Any] | None:
  """The motion, colour (and herd kind) a move shows with, or None.

  Args:
    planet: The planet (its interventions, for the Steward's moves).
    role: 'steward', 'people', 'species', 'herds', 'colony' or 'mission'.
    move: The move as chosen (an option's text).
    neutral: Options that hold course (no motion).

  Returns:
    {'motion', 'colour', 'key'?, 'herd'?}, or None for a move that shows
    nothing.
  """
  text = move.strip().lower()
  if any(text.startswith(n.lower()) for n in neutral):
    return None
  looks = data.load('motion.yaml')
  if role != 'steward':
    found = looks['roles'].get(role)
    return dict(found) if found else None
  spec = planet.intervention_for_option(move)
  if spec is None:
    return None
  kind = _kind(spec.effect)
  base: Mapping[str, Any] = looks['kinds'].get(kind) or {}
  if not base:
    return None
  return {**base, **(looks['keys'].get(spec.key) or {}), 'key': spec.key}


def _kind(effect: Any) -> str:
  return str(effect.get('kind')) if isinstance(effect, Mapping) else str(effect)

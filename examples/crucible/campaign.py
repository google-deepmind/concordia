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

"""The campaign: chapters that open with stars, and a daily challenge.

Progress is a mapping from scene to the best stars earned there. A chapter
is open when it is the first, or the chapter before it has at least one
star. The daily challenge is one scene with a seed taken from the date, so
everyone who plays it on a day plays the same world.
"""

from collections.abc import Mapping
import datetime
import json
import os
import threading
from typing import Any


def chapters(
    data: Mapping[str, Any], progress: Mapping[str, int]
) -> list[dict[str, Any]]:
  """The campaign's chapters, with their stars and whether they are open."""
  out = []
  previous_stars = 1
  for number, scenario in enumerate(data.get('campaign', []), start=1):
    scene = data['scenarios'][scenario]
    stars = int(progress.get(scenario, 0))
    out.append({
        'number': number,
        'scenario': scenario,
        'title': scene.get('title', scenario),
        'beat': scene.get('beat', ''),
        'stars': stars,
        'open': previous_stars > 0,
    })
    previous_stars = stars
  return out


def is_open(
    data: Mapping[str, Any], progress: Mapping[str, int], scenario: str
) -> bool:
  """Whether `scenario` may be played: an open chapter, or not a chapter."""
  for chapter in chapters(data, progress):
    if chapter['scenario'] == scenario:
      return chapter['open']
  return True


def daily(data: Mapping[str, Any], day: datetime.date) -> dict[str, Any]:
  """The day's challenge: a scene and a seed from the date.

  The scene rotates through `daily.scenarios` day by day (or is always
  `daily.scenario`); the seed shifts its events.

  Args:
    data: The toy's data.
    day: The date.

  Returns:
    The scenario, seed and date.
  """
  spec = data.get('daily', {})
  seeds = int(spec.get('seeds', 997))
  rotation = list(spec.get('scenarios', ()))
  scenario = (
      rotation[day.toordinal() % len(rotation)]
      if rotation
      else spec.get('scenario', data.get('default_scenario'))
  )
  return {
      'scenario': scenario,
      'opens_with': spec.get('opens_with', scenario),
      'title': (
          data.get('scenarios', {})
          .get(scenario, {})
          .get('title', spec.get('title', "Today's puzzle"))
      ),
      'seed': day.toordinal() % seeds,
      'date': day.isoformat(),
  }


class Progress:
  """Best stars per scene, kept in a JSON file; thread-safe."""

  def __init__(self, path: str | None = None):
    self._path = path
    self._lock = threading.Lock()
    self._stars: dict[str, int] = {}
    if path and os.path.exists(path):
      with open(path) as f:
        self._stars = {k: int(v) for k, v in json.load(f).items()}

  def snapshot(self) -> dict[str, int]:
    with self._lock:
      return dict(self._stars)

  def record(self, scenario: str, stars: int) -> bool:
    """Keeps the best stars for a scene; True if this was a new best."""
    with self._lock:
      if stars <= self._stars.get(scenario, 0):
        return False
      self._stars[scenario] = stars
      if self._path:
        with open(self._path, 'w') as f:
          json.dump(self._stars, f, indent=1)
      return True

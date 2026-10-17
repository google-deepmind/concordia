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

"""The campaign through deep time and the daily challenge.

The campaign is the scenarios in the order of their worlds' ages; a chapter
opens when the one before has a star. The daily challenge is one chapter's
scenario on a world seeded by the date, the same for every player that day.
`Progress` keeps the best stars of the games people played (a JSON file
beside the server's telemetry).
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-long-ternary,g-short-docstring-punctuation

import datetime
import hashlib
import json
import os
import threading
from typing import Any

from examples.deep_time.world import data


def chapters() -> list[dict[str, str]]:
  """The campaign's chapters, oldest world first."""
  return [dict(c) for c in data.load('campaign.yaml')['chapters']]


def offered() -> bool:
  """Whether the start screen offers the daily challenge."""
  return bool(data.load('campaign.yaml')['daily'].get('offered', True))


def daily(day: datetime.date) -> dict[str, Any]:
  """The day's challenge: a chapter's scenario and a world seed."""
  digest = hashlib.sha256(f'deep time {day.isoformat()}'.encode()).digest()
  pick = int.from_bytes(digest[:8], 'big')
  every = chapters()
  seeds = int(data.load('campaign.yaml')['daily']['seeds'])
  chapter = every[pick % len(every)]
  return {
      'date': day.isoformat(),
      'scenario': chapter['scenario'],
      'title': chapter['title'],
      'seed': 1 + (pick // len(every)) % seeds,
  }


class Progress:
  """The best stars people have earned, by scenario and by daily challenge."""

  def __init__(self, path: str | None):
    self._path = path
    self._lock = threading.Lock()

  def _load(self) -> dict[str, Any]:
    if not self._path or not os.path.exists(self._path):
      return {'scenarios': {}, 'daily': {}}
    try:
      with open(self._path, encoding='utf-8') as f:
        found = json.load(f)
    except (OSError, ValueError):
      return {'scenarios': {}, 'daily': {}}
    found.setdefault('scenarios', {})
    found.setdefault('daily', {})
    return found

  def record(
      self, scenario: str, seed: int, stars: int, day: datetime.date
  ) -> int:
    """Keeps a played game's stars if they beat the best so far.

    Returns:
      The scenario's best stars before this game.
    """
    if not self._path:
      return 0
    with self._lock:
      best = self._load()
      before = int(best['scenarios'].get(scenario, 0))
      if stars > before:
        best['scenarios'][scenario] = stars
      challenge = daily(day)
      if (scenario, seed) == (challenge['scenario'], challenge['seed']):
        key = challenge['date']
        best['daily'][key] = max(stars, int(best['daily'].get(key, 0)))
      os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
      with open(self._path, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=1)
    return before

  def chapter(
      self,
      scenario: str,
      seed: int,
      stars: int,
      before: int,
      day: datetime.date,
  ) -> dict[str, Any] | None:
    """A decided game's place in the campaign, for its recap.

    Args:
      scenario: The game's scenario.
      seed: Its world seed.
      stars: The stars it earned.
      before: The scenario's best stars before it (from `record`).
      day: Today, for the daily challenge.

    Returns:
      Its chapter's number and title, the stars it earned and the best now,
      whether it beat the best, and the next chapter: whether it is open
      and whether this game opened it. None for a scenario no chapter has.
    """
    every = chapters()
    keys = [c['scenario'] for c in every]
    if scenario not in keys:
      return None
    i = keys.index(scenario)
    best = max(stars, before)
    later = every[i + 1] if i + 1 < len(every) else None
    challenge = daily(day)
    return {
        'number': i + 1,
        'of': len(every),
        'title': every[i]['title'],
        'when': every[i]['when'],
        'stars': stars,
        'best': best,
        'new_best': stars > before,
        'next': (
            None
            if later is None
            else {
                'scenario': later['scenario'],
                'title': later['title'],
                'when': later['when'],
                'open': best > 0,
                'opened': before == 0 and stars > 0,
            }
        ),
        'daily': (
            offered()
            and (scenario, seed) == (challenge['scenario'], challenge['seed'])
        ),
        'total': sum(c['stars'] for c in self.campaign(day)['chapters']),
    }

  def campaign(self, day: datetime.date) -> dict[str, Any]:
    """The chapters with their best stars and whether each is open, and the

    day's challenge with its best stars.
    """
    best = self._load()
    found, opened = [], True
    for chapter in chapters():
      stars = int(best['scenarios'].get(chapter['scenario'], 0))
      found.append({**chapter, 'stars': stars, 'open': opened})
      opened = stars > 0
    challenge = daily(day)
    return {
        'chapters': found,
        'daily': (
            {
                **challenge,
                'stars': int(best['daily'].get(challenge['date'], 0)),
            }
            if offered()
            else None
        ),
    }

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

"""Chronological sorting of entity memory strings.

Handles four timestamp formats found in Concordia Island observations:

1. Full date: ``// location [Thursday, January 1st, 7:00 AM]: text``
2. Weekday + time: ``// home [Friday, 7:00 AM]: text``
3. Time only: ``// marketplace [11:00 PM]: text``
4. No timestamp (backstory / reflections)

Format (2) is emitted by ``food_consumption.py`` and ``fiscal_configs.py``.
Format (3) is the fallback from ``marketplace_night_gm.py`` when the clock
component is unavailable.
"""

import datetime
import re

_WEEKDAYS = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]

# Compiled patterns for performance (called once per memory per entity).
_FULL_DATE_RE = re.compile(
    r'\[.*?([A-Z][a-z]+) (\d+)(?:st|nd|rd|th), (\d+:\d+ [AP]M)\]'
)
_WEEKDAY_TIME_RE = re.compile(r'\[(\w+day),\s*(\d+:\d+\s*[AP]M)\]')
_TIME_ONLY_RE = re.compile(r'\[(\d+:\d+ [AP]M)\]')


def sort_entity_memories(memories: list[str]) -> list[str]:
  """Sort entity memories chronologically, preserving untimestamped order.

  Untimestamped entries (backstory, reflections) inherit the datetime of
  the most recent preceding timestamped entry — or ``datetime.min`` if none
  has been seen yet.  This keeps them anchored to the event they follow
  rather than floating to the start of the list.

  Within the same datetime, '[journal]' reflections sort after all other
  entries — a reflection carries the frozen start timestamp of the
  conversation it describes, so it must follow that conversation rather than
  tie with it. All remaining ties preserve original insertion order via a
  final sort on the original index.

  Args:
    memories: List of memory strings, possibly out of chronological order.

  Returns:
    A new list sorted chronologically.
  """
  if not memories:
    return memories

  parsed: list[tuple[datetime.datetime, int, str]] = []
  last_dt = datetime.datetime.min

  for idx, mem in enumerate(memories):
    mem_str = str(mem)

    # --- Pattern 1: Full date [Thursday, January 1st, 11:00 PM] ---
    match = _FULL_DATE_RE.search(mem_str)
    if match:
      try:
        dt = datetime.datetime.strptime(
            f'2026 {match.group(1)} {match.group(2)} {match.group(3)}',
            '%Y %B %d %I:%M %p',
        )
        last_dt = dt
        parsed.append((dt, idx, mem))
        continue
      except ValueError:
        pass

    # --- Pattern 2: Weekday + time [Friday, 7:00 AM] ---
    weekday_match = _WEEKDAY_TIME_RE.search(mem_str)
    if weekday_match and last_dt != datetime.datetime.min:
      w_name = weekday_match.group(1)
      if w_name in _WEEKDAYS:
        try:
          t = datetime.datetime.strptime(
              weekday_match.group(2).strip(), '%I:%M %p'
          ).time()
          target_wd = _WEEKDAYS.index(w_name)
          current_wd = last_dt.weekday()
          day_diff = (target_wd - current_wd) % 7
          target_date = last_dt.date() + datetime.timedelta(days=day_diff)
          inferred_dt = datetime.datetime.combine(target_date, t)
          last_dt = inferred_dt
          parsed.append((inferred_dt, idx, mem))
          continue
        except ValueError:
          pass

    # --- Pattern 3: Time only [11:00 PM] ---
    time_match = _TIME_ONLY_RE.search(mem_str)
    if time_match and last_dt != datetime.datetime.min:
      try:
        t = datetime.datetime.strptime(
            time_match.group(1), '%I:%M %p'
        ).time()
        inferred_dt = datetime.datetime.combine(last_dt.date(), t)
        if inferred_dt < last_dt:
          inferred_dt += datetime.timedelta(days=1)
        last_dt = inferred_dt
        parsed.append((inferred_dt, idx, mem))
        continue
      except ValueError:
        pass

    # --- Pattern 4: No timestamp / unrecognised ---
    parsed.append((last_dt, idx, mem))

  # A reflection shares the frozen start timestamp of the conversation it
  # describes, so ties are broken to place '[journal]' entries after the
  # observations they reflect on. Without this, equal timestamps fall back to
  # insertion order, which is non-deterministic across concurrent GM shards.
  parsed.sort(key=lambda x: (x[0], '[journal]' in str(x[2]), x[1]))
  return [x[2] for x in parsed]

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

"""Agent-side schedule awareness component.

Gives agents real-time awareness of the time of day, whether they should
be at work, and contextual cues for socializing during evenings and
weekends. This component is visible to the agent's SituationPerception,
PersonBySituation, and ConcatActComponent — ensuring the agent actually
*acts* on schedule information rather than just having it buried in GM
memory.

Usage:
  Add this component to IslandEntity's components dict and component_order.
  It reads the GM clock via the agent's memory (looking for [schedule] tags)
  or accepts direct time injection from the GM.
"""

from collections.abc import Sequence
import datetime
import re
import threading

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state

# Regex to parse time from GM observation prefix:
# e.g. "// sunset_apartments_unit_14 [Thursday, January 1st, 7:00 AM]:"
_OBS_TIME_PATTERN = re.compile(
    r'\[(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+'
    r'(\w+)\s+(\d+)(?:st|nd|rd|th),\s+'
    r'(\d{1,2}):(\d{2})\s*(AM|PM)\]'
)

_MONTH_MAP = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12,
}

# Time boundaries
WORK_START_HOUR = 9
WORK_END_HOUR = 17
EVENING_SOCIAL_HOUR = 18
BEDTIME_HOUR = 22

# Social location suggestions
EVENING_SOCIAL_LOCATIONS = [
    'cafe',
    'restaurant',
    'town_square',
    'park',
    'beach',
    'community_center',
]

WEEKEND_SOCIAL_LOCATIONS = [
    'beach',
    'park',
    'cafe',
    'market',
    'town_square',
    'community_center',
    'marina',
    'library',
]


class ScheduleAwareness(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Agent component providing real-time schedule and social awareness.

  Provides contextual text to the agent about:
  - Current time period (morning, workday, evening, weekend)
  - Whether they should be at work or are free
  - Suggestions for social activities and shared locations
  - Awareness of other residents who might be at social spots

  The component receives time updates from the GM via set_current_time()
  and surfaces them in pre_act so the agent sees them in every decision.
  """

  def __init__(
      self,
      agent_name: str,
      home_place: str,
      work_place: str | None = None,
      available_locations: Sequence[str] = (),
      pre_act_label: str = '\nCurrent schedule',
  ):
    super().__init__(pre_act_label)
    self._agent_name = agent_name
    self._home_place = home_place
    self._work_place = work_place
    self._allowed_locations = set(available_locations)
    self._is_sunset_apts = 'sunset_apartments' in home_place
    # Deliberately an RLock, replacing the plain Lock the base class installs.
    # ActionSpecIgnored.get_pre_act_value holds self._lock while it calls
    # _make_pre_act_value, and our override of _make_pre_act_value re-acquires
    # self._lock. With a non-reentrant Lock that self-deadlocks every time the
    # component is asked for its pre-act value. The base's type annotation is
    # what is wrong here, not this line.
    self._lock = threading.RLock()  # pyrefly: ignore[bad-assignment]

    # Current time state — updated by GM via set_current_time()
    self._current_dt: datetime.datetime | None = None
    self._is_laid_off: bool = False

  def set_current_time(self, dt: datetime.datetime) -> None:
    """Called by GM to update the agent's time awareness."""
    with self._lock:
      self._current_dt = dt

  def set_laid_off(self, laid_off: bool = True) -> None:
    """Mark agent as laid off — they no longer have work to go to."""
    with self._lock:
      self._is_laid_off = laid_off
      if laid_off:
        self._work_place = None

  def pre_observe(self, observation: str) -> str:
    """Parse timestamp from GM observation prefix and update time state.

    The GM prepends observations with a prefix like:
      // sunset_apartments_unit_14 [Thursday, January 1st, 7:00 AM]: ...
    We extract the date/time from this to keep the agent time-aware.

    Args:
      observation: The observation string from the GM.

    Returns:
      An empty string (this component does not modify the observation).
    """
    matches = list(_OBS_TIME_PATTERN.finditer(observation))
    if matches:
      match = matches[-1]
      month_name, day_str, hour_str, minute_str, ampm = match.groups()
      month = _MONTH_MAP.get(month_name, 1)
      day = int(day_str)
      hour = int(hour_str)
      minute = int(minute_str)
      if ampm == 'PM' and hour != 12:
        hour += 12
      elif ampm == 'AM' and hour == 12:
        hour = 0
      try:
        # Use year 2026 as default (matches sim start)
        dt = datetime.datetime(2026, month, day, hour, minute)
        with self._lock:
          self._current_dt = dt
      except ValueError:
        pass  # Invalid date, skip
    return ''

  def _make_pre_act_value(self) -> str:
    with self._lock:
      if self._current_dt is None:
        return ''

      dt = self._current_dt
      hour = dt.hour
      weekday = dt.weekday()  # 0=Mon, 6=Sun
      is_weekend = weekday >= 5

      day_name = dt.strftime('%A')
      time_str = dt.strftime('%I:%M %p')

      lines = [f'It is {day_name}, {time_str}.']

      if is_weekend:
        lines.append(self._weekend_cue(day_name))
      elif hour < WORK_START_HOUR:
        lines.append(self._morning_cue())
      elif hour < WORK_END_HOUR:
        lines.append(self._work_cue())
      elif hour < BEDTIME_HOUR:
        lines.append(self._evening_cue())
      else:
        lines.append(self._bedtime_cue())

      result = ' '.join(lines)

      self._logging_channel({
          'Key': self.get_pre_act_label(),
          'Summary': result[:100],
          'Value': result,
      })
      return result

  def _filter_spots(self, spots: list[str]) -> list[str]:
    if not self._allowed_locations:
      return list(spots)
    return [spot for spot in spots if spot in self._allowed_locations]

  def _morning_cue(self) -> str:
    if self._work_place and not self._is_laid_off:
      return (
          f'{self._agent_name} should get ready for work. '
          f'The workday starts at {WORK_START_HOUR}:00 AM at '
          f'{self._work_place}.'
      )
    filtered = self._filter_spots(WEEKEND_SOCIAL_LOCATIONS)
    return (
        f'{self._agent_name} does not have work today. '
        f'{self._agent_name} can visit shared spaces like '
        f'{", ".join(filtered[:4])} to see other residents.'
    )

  def _work_cue(self) -> str:
    if self._work_place and not self._is_laid_off:
      return (
          f'It is currently work hours. {self._agent_name} should be at '
          f'{self._work_place}.'
      )
    filtered = self._filter_spots(EVENING_SOCIAL_LOCATIONS)
    return (
        f'{self._agent_name} does not have a job right now. '
        f'{self._agent_name} is free and could visit shared spaces like '
        f'{", ".join(filtered[:3])} where other residents '
        'might be spending time.'
    )

  def _evening_cue(self) -> str:
    social_spots = self._filter_spots(EVENING_SOCIAL_LOCATIONS)
    if self._is_sunset_apts:
      sunset_room = 'sunset_apartments_common_room'
      if not self._allowed_locations or sunset_room in self._allowed_locations:
        social_spots.insert(0, sunset_room)

    spots_str = ', '.join(social_spots[:5])
    lines = [
        'The workday is over.',
        f'{self._agent_name} is free for the evening.',
        f'Other residents may be socializing at places like {spots_str}.',
        (
            f'{self._agent_name} could go to one of these shared spaces '
            'to meet and talk with neighbors, or head home.'
        ),
    ]
    return ' '.join(lines)

  def _weekend_cue(self, day_name: str) -> str:
    filtered = self._filter_spots(WEEKEND_SOCIAL_LOCATIONS)
    spots_str = ', '.join(filtered[:5])
    lines = [
        f'It is {day_name} — no work today.',
        f'{self._agent_name} has the whole day free.',
        f'Many residents will be out and about at places like {spots_str}.',
        (
            f'{self._agent_name} could visit one of these shared spaces to '
            'relax, socialize, or run errands.'
        ),
    ]
    return ' '.join(lines)

  def _bedtime_cue(self) -> str:
    return (
        f'It is getting late. {self._agent_name} should head home '
        f'to {self._home_place} for the night.'
    )

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'current_dt': (
              self._current_dt.isoformat() if self._current_dt else None
          ),
          'is_laid_off': self._is_laid_off,
          'work_place': self._work_place,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      dt_str = component_state.as_optional_str(state, 'current_dt')
      if dt_str:
        self._current_dt = datetime.datetime.fromisoformat(dt_str)
      else:
        self._current_dt = None
      self._is_laid_off = component_state.as_bool(state, 'is_laid_off')
      wp = component_state.as_optional_str(state, 'work_place')
      if wp is not None:
        self._work_place = wp

# Copyright 2023 DeepMind Technologies Limited.
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


"""This construct implements scheduled events."""

import dataclasses
import datetime
from typing import Callable, Optional

from concordia.typing import component


@dataclasses.dataclass(frozen=True)
class EventData:
  """Represents an event scheduled to happen at a specific time in the future.

  Attributes:
    time: when the event will happen.
    description: string to use to condition the game master's narration of the
      event.
    trigger: a function to call when event happens [optional]
  """

  time: datetime.datetime
  description: str
  trigger: Optional[Callable[[], None]] = None


class Schedule(component.Component):
  """A memory construct that represents a schedule of events."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      schedule: dict[str, EventData],
      players_observe: bool = False,
  ):
    self._clock_now = clock_now
    self._schedule = schedule
    self._state = None
    self._last_update = datetime.datetime.min
    self._players_observe = players_observe

  def name(self) -> str:
    return 'Current events'

  def state(self) -> str | None:
    return self._state

  def partial_state(
      self,
      player_name: str,
  ) -> str | None:
    """Return a player-specific view of the construct's state."""
    if self._players_observe:
      if self._state:
        return self._state

  def update(self) -> None:
    if self._last_update == self._clock_now():
      return
    self._last_update = self._clock_now()
    now = self._clock_now()
    events = []
    events_to_pop = []
    for event_name, event_data in self._schedule.items():
      if now == event_data.time:
        events.append(event_data.description)
        if event_data.trigger is not None:
          event_data.trigger()
        events_to_pop.append(event_name)

    for event_name in events_to_pop:
      self._schedule.pop(event_name)

    if events:
      self._state = '\n'.join(events)
    else:
      self._state = None

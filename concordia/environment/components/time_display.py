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


"""This component shows current time interval."""


from concordia.typing import clock
from concordia.typing import component


class TimeDisplay(component.Component):
  """Tracks the status of players."""

  def __init__(
      self,
      game_clock: clock.GameClock,
      name: str = 'Current time interval',
  ):
    self._clock = game_clock
    self._name = name

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._clock.current_time_interval_str()

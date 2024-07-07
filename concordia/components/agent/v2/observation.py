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

"""A simple component to receive observations."""

from collections.abc import Callable
import datetime
from concordia.associative_memory import associative_memory
from concordia.components.agent.v2 import action_spec_ignored

import overrides


class Observation(action_spec_ignored.ActionSpecIgnored):
  """A simple component to receive observations."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      timeframe: datetime.timedelta,
      memory: associative_memory.AssociativeMemory,
  ):
    self._timeframe = timeframe
    self._clock_now = clock_now
    self._memory = memory

  def pre_observe(
      self,
      observation: str,
  ) -> str:
    self._memory.add(
        f'[observation] {observation}',
        tags=['observation'],
    )
    return ''

  @overrides.override
  def make_pre_act_context(self) -> str:
    mems = self._memory.retrieve_time_interval(
        self._clock_now() - self._timeframe, self._clock_now(), add_time=True
    )
    # removes memories that are not observations
    mems = [mem for mem in mems if '[observation]' in mem]
    return '\n'.join(mems) + '\n'

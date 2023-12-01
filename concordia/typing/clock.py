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


"""An abstract class of a clock for synchronising the simulation."""

import abc
import datetime


class GameClock(metaclass=abc.ABCMeta):
  """An abstract clock for synchronising simulation."""

  @abc.abstractmethod
  def advance(self):
    """Advances the clock."""
    raise NotImplementedError

  def set(self, time: datetime.datetime):
    """Sets the clock to a specific time."""
    raise NotImplementedError

  def now(self) -> datetime.datetime:
    """Returns the current time."""
    raise NotImplementedError

  def get_step_size(self) -> datetime.timedelta:
    """Returns the step size."""
    raise NotImplementedError

  def get_step(self) -> int:
    """Returns the current step."""
    raise NotImplementedError

  def current_time_interval_str(self) -> str:
    """Returns the current time interval."""
    raise NotImplementedError

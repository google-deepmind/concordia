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


"""A clock for synchronising simulacra."""

from collections.abc import Sequence
import contextlib
import datetime
import threading

from concordia.typing import clock

_DEFAULT_STEP_SIZE = datetime.timedelta(minutes=1)


class FixedIntervalClock(clock.GameClock):
  """A fixed-interval clock for synchronising simulacra."""

  def __init__(
      self,
      start: datetime.datetime | None = None,
      step_size: datetime.timedelta = _DEFAULT_STEP_SIZE,
  ):
    """Initializes the clock.

    Args:
      start: The start time of the clock. If None, the current time is used.
      step_size: The step size of the clock.
    """
    if start is None:
      self._start = datetime.datetime.now()
    else:
      self._start = start
    self._step_size = step_size
    self._step = 0

    self._step_lock = threading.Lock()

  def advance(self):
    """Advances time by step_size."""
    with self._step_lock:
      self._step += 1

  def set(self, time: datetime.datetime):
    with self._step_lock:
      self._step = (time - self._start) // self._step_size

  def now(self) -> datetime.datetime:
    with self._step_lock:
      return self._start + self._step * self._step_size

  def get_step_size(self) -> datetime.timedelta:
    return self._step_size

  def get_step(self) -> int:
    with self._step_lock:
      return self._step

  def current_time_interval_str(self) -> str:
    this_time = self.now()
    next_time = this_time + self._step_size

    time_string = this_time.strftime(
        ' %d %b %Y [%H:%M:%S - '
    ) + next_time.strftime('%H:%M:%S]')
    return time_string


class MultiIntervalClock(clock.GameClock):
  """A multi-interval clock for synchronising simulacra.

  This clock takes in multiple step sizes, which can be switched between using
  gear_up and gear_down. Important: when advancing, the clock switches to the
  next step in the current gear and zeros all steps of all higher gears. For
  example if step sizes are 1 hour and 10 minutes and current time is 15:40,
  then going back to lowest gear and advancing will yield 16:00 and not 16:40.
  """

  def __init__(
      self,
      start: datetime.datetime | None = None,
      step_sizes: Sequence[datetime.timedelta] = (_DEFAULT_STEP_SIZE,),
  ):
    """Initializes the clock.

    Args:
      start: The start time of the clock. If None, the current time is used.
      step_sizes: The step sizes of the clock.

    Raises:
      RuntimeError: If step_sizes are not sorted from lowest to highest.
    """
    if start is None:
      self._start = datetime.datetime.now()
    else:
      self._start = start

    # the default makes it a fixed interval clock
    if step_sizes != sorted(step_sizes, reverse=True):
      raise RuntimeError('Step sizes have to be sorted from lowest to highest.')

    self._step_sizes = step_sizes
    self._steps = [0] * len(step_sizes)
    self._current_gear = 0

    self._step_lock = threading.RLock()
    self._control_lock = threading.RLock()

  def _gear_up(self) -> None:
    with self._step_lock:
      if self._current_gear + 1 >= len(self._step_sizes):
        raise RuntimeError('Already in highest gear.')
      self._current_gear += 1

  def _gear_down(self) -> None:
    with self._step_lock:
      if self._current_gear == 0:
        raise RuntimeError('Already in lowest gear.')
      self._current_gear -= 1

  @contextlib.contextmanager
  def higher_gear(self):
    with self._control_lock:
      self._gear_up()
      try:
        yield
      finally:
        self._gear_down()

  def advance(self):
    """Advances time by step_size."""
    with self._control_lock, self._step_lock:
      self._steps[self._current_gear] += 1
      for gear in range(self._current_gear + 1, len(self._step_sizes)):
        self._steps[gear] = 0
      self.set(self.now())  # resolve the higher gear running over the lower

  def set(self, time: datetime.datetime):
    with self._control_lock, self._step_lock:
      remainder = time - self._start
      for gear, step_size in enumerate(self._step_sizes):
        self._steps[gear] = remainder // step_size
        remainder -= step_size * self._steps[gear]

  def now(self) -> datetime.datetime:
    with self._step_lock:
      output = self._start
      for gear, step_size in enumerate(self._step_sizes):
        output += self._steps[gear] * step_size
      return output

  def get_step_size(self) -> datetime.timedelta:
    with self._step_lock:
      return self._step_sizes[self._current_gear]

  def get_step(self) -> int:
    """Returns the current step in the lowest gear."""
    with self._step_lock:
      # this is used for logging, so makes sense to use lowest gear
      return self._steps[0]

  def current_time_interval_str(self) -> str:
    with self._step_lock:
      this_time = self.now()
      next_time = this_time + self._step_sizes[self._current_gear]

      time_string = this_time.strftime(
          ' %d %b %Y [%H:%M - '
      ) + next_time.strftime('%H:%M]')
      return time_string

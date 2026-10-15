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

"""Numeric (non-calendar) simulation time.

`interrupt_time_model.DatetimeTimeModel` is limited to the years 1-9999,
`GenerativeTimeModel` needs a language model to format time, and the contrib
`FixedIncrementClock` steps a `datetime` in minutes. Simulations set in deep
time, in fantasy calendars or in abstract turns only need a number.

* `NumericTimeModel` implements `interrupt_time_model.TimeModel` with float
  timestamps measured in a base unit (years by default) and unit-suffixed
  durations such as `'10Myr'`, `'0.5 Myr'`, `'100yr'` or `'1h30m'`.
* `NumericClock` is a deterministic, LLM-free game master component that holds
  the current time, advances it by a variable step (for example per era) and
  shows the formatted time in the game master's context.
"""

from collections.abc import Callable, Mapping
import copy
import math
import re
import threading
from typing import Any, override

from concordia.components.agent import action_spec_ignored
from concordia.components.game_master import interrupt_time_model
from concordia.typing import entity_component

DEFAULT_NUMERIC_CLOCK_COMPONENT_KEY = '__clock__'
DEFAULT_NUMERIC_CLOCK_PRE_ACT_LABEL = 'Current time'

_SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Unit multipliers in years. Lookup is case-insensitive except that 'M' alone
# is not a unit, so 'm' always means minutes (as in `interrupt_time_model`).
YEAR_UNITS: Mapping[str, float] = {
    's': 1.0 / _SECONDS_PER_YEAR,
    'sec': 1.0 / _SECONDS_PER_YEAR,
    'm': 60.0 / _SECONDS_PER_YEAR,
    'min': 60.0 / _SECONDS_PER_YEAR,
    'h': 3600.0 / _SECONDS_PER_YEAR,
    'hr': 3600.0 / _SECONDS_PER_YEAR,
    'd': 1.0 / 365.25,
    'day': 1.0 / 365.25,
    'days': 1.0 / 365.25,
    'wk': 7.0 / 365.25,
    'week': 7.0 / 365.25,
    'weeks': 7.0 / 365.25,
    'mo': 1.0 / 12.0,
    'month': 1.0 / 12.0,
    'months': 1.0 / 12.0,
    'y': 1.0,
    'yr': 1.0,
    'yrs': 1.0,
    'year': 1.0,
    'years': 1.0,
    'kyr': 1e3,
    'ka': 1e3,
    'myr': 1e6,
    'ma': 1e6,
    'gyr': 1e9,
    'ga': 1e9,
}

_TERM_RE = re.compile(
    r'\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)'
)

# (threshold in years, divisor, suffix) used by `format_years`.
_DISPLAY_UNITS = (
    (1e9, 1e9, 'Gyr'),
    (1e6, 1e6, 'Myr'),
    (1e3, 1e3, 'kyr'),
    (1.0, 1.0, 'yr'),
)


def parse_duration(
    duration_str: str,
    units: Mapping[str, float] | None = None,
    default_unit: str = 'yr',
) -> float:
  """Parses a duration like `'10Myr'`, `'0.5 Myr'` or `'1h30m'`.

  Args:
    duration_str: One or more number-unit terms. A bare number uses
      `default_unit`.
    units: Unit multipliers in base units, keyed by lower-case suffix. Defaults
      to `YEAR_UNITS`.
    default_unit: Unit for bare numbers.

  Returns:
    The duration in base units.

  Raises:
    ValueError: If the string is empty or contains an unknown unit.
  """
  units = YEAR_UNITS if units is None else units
  text = duration_str.strip()
  if not text:
    raise ValueError('Empty duration.')
  total = 0.0
  position = 0
  while position < len(text):
    match = _TERM_RE.match(text, position)
    if match is None or match.end() == position:
      raise ValueError(f'Cannot parse duration: {duration_str!r}')
    number, unit = match.group(1), match.group(2) or default_unit
    multiplier = units.get(unit.lower())
    if multiplier is None:
      raise ValueError(f'Unknown time unit {unit!r} in {duration_str!r}')
    total += float(number) * multiplier
    position = match.end()
    while position < len(text) and text[position] in ' ,':
      position += 1
  return total


def format_years(years: float, precision: int = 3) -> str:
  """Formats a span in years with the largest fitting unit, e.g. `'2.5 Gyr'`."""
  magnitude = abs(years)
  for threshold, divisor, suffix in _DISPLAY_UNITS:
    if magnitude >= threshold:
      return f'{years / divisor:.{precision}g} {suffix}'
  if magnitude == 0:
    return '0 yr'
  return f'{years * 365.25:.{precision}g} days'


class NumericTimeModel(interrupt_time_model.TimeModel):
  """A `TimeModel` whose timestamps are floats in a base unit.

  Timestamps compare like numbers, so they work with the interrupt scheduler's
  sorted event queue, and they serialize losslessly with `repr`.
  """

  def __init__(
      self,
      start: float = 0.0,
      *,
      units: Mapping[str, float] | None = None,
      default_unit: str = 'yr',
      formatter: Callable[[float], str] | None = None,
  ):
    """Initializes the time model.

    Args:
      start: Initial timestamp in base units.
      units: Unit multipliers in base units, keyed by lower-case suffix. The
        default treats the base unit as one year.
      default_unit: Unit assumed for bare numbers in durations.
      formatter: Optional function formatting timestamps for display. Defaults
        to `format_years`, which is only meaningful when the base unit is a
        year.
    """
    units = YEAR_UNITS if units is None else units
    self._start = float(start)
    self._units = {k.lower(): v for k, v in units.items()}
    self._default_unit = default_unit
    self._formatter = formatter or format_years

  def initial_time(self) -> float:
    return self._start

  def duration(self, duration_str: str) -> float:
    """Returns a duration string in base units."""
    return parse_duration(duration_str, self._units, self._default_unit)

  def add_duration(self, time: float, duration_str: str) -> float:
    return float(time) + self.duration(duration_str)

  def parse_absolute_time(self, time_str: str, current_time: float) -> float:
    """Parses an absolute timestamp such as `'-2.4Gyr'` or `'1500'`.

    Args:
      time_str: A number with an optional unit, interpreted as a timestamp.
      current_time: The current simulated time.

    Returns:
      The timestamp, which must be strictly after `current_time`.

    Raises:
      ValueError: If the string cannot be parsed or is not in the future.
    """
    value = parse_duration(time_str, self._units, self._default_unit)
    if value <= current_time:
      raise ValueError(
          f'Absolute time {time_str!r} is not after the current time.'
      )
    return value

  def format_time(self, time: float) -> str:
    return self._formatter(float(time))

  def serialize_time(self, time: float) -> str:
    return repr(float(time))

  def deserialize_time(self, s: str) -> float:
    value = float(s)
    if math.isnan(value):
      raise ValueError('NaN is not a valid timestamp.')
    return value


class NumericClock(action_spec_ignored.ActionSpecIgnored):
  """A deterministic variable-step clock for game masters.

  The clock never calls a language model. Game logic calls `advance()` once per
  simulation step; `set_step()` changes the step size, for example when an era
  changes. The pre-act value is the formatted current time.
  """

  def __init__(
      self,
      time_model: NumericTimeModel | None = None,
      *,
      step: str = '1yr',
      pre_act_label: str = DEFAULT_NUMERIC_CLOCK_PRE_ACT_LABEL,
  ):
    """Initializes the clock.

    Args:
      time_model: Time model used to parse steps and format time. Defaults to
        `NumericTimeModel()` (years from zero).
      step: Initial step duration, e.g. `'10Myr'`.
      pre_act_label: Label shown in the game master's context.
    """
    super().__init__(pre_act_label)
    self._time_model = time_model or NumericTimeModel()
    self._step_str = step
    self._step = self._time_model.duration(step)
    self._now = self._time_model.initial_time()
    self._ticks = 0
    self._clock_lock = threading.Lock()

  @property
  def time_model(self) -> NumericTimeModel:
    """Returns the time model."""
    return self._time_model

  def now(self) -> float:
    """Returns the current time in base units."""
    with self._clock_lock:
      return self._now

  def ticks(self) -> int:
    """Returns the number of completed `advance` calls."""
    with self._clock_lock:
      return self._ticks

  def step(self) -> str:
    """Returns the current step duration string."""
    with self._clock_lock:
      return self._step_str

  def step_size(self) -> float:
    """Returns the current step duration in base units."""
    with self._clock_lock:
      return self._step

  def set_step(self, step: str) -> None:
    """Changes the step duration used by future `advance` calls."""
    size = self._time_model.duration(step)
    with self._clock_lock:
      self._step_str = step
      self._step = size

  def advance(self, step: str | None = None) -> float:
    """Advances the clock by one step and returns the new time.

    Args:
      step: Optional one-off step duration. Defaults to the current step.

    Returns:
      The updated time in base units.
    """
    size = self._time_model.duration(step) if step is not None else None
    with self._clock_lock:
      self._now += self._step if size is None else size
      self._ticks += 1
      return self._now

  def formatted_now(self) -> str:
    """Returns the current time formatted by the time model."""
    return self._time_model.format_time(self.now())

  @override
  def _make_pre_act_value(self) -> str:
    return self.formatted_now()

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._clock_lock:
      return {
          'now': self._time_model.serialize_time(self._now),
          'ticks': self._ticks,
          'step': self._step_str,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    step = state.get('step')
    if isinstance(step, str):
      self.set_step(step)
    with self._clock_lock:
      now = state.get('now')
      if isinstance(now, str):
        self._now = self._time_model.deserialize_time(now)
      ticks = state.get('ticks')
      if isinstance(ticks, int):
        self._ticks = ticks

  def __deepcopy__(self, memo: dict[int, Any] | None = None) -> 'NumericClock':
    copied = NumericClock(
        time_model=copy.deepcopy(self._time_model, memo),
        step=self.step(),
        pre_act_label=self.get_pre_act_label(),
    )
    copied.set_state(self.get_state())
    return copied

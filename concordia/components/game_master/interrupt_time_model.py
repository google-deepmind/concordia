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

"""Time model abstractions for interrupt-driven game master orchestration.

This module provides injectable time handling for the interrupt-driven GM
framework. Two implementations are available:

- ``DatetimeTimeModel``: A simple wrapper around ``datetime.datetime``.
  No LLM calls. Suitable for simulations set in ~0001–9999 CE.

- ``GenerativeTimeModel``: Uses an LLM to narrate timestamps for entity-
  facing display, while internally tracking time as an ``int`` counting
  seconds since an arbitrary simulation epoch (0). No range limitations.

All timestamp types must support Python comparison operators (``<``, ``<=``,
``==``, ``>=``, ``>``). This is required by the event queue's sorted order.
"""

import abc
import datetime
import re
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model


def parse_duration_seconds(duration_str: str) -> int:
  """Parses a duration string like '2h', '30m', '1h30m' into seconds.

  Args:
    duration_str: Duration string with optional hours ('h') and minutes ('m').

  Returns:
    The duration in seconds. Defaults to 3600 (1 hour) if unparseable.
  """
  duration_str = duration_str.strip()
  if not duration_str or duration_str == '0':
    return 0
  hours = 0
  minutes = 0
  parsed = False
  if 'h' in duration_str:
    parts = duration_str.split('h', 1)
    hours = int(parts[0])
    rest = parts[1]
    parsed = True
  else:
    rest = duration_str
  if rest.endswith('m'):
    minutes = int(rest[:-1])
    parsed = True
  if not parsed:
    return 3600  # Default fallback: 1 hour.
  return hours * 3600 + minutes * 60


_TIME_OF_DAY_RE = re.compile(r'^(\d{1,2}):(\d{2})$')


def parse_time_of_day_seconds(time_str: str) -> int | None:
  """Parses a 24-hour time-of-day string like '8:00' or '14:30'.

  Accepted format: ``HH:MM`` (24-hour clock, 0–23 hours, 0–59 minutes).

  Args:
    time_str: A time-of-day string.

  Returns:
    Seconds since midnight, or ``None`` if the string does not match.
  """
  m = _TIME_OF_DAY_RE.match(time_str.strip())
  if m is None:
    return None
  hour, minute = int(m.group(1)), int(m.group(2))
  if hour > 23 or minute > 59:
    return None
  return hour * 3600 + minute * 60


class TimeModel(metaclass=abc.ABCMeta):
  """Abstract base class for time handling in the interrupt-driven GM.

  Implementations must ensure that timestamps returned by ``initial_time``,
  ``add_duration``, and ``deserialize_time`` support Python's comparison
  operators (``<``, ``<=``, ``==``, ``>=``, ``>``). This is required for
  maintaining the sorted event queue.
  """

  @abc.abstractmethod
  def initial_time(self) -> Any:
    """Returns the starting timestamp for the simulation."""
    raise NotImplementedError()

  @abc.abstractmethod
  def add_duration(self, time: Any, duration_str: str) -> Any:
    """Adds a duration to a timestamp.

    Args:
      time: The current timestamp.
      duration_str: A duration string, e.g. ``'30m'``, ``'2h'``, ``'1h30m'``.

    Returns:
      A new timestamp representing ``time + duration``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def parse_absolute_time(self, time_str: str, current_time: Any) -> Any:
    """Parses an absolute time string into a timestamp.

    Used for timer specifications like ``{"until": "8:00"}``.  The
    ``current_time`` parameter is needed to resolve day-boundary ambiguity:
    if the requested time-of-day is at or before ``current_time``, the
    result should be on the following day.

    Args:
      time_str: An absolute time string, e.g. ``'8:00'``, ``'14:30'``.
      current_time: The current simulated time.

    Returns:
      A timestamp guaranteed to be strictly after ``current_time``.

    Raises:
      ValueError: If ``time_str`` cannot be parsed.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def format_time(self, time: Any) -> str:
    """Formats a timestamp for entity-facing display.

    Args:
      time: A timestamp to format.

    Returns:
      A human-readable string representation.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def serialize_time(self, time: Any) -> str:
    """Converts a timestamp to a JSON-safe string for state serialisation.

    Args:
      time: A timestamp to serialise.

    Returns:
      A string that can be stored in JSON and later deserialised.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def deserialize_time(self, s: str) -> Any:
    """Reconstructs a timestamp from its serialised form.

    Args:
      s: A string produced by ``serialize_time``.

    Returns:
      The original timestamp.
    """
    raise NotImplementedError()


class DatetimeTimeModel(TimeModel):
  """A simple ``datetime``-based time model. No LLM calls.

  Suitable for simulations set in ~0001–9999 CE. This is the default time
  model used when no explicit ``TimeModel`` is provided to the scheduler.
  """

  def __init__(self, start_time: datetime.datetime):
    """Initialises the time model.

    Args:
      start_time: The initial simulated time.
    """
    self._start_time = start_time

  def initial_time(self) -> datetime.datetime:
    return self._start_time

  def add_duration(
      self, time: datetime.datetime, duration_str: str
  ) -> datetime.datetime:
    seconds = parse_duration_seconds(duration_str)
    return time + datetime.timedelta(seconds=seconds)

  def parse_absolute_time(
      self, time_str: str, current_time: datetime.datetime,
  ) -> datetime.datetime:
    """Parses a ``HH:MM`` time string into a datetime.

    If the resulting time is at or before ``current_time``, the date is
    advanced by one day (the entity presumably means "tomorrow").

    Args:
      time_str: A 24-hour time string, e.g. ``'8:00'``, ``'14:30'``.
      current_time: The current simulated time.

    Returns:
      A ``datetime.datetime`` strictly after ``current_time``.

    Raises:
      ValueError: If ``time_str`` is not a valid ``HH:MM`` string.
    """
    seconds = parse_time_of_day_seconds(time_str)
    if seconds is None:
      raise ValueError(
          f'Cannot parse absolute time: {time_str!r}. Expected HH:MM format.'
      )
    target = current_time.replace(
        hour=seconds // 3600,
        minute=(seconds % 3600) // 60,
        second=0,
        microsecond=0,
    )
    if target <= current_time:
      target += datetime.timedelta(days=1)
    return target

  def format_time(self, time: datetime.datetime) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')

  def serialize_time(self, time: datetime.datetime) -> str:
    return time.isoformat()

  def deserialize_time(self, s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s)


class GenerativeTimeModel(TimeModel):
  """An LLM-narrated time model for the interrupt-driven GM.

  Internally tracks time as an ``int`` counting seconds since a simulation
  epoch (0). This removes the range limitations of ``datetime.datetime``
  (~0001–9999 CE), allowing simulations set in any historical period or
  fictional setting.

  The LLM provides human-facing time narration (e.g. "mid-morning on the
  Ides of March"). Duration arithmetic (``add_duration``) is mechanical
  and deterministic; only ``format_time`` involves LLM calls.

  Formatted time strings are cached to avoid redundant LLM calls.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock_prompt: str,
      start_time_label: str = 'the beginning of the simulation',
  ):
    """Initialises the generative time model.

    An initial LLM call calibrates the narration format.

    Args:
      model: The language model used for time narration.
      clock_prompt: A description of the simulation's time setting and how
        time should be narrated. For example: "This simulation takes place
        in ancient Rome. Time passes naturally — an hour of conversation
        takes about an hour."
      start_time_label: Human-readable label for the simulation start time
        (T=0). For example: "Dawn on the Ides of March, 44 BCE". This is
        used as the initial cached narration for ``format_time(0)``.
    """
    self._model = model
    self._clock_prompt = clock_prompt
    self._start_time_label = start_time_label

    # Calibrate narration format via LLM.
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(self._clock_prompt)
    chain_of_thought.statement(
        f'The simulation starts at: {start_time_label}'
    )
    self._clock_description = chain_of_thought.open_question(
        question=(
            'Given the context above, when is the clock updated? How are'
            ' times represented internally? (usually as a number of steps).'
            ' And, how do internal time representations map to the time'
            ' representations communicated to players?'
        ),
        max_tokens=1000,
        terminators=(),
    )

    # Cache: int seconds → narrated string.
    self._format_cache: dict[int, str] = {0: start_time_label}

  def initial_time(self) -> int:
    return 0

  def add_duration(self, time: int, duration_str: str) -> int:
    return time + parse_duration_seconds(duration_str)

  def parse_absolute_time(self, time_str: str, current_time: int) -> int:
    """Parses an absolute time string using the LLM.

    Prompts the LLM to convert the entity's absolute time specification into
    a number of seconds from now, then adds that offset to ``current_time``.
    If the result is not strictly after ``current_time``, one simulated day
    (86400 seconds) is added on the assumption the entity meant "tomorrow".

    Args:
      time_str: An absolute time string, e.g. ``'8:00'``.
      current_time: The current simulated time (seconds from epoch).

    Returns:
      A timestamp (int) strictly after ``current_time``.

    Raises:
      ValueError: If the LLM response cannot be parsed as an integer.
    """
    current_label = self.format_time(current_time)
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        f'{self._clock_description}\n\n'
        f'The simulation started at: {self._start_time_label}\n'
        f'The current simulated time is: {current_label} '
        f'({current_time} seconds since the start).'
    )
    response = prompt.open_question(
        question=(
            f'The entity wants to resume at \'{time_str}\'. How many seconds'
            f' from now until that time? Respond with only an integer.'
        ),
        max_tokens=64,
    )
    try:
      offset = int(response.strip())
    except ValueError as e:
      raise ValueError(
          f'Cannot parse LLM absolute-time response: {response!r}'
      ) from e
    result = current_time + offset
    if result <= current_time:
      result += 86400  # Assume "tomorrow".
    return result

  def format_time(self, time: int) -> str:
    """Formats a timestamp using the LLM. Results are cached."""
    if time in self._format_cache:
      return self._format_cache[time]

    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_str = f'{hours}h {minutes}m {seconds}s'

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        f'{self._clock_description}\n\n'
        f'The simulation started at: {self._start_time_label}\n'
        f'The current simulation clock reads: {time} seconds since the'
        f' start (= {elapsed_str} elapsed).'
    )
    narration = prompt.open_question(
        question=(
            'Express the current time in a style appropriate for the'
            ' simulation setting. Respond with only the formatted time,'
            ' nothing else.'
        ),
        max_tokens=128,
    )

    self._format_cache[time] = narration
    return narration

  def serialize_time(self, time: int) -> str:
    return str(time)

  def deserialize_time(self, s: str) -> int:
    return int(s)

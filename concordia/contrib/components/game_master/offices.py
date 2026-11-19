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

"""Offices: named positions, who holds them, for how long, and who held them.

`Offices` keeps a body's offices (a chief, a priest, a judge). Each has a
holder or is vacant, a `term` in steps (0: for life), and the way it passes
on, in the game's word (`succession`: say 'election', 'heredity' or
'acclaim'). The game installs a holder (`install`); a term that runs out,
a holder who leaves (`forget`) or steps down (`vacate`) leaves the office
vacant. `vacant` lists the offices to fill, with their succession, and the
game decides who fills each: by its own vote, an heir, the best reputed.
Each office remembers its last `max_history` holders, so an office's record
stays bounded however long a body runs; how many offices there are is the
game's (it creates them).

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_OFFICES_COMPONENT_KEY`). It adds nothing to prompts: the game
decides what an office lets its holder do, and tells who holds it.

  offices = offices_lib.Offices()
  offices.create('chief', succession='acclaim', term=20)
  offices.install('chief', 'Ada', step=3, how='acclaimed')
  offices.holder('chief')  # 'Ada', until step 23.
"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The key a game master registers its offices under, by convention.
DEFAULT_OFFICES_COMPONENT_KEY = '__offices__'

# How many past holders each office remembers.
MAX_HISTORY = 20


@dataclasses.dataclass(frozen=True)
class Tenure:
  """A holder's time in an office.

  Attributes:
    office: The office.
    holder: Who held it.
    began: The step it began.
    ended: The step it ended (None while it lasts).
    how: How it was had (the game's word, e.g. 'elected').
    why_ended: Why it ended ('term', 'left', or the game's word; '' while
      it lasts).
  """

  office: str
  holder: str
  began: int
  ended: int | None = None
  how: str = ''
  why_ended: str = ''


@dataclasses.dataclass
class _Office:
  name: str
  succession: str
  term: int
  holder: str = ''
  began: int = 0
  how: str = ''
  history: list[Tenure] = dataclasses.field(default_factory=list)


class Offices(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A body's offices and their holders. Thread-safe."""

  def __init__(self, max_history: int = MAX_HISTORY):
    """Initializes the offices, with none created.

    Args:
      max_history: How many past holders each office remembers.

    Raises:
      ValueError: If `max_history` is below 1.
    """
    super().__init__()
    if max_history < 1:
      raise ValueError(f'max_history must be at least 1: {max_history}.')
    self._max_history = int(max_history)
    self._lock = threading.RLock()
    self._offices: dict[str, _Office] = {}

  def create(self, office: str, succession: str = '', term: int = 0) -> None:
    """Creates an office, vacant (an existing one keeps its holder).

    Args:
      office: Its name.
      succession: How it passes on, in the game's word.
      term: Its term in steps (0: for life).

    Raises:
      ValueError: If `term` is negative.
    """
    if term < 0:
      raise ValueError(f'term must not be negative, not {term}.')
    with self._lock:
      found = self._offices.get(office)
      if found is None:
        self._offices[office] = _Office(office, str(succession), int(term))
      else:
        found.succession, found.term = str(succession), int(term)

  def install(self, office: str, holder: str, step: int, how: str = '') -> None:
    """Installs a holder (the one before, if any, steps down).

    Args:
      office: Which office (created if it does not exist, for life).
      holder: Who.
      step: When its tenure begins.
      how: How it was had (the game's word).
    """
    with self._lock:
      found = self._offices.setdefault(office, _Office(office, '', 0))
      if found.holder:
        self._end(found, step, 'succeeded')
      found.holder, found.began, found.how = str(holder), int(step), str(how)
      self._logging_channel({
          'Key': 'Office',
          'Value': {'office': office, 'holder': holder, 'how': how},
      })

  def vacate(self, office: str, step: int, why: str = '') -> Tenure | None:
    """The holder steps down (or is removed).

    Args:
      office: Which office.
      step: When.
      why: Why (the game's word).

    Returns:
      The tenure that ended, or None if the office was vacant.
    """
    with self._lock:
      found = self._offices.get(office)
      if found is None or not found.holder:
        return None
      return self._end(found, step, why)

  def forget(self, name: str, step: int) -> list[Tenure]:
    """One who leaves vacates every office it holds.

    Args:
      name: Who left.
      step: When.

    Returns:
      The tenures that ended.
    """
    with self._lock:
      return [
          self._end(found, step, 'left')
          for found in self._ordered()
          if found.holder == name
      ]

  def expire(self, step: int) -> list[Tenure]:
    """Terms that have run out by `step` end; the offices fall vacant.

    Returns:
      The tenures that ended.
    """
    with self._lock:
      return [
          self._end(found, step, 'term')
          for found in self._ordered()
          if found.holder and found.term and found.began + found.term <= step
      ]

  def vacant(self) -> list[tuple[str, str]]:
    """The offices with no holder: (office, succession), by name."""
    with self._lock:
      return [(o.name, o.succession) for o in self._ordered() if not o.holder]

  def holder(self, office: str) -> str:
    """Who holds an office ('' if vacant or unknown)."""
    with self._lock:
      found = self._offices.get(office)
      return found.holder if found is not None else ''

  def holds(self, name: str) -> list[str]:
    """The offices someone holds, by name."""
    with self._lock:
      return [o.name for o in self._ordered() if o.holder == name]

  def tenure(self, office: str) -> Tenure | None:
    """The current holder's tenure (None if vacant or unknown)."""
    with self._lock:
      found = self._offices.get(office)
      if found is None or not found.holder:
        return None
      return Tenure(found.name, found.holder, found.began, how=found.how)

  def history(self, office: str) -> list[Tenure]:
    """An office's past holders, the latest last."""
    with self._lock:
      found = self._offices.get(office)
      return list(found.history) if found is not None else []

  def offices(self) -> list[dict[str, Any]]:
    """Every office: {name, succession, term, holder, began, how, ends}."""
    with self._lock:
      return [
          {
              'name': o.name,
              'succession': o.succession,
              'term': o.term,
              'holder': o.holder,
              'began': o.began if o.holder else None,
              'how': o.how,
              'ends': o.began + o.term if o.holder and o.term else None,
          }
          for o in self._ordered()
      ]

  def _end(self, found: _Office, step: int, why: str) -> Tenure:
    """Ends the holder's tenure (holds lock)."""
    tenure = Tenure(
        found.name, found.holder, found.began, int(step), found.how, str(why)
    )
    found.history.append(tenure)
    del found.history[: max(0, len(found.history) - self._max_history)]
    found.holder, found.began, found.how = '', 0, ''
    return tenure

  def _ordered(self) -> list[_Office]:
    return [self._offices[k] for k in sorted(self._offices)]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'offices': [
              {
                  'name': o.name,
                  'succession': o.succession,
                  'term': o.term,
                  'holder': o.holder,
                  'began': o.began,
                  'how': o.how,
                  'history': [dataclasses.asdict(t) for t in o.history],
              }
              for o in self._ordered()
          ]
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('offices')
    records: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    restored: dict[str, _Office] = {}
    for record in records:
      try:
        found = _Office(
            name=str(record['name']),
            succession=str(record.get('succession', '')),
            term=int(record.get('term', 0)),
            holder=str(record.get('holder', '')),
            began=int(record.get('began', 0)),
            how=str(record.get('how', '')),
            history=[
                Tenure(
                    office=str(t['office']),
                    holder=str(t['holder']),
                    began=int(t['began']),
                    ended=None if t.get('ended') is None else int(t['ended']),
                    how=str(t.get('how', '')),
                    why_ended=str(t.get('why_ended', '')),
                )
                for t in record.get('history') or ()
            ],
        )
      except (AttributeError, KeyError, TypeError, ValueError):
        continue  # A malformed record is skipped.
      restored[found.name] = found
    with self._lock:
      self._offices = restored

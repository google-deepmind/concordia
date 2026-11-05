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

"""Gatherings: recurring occasions where participants come together.

`Gatherings` is a calendar of gatherings (festivals, market days, councils)
held at named places. Each is held every `every` steps (at steps whose
remainder is its `phase`), and, if it has a `chance` below 1, only when a
draw seeded by the gathering and the step says so. While one is due,
participants may `attend` it; when the step closes (`close`), the game is
handed who came to each, and decides what coming together does (ties made,
news passed, customs seen, matches made). Each gathering keeps how often it
was held, its last attendance and the most it drew, so a calendar that runs
indefinitely stays bounded (at most `max_gatherings` are kept).

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_GATHERINGS_COMPONENT_KEY`). It adds nothing to prompts: the game
offers the move to attend while one is due, and tells what came of it.

  calendar = gatherings.Gatherings(seed=0)
  calendar.add('midsummer', place='the ford', every=5)
  if 'midsummer' in calendar.due(step=10):
    calendar.attend('Ada', 'midsummer', step=10)
  held = calendar.close(step=10)  # [Gathered(id='midsummer', ...)]
"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any, override
import zlib

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# The key a game master registers its gatherings under, by convention.
DEFAULT_GATHERINGS_COMPONENT_KEY = '__gatherings__'

# The most gatherings a calendar keeps.
MAX_GATHERINGS = 20


@dataclasses.dataclass(frozen=True)
class Gathered:
  """A gathering held.

  Attributes:
    id: The gathering.
    place: Where (the game's word for it).
    step: When.
    attendees: Who came, by name.
    host: Who calls it ('' if no one).
  """

  id: str
  place: str
  step: int
  attendees: Sequence[str]
  host: str = ''


@dataclasses.dataclass
class _Gathering:
  id: str
  place: str
  every: int
  phase: int
  chance: float
  host: str
  added: int
  held: int = 0
  last: int | None = None
  last_count: int = 0
  most: int = 0


class Gatherings(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A calendar of recurring gatherings and who comes. Thread-safe."""

  def __init__(self, seed: int = 0, max_gatherings: int = MAX_GATHERINGS):
    """Initializes the calendar, with no gatherings.

    Args:
      seed: Seeds the draws of gatherings held by chance.
      max_gatherings: The most gatherings kept (adding more drops the
        oldest).

    Raises:
      ValueError: If `max_gatherings` is below 1.
    """
    super().__init__()
    if max_gatherings < 1:
      raise ValueError(f'max_gatherings must be at least 1: {max_gatherings}.')
    self._seed = int(seed)
    self._max = int(max_gatherings)
    self._lock = threading.RLock()
    self._gatherings: dict[str, _Gathering] = {}
    # This step's attendance: gathering id -> names.
    self._coming: dict[str, set[str]] = {}

  def add(
      self,
      gathering_id: str,
      place: str,
      every: int,
      phase: int = 0,
      chance: float = 1.0,
      host: str = '',
      step: int = 0,
  ) -> None:
    """Adds (or replaces) a gathering.

    Args:
      gathering_id: Its id.
      place: Where it is held (the game's word for it).
      every: It is held every so many steps.
      phase: At steps whose remainder (by `every`) is this.
      chance: The chance it is held when it falls (1: always).
      host: Who calls it.
      step: When it was added.

    Raises:
      ValueError: If `every` is below 1 or `chance` is outside [0, 1].
    """
    if every < 1:
      raise ValueError(f'every must be at least 1, not {every}.')
    if not 0.0 <= chance <= 1.0:
      raise ValueError(f'chance must be in [0, 1], not {chance}.')
    with self._lock:
      self._gatherings.pop(gathering_id, None)
      self._gatherings[gathering_id] = _Gathering(
          id=str(gathering_id),
          place=str(place),
          every=int(every),
          phase=int(phase) % int(every),
          chance=float(chance),
          host=str(host),
          added=int(step),
      )
      while len(self._gatherings) > self._max:
        oldest = min(self._gatherings.values(), key=lambda g: (g.added, g.id))
        self.remove(oldest.id)

  def remove(self, gathering_id: str) -> None:
    """Ends a gathering (and this step's attendance of it)."""
    with self._lock:
      self._gatherings.pop(gathering_id, None)
      self._coming.pop(gathering_id, None)

  def due(self, step: int) -> list[str]:
    """The gatherings held this step, by id."""
    with self._lock:
      return [g.id for g in self._ordered() if self._falls(g, step)]

  def attend(self, name: str, gathering_id: str, step: int) -> bool:
    """Someone comes to a gathering held this step.

    Args:
      name: Who.
      gathering_id: Which gathering.
      step: The step (a gathering not held then cannot be attended).

    Returns:
      Whether they came (False if it is not held this step).
    """
    with self._lock:
      found = self._gatherings.get(gathering_id)
      if found is None or not self._falls(found, step):
        return False
      self._coming.setdefault(gathering_id, set()).add(name)
      return True

  def close(self, step: int) -> list[Gathered]:
    """The step's gatherings are held: who came to each.

    Every gathering held this step is reported, even with nobody there.
    Attendance is then cleared for the next step.

    Args:
      step: The step.

    Returns:
      The gatherings held, in id order.
    """
    with self._lock:
      held = []
      for gathering in self._ordered():
        if not self._falls(gathering, step):
          continue
        came = sorted(self._coming.get(gathering.id, ()))
        gathering.held += 1
        gathering.last = step
        gathering.last_count = len(came)
        gathering.most = max(gathering.most, len(came))
        held.append(
            Gathered(
                id=gathering.id,
                place=gathering.place,
                step=step,
                attendees=came,
                host=gathering.host,
            )
        )
      self._coming = {}
      self._logging_channel({
          'Key': 'Gatherings',
          'Value': {
              'held': len(held),
              'came': sum(len(g.attendees) for g in held),
          },
      })
      return held

  def forget(self, name: str) -> None:
    """Forgets someone's attendance (one who has left)."""
    with self._lock:
      for coming in self._coming.values():
        coming.discard(name)

  def gatherings(self) -> list[dict[str, Any]]:
    """Every gathering kept: {id, place, every, phase, chance, host, added,
    held (times), last (step), last_count, most}, in id order."""
    with self._lock:
      return [dataclasses.asdict(g) for g in self._ordered()]

  def next_held(self, gathering_id: str, step: int) -> int | None:
    """The next step at or after `step` a gathering falls (None if unknown).

    A gathering held by chance may not be held when it falls.
    """
    with self._lock:
      found = self._gatherings.get(gathering_id)
      if found is None:
        return None
      return step + (found.phase - step) % found.every

  def _falls(self, gathering: _Gathering, step: int) -> bool:
    if step % gathering.every != gathering.phase:
      return False
    if gathering.chance >= 1.0:
      return True
    rng = np.random.default_rng(
        [self._seed, zlib.crc32(gathering.id.encode('utf-8')), step]
    )
    return bool(rng.random() < gathering.chance)

  def _ordered(self) -> list[_Gathering]:
    return [self._gatherings[k] for k in sorted(self._gatherings)]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'gatherings': [dataclasses.asdict(g) for g in self._ordered()],
          'coming': {k: sorted(v) for k, v in sorted(self._coming.items())},
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('gatherings')
    records: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    saved_coming = state.get('coming')
    coming: Any = saved_coming if isinstance(saved_coming, dict) else {}
    with self._lock:
      self._gatherings = {}
      for record in records:
        gathering = _Gathering(
            id=str(record['id']),
            place=str(record.get('place', '')),
            every=max(1, int(record.get('every', 1))),
            phase=int(record.get('phase', 0)),
            chance=float(record.get('chance', 1.0)),
            host=str(record.get('host', '')),
            added=int(record.get('added', 0)),
            held=int(record.get('held', 0)),
            last=record.get('last'),
            last_count=int(record.get('last_count', 0)),
            most=int(record.get('most', 0)),
        )
        self._gatherings[gathering.id] = gathering
      self._coming = {
          str(k): {str(n) for n in v}
          for k, v in coming.items()
          if str(k) in self._gatherings
      }

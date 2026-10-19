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

"""Attention slots: an entity watches only some of its instruments.

An unwatched instrument's `Subscriber` stops receiving readings, so its
belief visibly ages, and it shows a single short line in the prompt. Choosing
what to listen to is strategy and also the control on prompt size. The
number of slots can change during play (better instruments widen it).
"""

# pylint: disable=g-complex-comprehension

from collections.abc import Callable, Mapping, Sequence
import threading
from typing import Any


class Attention:
  """Which of an entity's instruments it is watching."""

  def __init__(
      self,
      instruments: Sequence[str],
      slots: Callable[[], int | None],
  ):
    """Initializes attention, watching the first instruments in order.

    Args:
      instruments: Instrument names, most important first.
      slots: Returns how many instruments can be watched now (None: all).
    """
    self._instruments = list(instruments)
    self._slots = slots
    self._watched: list[str] = []
    self._lock = threading.Lock()
    self.refocused = 0

  def _cap(self) -> int:
    slots = self._slots()
    if slots is None:
      return len(self._instruments)
    return max(1, min(int(slots), len(self._instruments)))

  def _sync(self) -> None:
    cap = self._cap()
    while len(self._watched) > cap:
      self._watched.pop()
    for key in self._instruments:
      if len(self._watched) >= cap:
        break
      if key not in self._watched:
        self._watched.append(key)

  def watching(self, key: str) -> bool:
    """Whether `key` is watched now (instruments not listed always are)."""
    if key not in self._instruments:
      return True
    with self._lock:
      self._sync()
      return key in self._watched

  def unwatched(self) -> list[str]:
    """The instruments not watched now (their readings freeze)."""
    with self._lock:
      self._sync()
      return [k for k in self._instruments if k not in self._watched]

  def watched(self) -> list[str]:
    with self._lock:
      self._sync()
      return [k for k in self._instruments if k in self._watched]

  def options(self) -> dict[str, tuple[str, str]]:
    """Option text -> (instrument to watch, instrument to stop watching)."""
    with self._lock:
      self._sync()
      unwatched = [k for k in self._instruments if k not in self._watched]
      return {
          f'watch the {new.lower()} instead of the {old.lower()}': (new, old)
          for new in unwatched
          for old in self._watched
      }

  def swap(self, new: str, old: str) -> bool:
    """Watches `new` in place of `old`; False if that is not possible."""
    with self._lock:
      self._sync()
      if new in self._watched or old not in self._watched:
        return False
      if new not in self._instruments:
        return False
      self._watched[self._watched.index(old)] = new
      self.refocused += 1
      return True

  def get_state(self) -> dict[str, Any]:
    with self._lock:
      return {'watched': list(self._watched), 'refocused': self.refocused}

  def set_state(self, state: Mapping[str, Any]) -> None:
    with self._lock:
      self._watched = [
          k for k in state.get('watched', []) if k in self._instruments
      ]
      self.refocused = int(state.get('refocused', 0))

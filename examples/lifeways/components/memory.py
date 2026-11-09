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

"""A memory that keeps only its latest entries, for an entity that never dies.

A person's memory is bounded by its lifetime. The Experimenter lives as long
as the world, which may be for ever: it remembers only what it needs to act,
the latest few observations.
"""

from collections.abc import Mapping
from typing import Any, override

from concordia.components.agent import memory


class RecentMemory(memory.ListMemory):
  """The stock list memory, keeping only its latest `kept` entries."""

  def __init__(self, kept: int = 32):
    """Initializes the memory.

    Args:
      kept: How many entries to keep (the oldest go first).
    """
    super().__init__(memory_bank=[])
    self._kept = int(kept)

  def _trim(self) -> None:
    with self._lock:
      del self._memory_bank[: max(0, len(self._memory_bank) - self._kept)]

  @override
  def update(self) -> None:
    super().update()
    self._trim()

  @override
  def set_state(self, state: Mapping[str, Any]) -> None:
    super().set_state(state)
    self._trim()

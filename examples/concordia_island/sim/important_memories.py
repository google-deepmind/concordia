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

"""ImportantMemories observation component.

Replaces the naive LastNObservations with a three-tier memory retrieval
strategy that prevents persona amnesia at scale:

1. **Formative memories** — Always includes memories tagged [self] and
   [background] (persona foundation, ~5-20 entries).
2. **Journal reflections** — Always includes memories tagged [journal]
   (end-of-day reflections from JournalReflection component).
3. **Recent observations** — Last N observations tagged [observation]
   (bounded sliding window).

This ensures that even after hundreds of ticks, agents retain their
founding identity and accumulated wisdom while staying grounded in
recent events.
"""

from collections.abc import Sequence

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state


DEFAULT_IMPORTANT_MEMORIES_KEY = 'ImportantMemories'

# Tags used by Concordia to mark different memory types
FORMATIVE_TAGS = ('[self]', '[background]', '[formative]')
JOURNAL_TAG = '[journal]'
OBSERVATION_TAG = observation.OBSERVATION_TAG  # '[observation]'


class ImportantMemories(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Observation component that always preserves formative memories.

  Instead of a naive LastN window that loses persona memories once the
  agent has accumulated enough observations, this component constructs a
  context window from three tiers:

  1. Hardcoded formative memories passed at construction
  2. All [self] and [background] tagged memories from memory bank
  3. All [journal] tagged memories (periodic reflections)
  4. Last N [observation] tagged recent memories

  The output is ordered: hardcoded → bank formative → journals → recent,
  with deduplication to avoid repeating entries.
  """

  def __init__(
      self,
      recent_history_length: int = 50,
      formative_memories: Sequence[str] = (),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = '\nRecent events',
  ):
    """Initializes the ImportantMemories component.

    Args:
      recent_history_length: Maximum number of recent [observation] tagged
        memories to include in the sliding window.
      formative_memories: Hardcoded formative memories to always include, even
        if not found in the memory bank.
      memory_component_key: Key for the memory component to read from.
      pre_act_label: Label prefix for pre_act output.
    """
    super().__init__(pre_act_label)
    self._memory_component_key = memory_component_key
    self._recent_history_length = recent_history_length
    self._formative_memories = tuple(formative_memories)

  def _make_pre_act_value(self) -> str:
    """Returns formative + journal + recent memories as context."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    # Tier 1: Formative memories from bank
    formative = memory.scan(lambda m: any(tag in m for tag in FORMATIVE_TAGS))

    # Tier 2: Journal reflections
    journals = memory.scan(lambda m: JOURNAL_TAG in m)

    # Tier 3: Recent observations (bounded sliding window)
    recent = memory.retrieve_recent(limit=self._recent_history_length)
    recent_obs = [m for m in recent if OBSERVATION_TAG in m]

    # Combine with deduplication, preserving order
    seen = set()
    result = []

    for mem in (
        list(self._formative_memories)
        + list(formative)
        + list(journals)
        + list(recent_obs)
    ):
      if mem not in seen:
        seen.add(mem)
        result.append(mem)

    output = '\n'.join(result) + '\n'

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': output.splitlines(),
        'formative_count': len(formative),
        'journal_count': len(journals),
        'recent_obs_count': len(recent_obs),
        'total': len(result),
    })

    return output

  def get_state(self) -> entity_component.ComponentState:
    return {
        'memory_component_key': self._memory_component_key,
        'recent_history_length': self._recent_history_length,
        'formative_memories': list(self._formative_memories),
        'pre_act_label': self.get_pre_act_label(),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'memory_component_key' in state:
      self._memory_component_key = component_state.as_str(
          state, 'memory_component_key'
      )
    if 'recent_history_length' in state:
      self._recent_history_length = component_state.as_int(
          state, 'recent_history_length', self._recent_history_length
      )
    if 'formative_memories' in state:
      self._formative_memories = tuple(
          component_state.as_str_list(state, 'formative_memories')
      )

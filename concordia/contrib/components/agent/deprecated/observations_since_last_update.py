# Copyright 2024 DeepMind Technologies Limited.
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

"""A component for tracking observations since the last update.
"""

from collections.abc import Callable
import datetime

from absl import logging as absl_logging
from concordia.components.agent import deprecated as agent_components
from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging


def _get_earliest_timepoint(
    memory_component_: agent_components.memory_component.MemoryComponent,
) -> datetime.datetime:
  """Returns all memories in the memory bank.

  Args:
    memory_component_: The memory component to retrieve memories from.
  """
  memories_data_frame = memory_component_.get_raw_memory()
  if not memories_data_frame.empty:
    sorted_memories_data_frame = memories_data_frame.sort_values(
        'time', ascending=True)
    return sorted_memories_data_frame['time'][0]
  else:
    absl_logging.warning('No memories found in memory bank.')
    return datetime.datetime.now()


class ObservationsSinceLastUpdate(action_spec_ignored.ActionSpecIgnored):
  """Report all observations since the last update."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      pre_act_key: str = '\nObservations',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to consider the latest observations.

    Args:
      model: The language model to use.
      clock_now: Function that returns the current time.
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._memory_component_name = memory_component_name
    self._logging_channel = logging_channel

    self._previous_time = None

  def pre_observe(
      self,
      observation: str,
  ) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[observation] {observation}',
        metadata={'tags': ['observation']},
    )
    return ''

  def _make_pre_act_value(self) -> str:
    """Returns a representation of the current situation to pre act."""
    current_time = self._clock_now()
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    if self._previous_time is None:
      self._previous_time = _get_earliest_timepoint(memory)

    interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
        time_from=self._previous_time,
        time_until=current_time,
        add_time=True,
    )
    mems = [mem.text for mem in memory.retrieve(scoring_fn=interval_scorer)]
    result = '\n'.join(mems) + '\n'

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
    })

    self._previous_time = current_time

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      if self._previous_time is None:
        previous_time = ''
      else:
        previous_time = self._previous_time.strftime('%Y-%m-%d %H:%M:%S')
      return {
          'previous_time': previous_time,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    with self._lock:
      if state['previous_time']:
        previous_time = datetime.datetime.strptime(
            state['previous_time'], '%Y-%m-%d %H:%M:%S')
      else:
        previous_time = None
      self._previous_time = previous_time

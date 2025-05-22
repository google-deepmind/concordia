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

"""A component for representing the current situation via narrative.
"""

from collections.abc import Callable, Mapping, Sequence
import datetime
import types

from absl import logging as absl_logging
from concordia.components.agent import deprecated as agent_components
from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging
from concordia.typing.deprecated import memory as memory_lib


def _get_all_memories(
    memory_component_: agent_components.memory_component.MemoryComponent,
    add_time: bool = True,
    sort_by_time: bool = True,
    constant_score: float = 0.0,
) -> Sequence[memory_lib.MemoryResult]:
  """Returns all memories in the memory bank.

  Args:
    memory_component_: The memory component to retrieve memories from.
    add_time: whether to add time
    sort_by_time: whether to sort by time
    constant_score: assign this score value to each memory
  """
  texts = memory_component_.get_all_memories_as_text(add_time=add_time,
                                                     sort_by_time=sort_by_time)
  return [memory_lib.MemoryResult(text=t, score=constant_score) for t in texts]


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


class SituationRepresentation(action_spec_ignored.ActionSpecIgnored):
  """Consider ``what kind of situation am I in now?``."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      declare_entity_as_protagonist: bool = True,
      pre_act_key: str = 'The current situation',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to consider the current situation.

    Args:
      model: The language model to use.
      clock_now: Function that returns the current time.
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      components: Components to condition the narrative on. This is a mapping
        of the component name to a label to use in the prompt.
      declare_entity_as_protagonist: Whether to declare the entity to be the
        protagonist in the prompt.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._declare_entity_as_protagonist = declare_entity_as_protagonist
    self._logging_channel = logging_channel

    self._previous_time = None
    self._situation_thus_far = None

  def _add_components_if_any(
      self,
      chain_of_thought: interactive_document.InteractiveDocument,
  ) -> None:
    """Adds components to the chain of thought if any are present."""
    if self._components:
      component_states = '\n'.join([
          f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      chain_of_thought.statement(f'Considerations:\n{component_states}\n')

  def _make_pre_act_value(self) -> str:
    """Returns a representation of the current situation to pre act."""
    agent_name = self.get_entity().name
    current_time = self._clock_now()
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    initial_step_thought_chain = ''
    if self._situation_thus_far is None:
      self._previous_time = _get_earliest_timepoint(memory)
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement('~~ Creative Writing Assignment ~~')
      if self._declare_entity_as_protagonist:
        chain_of_thought.statement(f'Protagonist: {agent_name}')
      mems = '\n'.join([mem.text for mem in _get_all_memories(memory)])
      chain_of_thought.statement(f'Story fragments and world data:\n{mems}')
      self._add_components_if_any(chain_of_thought)
      chain_of_thought.statement(f'Events continue after {current_time}')
      self._situation_thus_far = chain_of_thought.open_question(
          question=(
              'Narratively summarize the story fragments and world data. Give '
              'special emphasis to atypical features of the setting such as '
              'when and where the story takes place as well as any causal '
              'mechanisms or affordances mentioned in the information '
              'provided. Highlight the goals, personalities, occupations, '
              'skills, and affordances of the named characters and '
              'relationships between them. If any specific numbers were '
              'mentioned then make sure to include them. Use third-person '
              'omniscient perspective.'),
          max_tokens=1000,
          terminators=(),
          question_label='Exercise')
      initial_step_thought_chain = '\n'.join(
          chain_of_thought.view().text().splitlines())

    interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
        time_from=self._previous_time,
        time_until=current_time,
        add_time=True,
    )
    mems = [mem.text for mem in memory.retrieve(scoring_fn=interval_scorer)]
    result = '\n'.join(mems) + '\n'
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(f'Context:\n{self._situation_thus_far}')
    self._add_components_if_any(chain_of_thought)
    if self._declare_entity_as_protagonist:
      chain_of_thought.statement(f'Protagonist: {agent_name}')
      chain_of_thought.statement(
          f'Thoughts and memories of {agent_name}:\n{result}'
      )
    else:
      chain_of_thought.statement(f'Notes:\n{result}')
    self._situation_thus_far = chain_of_thought.open_question(
        question=(
            'What situation does the protagonist find themselves in? '
            'Make sure to provide enough detail to give the '
            'reader a comprehensive understanding of the world '
            'inhabited by the protagonist, their affordances in that '
            'world, actions they may be able to take, effects their '
            'actions may produce, and what is currently going on. If any '
            'specific numbers were mentioned then make sure to include them.'
            'Also, make sure to repeat all details of the context that could '
            'ever be relevant, now or in the future.'
        ),
        max_tokens=1000,
        terminators=(),
        question_label='Exercise',
    )
    chain_of_thought.statement(f'The current date and time is {current_time}')

    chain_of_thought_text = '\n'.join(
        chain_of_thought.view().text().splitlines())

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': self._situation_thus_far,
        'Chain of thought': (initial_step_thought_chain +
                             '\n***\n' +
                             chain_of_thought_text),
    })

    self._previous_time = current_time

    return self._situation_thus_far

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      if self._previous_time is None:
        previous_time = ''
      else:
        previous_time = self._previous_time.strftime('%Y-%m-%d %H:%M:%S')
      return {
          'previous_time': previous_time,
          'situation_thus_far': self._situation_thus_far,
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
      self._situation_thus_far = state['situation_thus_far']

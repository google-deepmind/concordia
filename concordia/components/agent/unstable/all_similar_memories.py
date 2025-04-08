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

"""Return all memories similar to a prompt and filter them for relevance."""

import types
from typing import Mapping

from concordia.components.agent.unstable import action_spec_ignored
from concordia.components.agent.unstable import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import logging
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import entity_component


class AllSimilarMemories(action_spec_ignored.ActionSpecIgnored):
  """Get all memories similar to the state of the components and filter them."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      num_memories_to_retrieve: int = 25,
      pre_act_label: str = 'Relevant memories',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to report relevant memories (similar to a prompt).

    Args:
      model: The language model to use.
      memory_component_key: The name of the memory component from which to
        retrieve related memories.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      num_memories_to_retrieve: The number of memories to retrieve.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = dict(components)
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(f'Statements:\n{component_states}\n')
    prompt_summary = prompt.open_question(
        'Summarize the statements above.', max_tokens=750
    )

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.AssociativeMemory
    )

    query = f'{agent_name}, {prompt_summary}'
    result = '\n'.join([
        mem
        for mem in memory.retrieve_associative(
            query=query, limit=self._num_memories_to_retrieve
        )
    ])

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
        'Query': f'{query}',
    })

    return result + '\n'


class AllSimilarMemoriesWithoutPreAct(
    action_spec_ignored.ActionSpecIgnored
):
  """An AllSimilarMemories component that does not output its state to pre_act.
  """

  def __init__(self, *args, **kwargs):
    self._component = AllSimilarMemories(*args, **kwargs)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    self._component.set_entity(entity)

  def _make_pre_act_value(self) -> str:
    return ''

  def get_pre_act_value(self) -> str:
    return self._component.get_pre_act_value()

  def get_pre_act_label(self) -> str:
    return self._component.get_pre_act_label()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()

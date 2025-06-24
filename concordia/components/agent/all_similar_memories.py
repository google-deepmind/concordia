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

"""Return memories similar to a prompt composed of component pre_act values."""

from collections.abc import Sequence

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class AllSimilarMemories(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """Get all memories similar to the state of the components and filter them."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      components: Sequence[str] = (),
      num_memories_to_retrieve: int = 25,
      pre_act_label: str = 'Relevant memories',
  ):
    """Initialize a component to report relevant memories (similar to a prompt).

    Args:
      model: The language model to use.
      memory_component_key: The name of the memory component from which to
        retrieve related memories.
      components: Keys of components to condition the answer on.
      num_memories_to_retrieve: The number of memories to retrieve.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = components
    self._num_memories_to_retrieve = num_memories_to_retrieve

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}')

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
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

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    with self._lock:
      return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    with self._lock:
      pass


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

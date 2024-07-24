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

"""Agent identity component."""

from collections.abc import Sequence

from concordia.components.agent.v2 import action_spec_ignored
from concordia.components.agent.v2 import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import concurrency


DEFAULT_PRE_ACT_KEY = 'Identity characteristics'
_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


class Identity(action_spec_ignored.ActionSpecIgnored):
  """Identity component containing a few characteristics.

  Identity is built out of individual characteristic queries to memory. For
  example, they could be:
  1. 'core characteristics',
  2. 'current daily occupation',
  3. 'feeling about recent progress in life',
  """

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
      queries: Sequence[str] = (
          'core characteristics',
          'current daily occupation',
          'feeling about recent progress in life',
      ),
      num_memories_to_retrieve: int = 25,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize an identity component.

    Args:
      model: a language model
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      queries: strings to use as queries to the associative memory
      num_memories_to_retrieve: how many related memories to retrieve per query
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name

    self._queries = queries
    self._num_memories_to_retrieve = num_memories_to_retrieve

    self._logging_channel = logging_channel

  def _query_memory(self, query: str) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    name_with_query = f"{agent_name}'s {query}"
    mems = '\n'.join(
        [mem.text for mem in memory.retrieve(
            query=name_with_query,
            scoring_fn=_ASSOCIATIVE_RETRIEVAL,
            limit=self._num_memories_to_retrieve)]
    )
    prompt = interactive_document.InteractiveDocument(self._model)
    question = (
        f"How would one describe {agent_name}'s"
        f' {query} given the following statements? '
    )
    result = prompt.open_question(
        '\n'.join([question, f'Statements:\n{mems}']),
        max_tokens=1000,
        answer_prefix=f'{agent_name} is ',
    )
    return result

  def _make_pre_act_value(self) -> str:
    results = concurrency.map_parallel(self._query_memory, self._queries)
    output = '\n'.join(
        [f'{query}: {result}' for query, result in zip(self._queries, results)]
    )

    self._logging_channel({'Key': self.get_pre_act_key(), 'Value': output})

    return output


class IdentityWithoutPreAct(action_spec_ignored.ActionSpecIgnored):
  """An identity component that does not output its state to pre_act."""

  def __init__(self, *args, **kwargs):
    self._component = Identity(*args, **kwargs)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    self._component.set_entity(entity)

  def _make_pre_act_value(self) -> str:
    return ''

  def get_pre_act_value(self) -> str:
    return self._component.get_pre_act_value()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()

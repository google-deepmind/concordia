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
from typing import Sequence
from concordia.associative_memory import associative_memory
from concordia.components.agent.v2 import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component_v2
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency

import termcolor


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
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      queries: Sequence[str] = (
          'core characteristics',
          'current daily occupation',
          'feeling about recent progress in life',
      ),
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
      log_color: str = 'green',
  ):
    """Initialize an identity component.

    Args:
      model: a language model
      memory: an associative memory
      queries: strings to use as queries to the associative memory
      num_memories_to_retrieve: how many related memories to retrieve per query
      verbose: whether or not to print the result for debugging
      log_color: color to print the debug log
    """
    self._model = model
    self._memory = memory
    self._last_log = None

    self._queries = queries
    self._num_memories_to_retrieve = num_memories_to_retrieve

    self._verbose = verbose
    self._log_color = log_color

  def _query_memory(self, query: str) -> str:
    agent_name = self.get_entity().name
    name_with_query = f"{agent_name}'s {query}"
    mems = '\n'.join(
        self._memory.retrieve_associative(
            name_with_query, self._num_memories_to_retrieve, add_time=True
        )
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

  def make_pre_act_context(self) -> str:
    results = concurrency.map_parallel(self._query_memory, self._queries)
    output = '\n'.join(
        [f'{query}: {result}' for query, result in zip(self._queries, results)]
    )

    self._last_log = {
        'State': output,
    }
    if self._verbose:
      self._log(output)

    return output

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def get_last_log(self):
    if self._last_log:
      return self._last_log.copy()


class IdentityWithoutPreAct(component_v2.EntityComponent):
  """An identity component that does not output its state to pre_act."""

  def __init__(self, *args, **kwargs):
    self._component = Identity(*args, **kwargs)

  def set_entity(self, entity: component_v2.ComponentEntity) -> None:
    self._component.set_entity(entity)

  def get_pre_act_context(self) -> str:
    return self._component.get_pre_act_context()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()

  def get_last_log(self):
    return self._component.get_last_log()

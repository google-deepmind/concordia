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

"""Return all memories similar to a prompt and filter them for relevance.
"""

from collections.abc import Mapping
import types

from concordia.components.agent.v2 import action_spec_ignored
from concordia.components.agent.v2 import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity as entity_lib
import termcolor


_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


class AllSimilarMemories(action_spec_ignored.ActionSpecIgnored):
  """Get all memories similar to the state of the components and filter them."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
      components: Mapping[str, action_spec_ignored.ActionSpecIgnored] = (
          types.MappingProxyType({})
      ),
      num_memories_to_retrieve: int = 25,
      pre_act_label: str = 'Relevant memories',
      verbose: bool = False,
  ):
    """Initialize a component to report relevant memories (similar to a prompt).

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      components: The components to condition the answer on.
      num_memories_to_retrieve: The number of memories to retrieve.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      verbose: Whether to print the state of the component.
    """
    self._verbose = verbose
    self._model = model
    self._memory_component_name = memory_component_name
    self._state = ''
    self._components = dict(components)
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._pre_act_label = pre_act_label
    self._last_log = None

  def _make_pre_act_context(self) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_context(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(f'Statements:\n{component_states}\n')
    prompt_summary = prompt.open_question(
        'Summarize the statements above.', max_tokens=750
    )

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    query = f'{agent_name}, {prompt_summary}'
    mems = '\n'.join(
        [mem.text for mem in memory.retrieve(
            query=query,
            scoring_fn=_ASSOCIATIVE_RETRIEVAL,
            limit=self._num_memories_to_retrieve)]
    )

    question = (
        'Select the subset of the following set of statements that is most '
        f'important for {agent_name} to consider right now. Whenever two '
        'or more statements are not mutally consistent with each other '
        'select whichever statement is more recent. Repeat all the '
        'selected statements verbatim. Do not summarize. Include timestamps. '
        'When in doubt, err on the side of including more, especially for '
        'recent events. As long as they are not inconsistent, revent events '
        'are usually important to consider.'
    )
    new_prompt = prompt.new()
    result = new_prompt.open_question(
        f'{question}\nStatements:\n{mems}',
        max_tokens=2000,
        terminators=(),
    )

    if self._verbose:
      print(termcolor.colored(prompt.view().text(), 'green'), end='')
      print(termcolor.colored(f'Query: {query}\n', 'green'), end='')
      print(termcolor.colored(new_prompt.view().text(), 'green'), end='')
      print(termcolor.colored(result, 'green'), end='')

    self._last_log = {
        'State': result,
        'Initial chain of thought': prompt.view().text().splitlines(),
        'Query': f'{query}',
        'Final chain of thought': new_prompt.view().text().splitlines(),
    }

    return result

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    context = super().pre_act(action_spec)
    return  f'{self._pre_act_label}: {context}'

  def get_last_log(self):
    if self._last_log:
      return self._last_log.copy()

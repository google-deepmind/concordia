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

from concordia.associative_memory import associative_memory
from concordia.components.agent.v2 import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component_v2
import termcolor


EMPTY_MAPPING = component_v2.EMPTY_MAPPING


class AllSimilarMemories(action_spec_ignored.ActionSpecIgnored):
  """Get all memories similar to the state of the components and filter them."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      components: Mapping[
          str, action_spec_ignored.ActionSpecIgnored] = EMPTY_MAPPING,
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
  ):
    """Initialize a component to report relevant memories (similar to a prompt).

    Args:
      model: The language model to use.
      memory: The memory to use.
      components: The components to condition the answer on.
      num_memories_to_retrieve: The number of memories to retrieve.
      verbose: Whether to print the state of the component.
    """

    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._components = dict(components)
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._last_log = None

  def make_pre_act_context(self) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f"{agent_name}'s {key}:\n{component.get_pre_act_context()}"
        for key, component in self._components.items()
    ])
    prompt.statement(f'Statements:\n{component_states}\n')
    prompt_summary = prompt.open_question(
        'Summarize the statements above.', max_tokens=750
    )

    query = f'{agent_name}, {prompt_summary}'
    mems = '\n'.join(
        self._memory.retrieve_associative(
            query, self._num_memories_to_retrieve, add_time=True
        )
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

  def get_last_log(self):
    if self._last_log:
      return self._last_log.copy()

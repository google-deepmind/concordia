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

"""Agent component for representing what kind of person the agent is."""
from typing import Mapping

from concordia.associative_memory import associative_memory
from concordia.components.agent.v2 import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component_v2

import termcolor

EMPTY_MAPPING = component_v2.EMPTY_MAPPING


class SelfPerception(action_spec_ignored.ActionSpecIgnored):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      components: Mapping[
          str, action_spec_ignored.ActionSpecIgnored] = EMPTY_MAPPING,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
      log_color: str = 'green',
  ):
    """Initializes the SelfPerception component.

    Args:
      model: Language model.
      memory: Associative memory.
      components: The components to condition the answer on.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state or not for debugging.
      log_color: color to print the debug log.
    """
    self._model = model
    self._memory = memory
    self._components = dict(components)
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._queued_memory_additions = []

    self._verbose = verbose
    self._log_color = log_color
    self._last_log = None

  def make_pre_act_context(self) -> str:
    agent_name = self.get_entity().name

    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Memories of {agent_name}:\n{mems}')

    component_states = '\n'.join([
        f"{agent_name}'s {key}:\n{component.get_pre_act_context()}"
        for key, component in self._components.items()
    ])
    prompt.statement(component_states)

    question = (
        f'Given the above, what kind of person is {agent_name}?'
    )
    result = prompt.open_question(
        question,
        answer_prefix=f'{agent_name} is ',
        max_tokens=1000,
    )
    result = f'{agent_name} is {result}'

    # In the future this will use a new style of memory bank that automatically
    # adds queued items on the following timestep.
    self._queued_memory_additions.append(f'[self reflection] {result}')

    self._last_chain = prompt
    self._last_log = {
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    if self._verbose:
      self._log(self._last_chain.view().text())

    return result

  def post_observe(self) -> str:
    # In the future we will use a new style of memory bank which will have
    # functionality to automatically queue additions to be added on the next
    # timetep.
    if self._queued_memory_additions:
      for memory_item in self._queued_memory_additions:
        self._memory.add(memory_item)

    return ''

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def get_last_log(self):
    if self._last_log:
      return self._last_log.copy()

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

"""Agent component for situation perception."""
from collections.abc import Callable, Mapping
import datetime
import types

from concordia.components.agent.v2 import action_spec_ignored
from concordia.components.agent.v2 import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity as entity_lib
import termcolor

DEFAULT_PRE_ACT_LABEL = 'What situation they are in'


class SituationPerception(action_spec_ignored.ActionSpecIgnored):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
      components: Mapping[str, action_spec_ignored.ActionSpecIgnored] = (
          types.MappingProxyType({})
      ),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
      verbose: bool = False,
      log_color: str = 'green',
  ):
    """Initializes the component.

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve recent memories.
      components: The components to condition the answer on.
      clock_now: time callback to use.
      num_memories_to_retrieve: The number of recent memories to retrieve.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      verbose: Whether to print intermediate reasoning.
      log_color: color to print the debug log.
    """
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._pre_act_label = pre_act_label

    self._verbose = verbose
    self._log_color = log_color
    self._last_log = None

  def _make_pre_act_context(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join(
        [mem.text for mem in memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve)]
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Memories of {agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    component_states = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_context(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)

    question = (
        'Given the statements above, what kind of situation is'
        f' {agent_name} in right now?'
    )
    result = prompt.open_question(
        question,
        answer_prefix=f'{agent_name} is currently ',
        max_tokens=1000,
    )
    result = f'{agent_name} is currently {result}'

    self._last_log = {
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    if self._verbose:
      print(termcolor.colored(prompt.view().text(), 'green'), end='')

    return result

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    context = super().pre_act(action_spec)
    return  f'{self._pre_act_label}: {context}'

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def get_last_log(self):
    if self._last_log:
      return self._last_log.copy()

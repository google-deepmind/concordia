# Copyright 2026 DeepMind Technologies Limited.
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

"""A component for maintaining a working memory narrative of the simulation."""

from collections.abc import Sequence
import re

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component


DEFAULT_PRE_ACT_LABEL = 'GM Working Memory'


class GMWorkingMemory(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A stateful component that maintains a narrative summary of the simulation.

  This is for the Game Master to track the overall story state.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      components: Sequence[str] = (),
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
      num_memories_to_retrieve: int = 100,
      verbose: bool = True,
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = components
    self._num_memories = num_memories_to_retrieve
    self._verbose = verbose
    self._working_memory_narrative = ''

  def _extract_time_from_memories(self, memories: list[str]) -> str:
    if not memories:
      return 'Unknown'
    latest_memory = memories[-1] if memories else ''
    time_patterns = [
        r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\]',
        r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)',
        r'at (\d{1,2}:\d{2})',
        r'(\d{4}-\d{2}-\d{2})',
    ]
    for pattern in time_patterns:
      match = re.search(pattern, latest_memory)
      if match:
        return match.group(1)
    return 'Unknown (infer from context)'

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    mems = list(memory.retrieve_recent(limit=self._num_memories))
    current_time = self._extract_time_from_memories(mems)
    mems_text = '\n'.join(mems)
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      component = self.get_entity().get_component(
          key, type_=action_spec_ignored.ActionSpecIgnored
      )
      value = self.get_named_component_pre_act_value(key)
      prompt.statement(f'{component.get_pre_act_label()}: {value}')
    prompt.statement(f'Recent memories (with timestamps):\n{mems_text}')
    prompt.statement(f'Current time: {current_time}')
    if self._working_memory_narrative:
      prompt.statement(
          f'Previous working memory:\n{self._working_memory_narrative}'
      )
    question = """You are the Game Master. Update your working memory - a running narrative of the simulation state.

1. **Overall story** (3-4 paragraphs): Current situation, key events, where characters are and what they're doing.
2. **Key uncertainties**: Unresolved elements, what characters are trying to find out.

Write in third person. Be specific and grounded in memories. Keep it concise (500-700 words)."""
    result = prompt.open_question(
        question=question,
        answer_prefix='Game Master\'s working memory: ',
        max_tokens=2000,
        terminators=(),
    )
    result = 'Game Master\'s working memory: ' + result
    self._working_memory_narrative = result
    if self._verbose:
      print(f'GM Working Memory:\n{result}')
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Summary': 'GM Working Memory update',
        'State': result,
    })
    return result

  def get_narrative(self) -> str:
    return self._working_memory_narrative

  def get_state(self) -> entity_component.ComponentState:
    return {'working_memory_narrative': self._working_memory_narrative}

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._working_memory_narrative = str(
        state.get('working_memory_narrative', '')
    )

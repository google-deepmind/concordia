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

"""Agent thinks about how to justify their recent voluntary actions."""

import datetime
from typing import Callable
from typing import Sequence

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


DEFAULT_AUDIENCES = (
    'themself',
    'their peers',
    'their superiors',
    'their subordinates',
    'their children',
    'their clan',
    'their spouse',
    'their friends',
    'religious people',
    'atheists',
    'strangers',
    'the poor',
    'the rich',
    'a court of law',
    'god',
)


def concat_interactive_documents(
    doc_a: interactive_document.InteractiveDocument,
    doc_b: interactive_document.InteractiveDocument,
) -> interactive_document.InteractiveDocument:
  """Concatenates two interactive documents. Returns a copy."""
  copied_doc = doc_a.copy()
  copied_doc.extend(doc_b.contents())
  return copied_doc


class JustifyRecentVoluntaryActions(component.Component):
  """Make new thoughts concerning justification of recent voluntary actions."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: list[component.Component],
      audiences: Sequence[str] = DEFAULT_AUDIENCES,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the JustifyRecentVoluntaryActions component.

    Args:
      name: The name of the component.
      model: The language model to use.
      memory: The memory to use.
      agent_name: The name of the agent.
      components: 
      audiences:
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: The number of memories to retrieve.
      verbose: Whether to print the state of the component.
    """
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._components = components
    self._audiences = audiences

    self._name = name
    self._history = []
    self._last_update = datetime.datetime.min

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_components(self) -> Sequence[component.Component]:
    return self._components

  def update(self) -> None:
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()

    # First determine what voluntary actions the agent recently took.
    what_they_did_chain_of_thought = interactive_document.InteractiveDocument(
        self._model)
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )
    what_they_did_chain_of_thought.statement(
        f'Memories of {self._agent_name}:\n{mems}')
    what_they_did_chain_of_thought.statement(
        f'The current time: {self._clock_now()}.')
    what_they_did = what_they_did_chain_of_thought.open_question(
        question=(
            f"Summarize the gist of {self._agent_name}'s most recent "
            + 'voluntary actions. Do not speculate about their motives. '
            + 'Just straightforwardly describe what they did most recently.'
        ),
        max_tokens=1000,
        terminators=(),
    )
    what_effect_it_had = what_they_did_chain_of_thought.open_question(
        question=(
            f"If any, what consequences did {self._agent_name}'s "
            + 'most recent voluntary actions have? Only consider effects '
            + f'that have already occurred (before {self._clock_now()}).'
        ),
        max_tokens=1000,
        terminators=(),
    )
    # Now consider how to justify the voluntary actions for all audiences.
    justification_chain_of_thought = interactive_document.InteractiveDocument(
        self._model)
    component_states = '\n'.join([
        f"{self._agent_name}'s "
        + (comp.name() + ':\n' + comp.state())
        for comp in self._components
    ])
    justification_chain_of_thought.statement(component_states)
    justification_chain_of_thought.statement(
        f'The current time: {self._clock_now()}.')
    justification_chain_of_thought.statement(
        f'{self._agent_name}\'s latest voluntary action: {what_they_did}')
    justification_chain_of_thought.statement(
        f'The effect of {self._agent_name}\'s voluntary action (if any): ' +
        f'{what_effect_it_had}')
    audiences_str = ', '.join(self._audiences[:-1])
    audiences_str += f', and {self._audiences[-1]}'
    _ = justification_chain_of_thought.open_question(
        question=(
            f'How would {self._agent_name} justify their actions to all the '
            + f'following audiences: {audiences_str}?'
        ),
        max_tokens=2000,
        terminators=(),
    )
    most_salient_justification = justification_chain_of_thought.open_question(
        question=(
            f"Given {self._agent_name}'s current situation, which "
            + 'justification is most salient to them? Describe the action '
            + 'itself, as well as some reasons why, and to whom, it can be '
            + 'justified. Feel free to blend justifications crafted for '
            + 'different audiences.'
        ),
        answer_prefix=f'{self._agent_name} ',
        max_tokens=1000,
        terminators=(),
    )
    salient_justification = (
        f'[thought] {self._agent_name} {most_salient_justification}')
    self._state = salient_justification
    self._memory.add(salient_justification)

    self._last_chain = concat_interactive_documents(
        what_they_did_chain_of_thought, justification_chain_of_thought)
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
        'Chain of thought': self._last_chain.view().text().splitlines(),
    }
    self._history.append(update_log)

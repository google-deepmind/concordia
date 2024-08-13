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

"""Agent component for dialectical reflection."""
import datetime
from typing import Callable
from typing import Sequence

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


def concat_interactive_documents(
    doc_a: interactive_document.InteractiveDocument,
    doc_b: interactive_document.InteractiveDocument,
) -> interactive_document.InteractiveDocument:
  """Concatenates two interactive documents. Returns a copy."""
  copied_doc = doc_a.copy()
  copied_doc.extend(doc_b.contents())
  return copied_doc


class CreativeReflection(component.Component):
  """Make new thoughts from memories by thesis-antithesis-synthesis."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      source_of_abstraction: Sequence[component.Component],
      topic_component: component.Component,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
  ):
    """Initializes the DialecticReflection component.

    Args:
      name: The name of the component.
      model: The language model to use.
      memory: The memory to use.
      agent_name: The name of the agent.
      source_of_abstraction: Components to condition abstraction of principles.
      topic_component: Components that represents the topic of reflection.
      clock_now: time callback to use for the state.
      verbose: Whether to print the state of the component.
    """
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._source_of_abstraction = source_of_abstraction
    self._topic_component = topic_component
    self._agent_name = agent_name
    self._clock_now = clock_now

    self._components = self._source_of_abstraction
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

    old_state = self._state

    abstraction_chain = interactive_document.InteractiveDocument(self._model)
    prethoughts = '-' + '\n'.join([
        f"-{self._agent_name}'s {comp.name()}: {comp.state()}\n"
        for comp in self._source_of_abstraction
    ])

    abstraction_chain.statement(prethoughts)

    abstration = abstraction_chain.open_question(
        'From the above, what are the basic principles that guides'
        f' {self._agent_name} thinking?.',
        answer_prefix=(
            f'{self._agent_name} uses the following principles to guide its'
            ' thinking: '
        ),
        max_tokens=2000,
        terminators=(),
    )

    application_chain = interactive_document.InteractiveDocument(self._model)
    application_chain.statement(
        f'{self._agent_name} is going to use the following principles to guide'
        f' its thinking: {abstration}'
    )

    observations = application_chain.open_question(
        f'What observations would {self._agent_name} make on the following'
        f' topic, given the principles: {self._topic_component.state()}',
        max_tokens=2000,
        terminators=(),
    )

    self._state = f'{self._agent_name} just realized that {observations}'

    if old_state != self._state:
      self._memory.add(f'[idea] {observations}')

    self._last_chain = concat_interactive_documents(
        abstraction_chain, application_chain
    )
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
        'Chain of thought': self._last_chain.view().text().splitlines(),
    }
    self._history.append(update_log)

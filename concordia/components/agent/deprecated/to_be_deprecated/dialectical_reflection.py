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

from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import termcolor


def concat_interactive_documents(
    doc_a: interactive_document.InteractiveDocument,
    doc_b: interactive_document.InteractiveDocument,
) -> interactive_document.InteractiveDocument:
  """Concatenates two interactive documents. Returns a copy."""
  copied_doc = doc_a.copy()
  copied_doc.extend(doc_b.contents())
  return copied_doc


class DialecticalReflection(component.Component):
  """Make new thoughts from memories by thesis-antithesis-synthesis."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      intuition_components: Sequence[component.Component] | None = None,
      thinking_components: Sequence[component.Component] | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 5,
      topic: component.Component | None = None,
      verbose: bool = False,
  ):
    """Initializes the DialecticReflection component.

    Args:
      name: The name of the component.
      model: The language model to use.
      memory: The memory to use.
      agent_name: The name of the agent.
      intuition_components: Components to condition thesis generation.
      thinking_components: Components to condition synthesis of thesis and
        antithesis.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: The number of memories to retrieve.
      topic: a component to represent the topic of theoretical reflection.
      verbose: Whether to print the state of the component.
    """
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._intuition_components = intuition_components or []
    self._thinking_components = thinking_components or []
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._topic_component = topic

    self._components = self._intuition_components + self._thinking_components
    if self._topic_component:
      self._components.append(self._topic_component)

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
    # The following query looks for conversations using the fact that their
    # observations are preceded by ' -- "'.
    prethoughts = list(self._memory.retrieve_associative(
        ' -- "',
        self._num_memories_to_retrieve,
        use_recency=True,
        add_time=False
    ))
    # The following query looks for memories of reading and learning.
    prethoughts += list(self._memory.retrieve_associative(
        'book, article, read, idea, concept, study, learn, research, theory',
        k=self._num_memories_to_retrieve,
        use_recency=False,
        add_time=False,
    ))

    if self._topic_component:
      prethoughts += list(self._memory.retrieve_associative(
          self._topic_component.state(),
          k=self._num_memories_to_retrieve,
          use_recency=False,
          add_time=False,
      ))

    prethoughts = '-' + '\n-'.join(prethoughts) + '\n'

    if self._intuition_components:
      prethoughts += '-' + '\n'.join([
          f"-{self._agent_name}'s {comp.name()}: {comp.state()}\n"
          for comp in self._intuition_components
      ])

    # Apply the 'thesis->antithesis->synthesis' method to generate insight.
    thesis_chain = interactive_document.InteractiveDocument(self._model)
    thesis_chain.statement(f'* The intuition of {self._agent_name} *\n')
    thesis_chain.statement(
        (f'For {self._agent_name}, all the following statements feel ' +
         f'connected:\nStatements:\n{prethoughts}'))

    thesis_question = (
        f'In light of the information above, what may {self._agent_name} ' +
        'infer')
    if self._topic_component:
      thesis_question += f' about {self._topic_component.state()}?'
    else:
      thesis_question += '?'

    thesis = thesis_chain.open_question(
        thesis_question,
        max_tokens=1200,
        terminators=(),
    )

    synthesis_chain = interactive_document.InteractiveDocument(self._model)
    synthesis_chain.statement(f'* The mind of {self._agent_name} *\n')
    synthesis_chain.statement('\n'.join([
        f"{self._agent_name}'s {comp.name()}:\n{comp.state()}"
        for comp in self._thinking_components
    ]))
    synthesis_chain.statement(
        (f'\n{self._agent_name} is applying the dialectical mode of reasoning' +
         '.\nThis involves a thesis-antithesis-synthesis pattern of logic.'))
    _ = synthesis_chain.open_question(
        question=('Given all the information above, what thesis would '
                  f'{self._agent_name} consider next?'),
        forced_response=thesis)
    _ = synthesis_chain.open_question(
        question=(
            f'How would {self._agent_name} describe the antithesis of '
            + 'the aforementioned thesis?'
        ),
        max_tokens=2000,
        terminators=(),
    )
    _ = synthesis_chain.open_question(
        question=(
            f'How would {self._agent_name} synthesize the thesis with '
            + 'its antithesis in a novel and insightful way?'
        ),
        answer_prefix=(
            f'{self._agent_name} would think step by step, and start by '
            + 'pointing out that '
        ),
        max_tokens=2000,
        terminators=(),
    )
    synthesis = synthesis_chain.open_question(
        question=(
            f'How might {self._agent_name} summarize the synthesis '
            + 'above as a bold new argument?'
        ),
        answer_prefix=(
            f"In {self._agent_name}'s view, the full argument "
            + 'is complex but the TLDR is that '
        ),
        max_tokens=1000,
        terminators=('\n',),
    )
    synthesis = synthesis[0].lower() + synthesis[1:]
    self._state = f'{self._agent_name} just realized that {synthesis}'

    if old_state != self._state:
      self._memory.add(f'[idea] {synthesis}')

    self._last_chain = concat_interactive_documents(
        thesis_chain, synthesis_chain)
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
        'Chain of thought': self._last_chain.view().text().splitlines(),
    }
    self._history.append(update_log)

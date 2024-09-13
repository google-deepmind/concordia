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
import types
from typing import Callable, Mapping

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging

DEFAULT_PRE_ACT_KEY = '\nReflection'


def concat_interactive_documents(
    doc_a: interactive_document.InteractiveDocument,
    doc_b: interactive_document.InteractiveDocument,
) -> interactive_document.InteractiveDocument:
  """Concatenates two interactive documents. Returns a copy."""
  copied_doc = doc_a.copy()
  copied_doc.extend(doc_b.contents())
  return copied_doc


class DialecticalReflection(action_spec_ignored.ActionSpecIgnored):
  """Make new thoughts from memories by thesis-antithesis-synthesis."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
      intuition_components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      thinking_components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 5,
      topic: action_spec_ignored.ActionSpecIgnored | None = None,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the DialecticReflection component.

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve memories.
      intuition_components: Components to condition thesis generation.
      thinking_components: Components to condition synthesis of thesis and
        antithesis.
      clock_now: callback function to get the current time in the game world.
      num_memories_to_retrieve: The number of memories to retrieve.
      topic: a component to represent the topic of theoretical reflection.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._intuition_components = dict(intuition_components)
    self._thinking_components = dict(thinking_components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._topic_component = topic

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    # The following query looks for conversations using the fact that their
    # observations are preceded by ' -- "'.
    prethoughts = [
        mem.text
        for mem in memory.retrieve(
            query=' -- "',
            scoring_fn=legacy_associative_memory.RetrieveAssociative(
                use_recency=True, add_time=False
            ),
            limit=self._num_memories_to_retrieve,
        )
    ]

    # The following query looks for memories of reading and learning.
    prethoughts += [
        mem.text
        for mem in memory.retrieve(
            query=('book, article, read, idea, concept, study, learn, '
                   'research, theory'),
            scoring_fn=legacy_associative_memory.RetrieveAssociative(
                use_recency=False, add_time=False
            ),
            limit=self._num_memories_to_retrieve,
        )
    ]

    if self._topic_component is not None:
      prethoughts += [
          mem.text
          for mem in memory.retrieve(
              query=self._topic_component.get_pre_act_value(),
              scoring_fn=legacy_associative_memory.RetrieveAssociative(
                  use_recency=False, add_time=False
              ),
              limit=self._num_memories_to_retrieve,
          )
      ]

    prethoughts = '-' + '\n-'.join(prethoughts) + '\n'

    if self._intuition_components:
      prethoughts += '-' + '\n'.join([
          f"-{agent_name}'s"
          f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._intuition_components.items()
      ])

    # Apply the 'thesis->antithesis->synthesis' method to generate insight.
    thesis_chain = interactive_document.InteractiveDocument(self._model)
    thesis_chain.statement(f'* The intuition of {agent_name} *\n')
    thesis_chain.statement(
        (f'For {agent_name}, all the following statements feel ' +
         f'connected:\nStatements:\n{prethoughts}'))

    thesis_question = (
        f'In light of the information above, what may {agent_name} ' +
        'infer')
    if self._topic_component:
      thesis_question += (
          f' about {self._topic_component.get_pre_act_value()}?')
    else:
      thesis_question += '?'

    thesis = thesis_chain.open_question(
        thesis_question,
        max_tokens=1200,
        terminators=(),
    )

    synthesis_chain = interactive_document.InteractiveDocument(self._model)
    synthesis_chain.statement(f'* The mind of {agent_name} *\n')
    synthesis_chain.statement('\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._thinking_components.items()
    ]))
    synthesis_chain.statement(
        (f'\n{agent_name} is applying the dialectical mode of reasoning' +
         '.\nThis involves a thesis-antithesis-synthesis pattern of logic.'))
    _ = synthesis_chain.open_question(
        question=('Given all the information above, what thesis would '
                  f'{agent_name} consider next?'),
        forced_response=thesis)
    _ = synthesis_chain.open_question(
        question=(
            f'How would {agent_name} describe the antithesis of '
            + 'the aforementioned thesis?'
        ),
        max_tokens=2000,
        terminators=(),
    )
    _ = synthesis_chain.open_question(
        question=(
            f'How would {agent_name} synthesize the thesis with '
            + 'its antithesis in a novel and insightful way?'
        ),
        answer_prefix=(
            f'{agent_name} would think step by step, and start by '
            + 'pointing out that '
        ),
        max_tokens=2000,
        terminators=(),
    )
    synthesis = synthesis_chain.open_question(
        question=(
            f'How might {agent_name} summarize the synthesis '
            + 'above as a bold new argument?'
        ),
        answer_prefix=(
            f"In {agent_name}'s view, the full argument "
            + 'is complex but the TLDR is that '
        ),
        max_tokens=1000,
        terminators=('\n',),
    )
    synthesis = synthesis[0].lower() + synthesis[1:]
    result = f'{agent_name} just realized that {synthesis}'

    memory.add(f'[idea] {synthesis}', metadata={})

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': concat_interactive_documents(
            thesis_chain, synthesis_chain).view().text().splitlines(),
    })

    return result

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

from collections.abc import Callable, Mapping, Sequence
import datetime
import types

from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging


DEFAULT_PRE_ACT_KEY = 'Self justification'

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


class JustifyRecentVoluntaryActions(action_spec_ignored.ActionSpecIgnored):
  """Make new thoughts concerning justification of recent voluntary actions."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      audiences: Sequence[str] = DEFAULT_AUDIENCES,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the JustifyRecentVoluntaryActions component.

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve recent memories.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      audiences: Intended audiences for the justification.
      clock_now: Function that returns the current time.
      num_memories_to_retrieve: The number of memories to retrieve.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._audiences = audiences

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    # First determine what voluntary actions the agent recently took.
    what_they_did_chain_of_thought = interactive_document.InteractiveDocument(
        self._model)
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join(
        [mem.text for mem in memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve)]
    )
    what_they_did_chain_of_thought.statement(
        f'Memories of {agent_name}:\n{mems}')
    what_they_did_chain_of_thought.statement(
        f'The current time: {self._clock_now()}.')
    what_they_did = what_they_did_chain_of_thought.open_question(
        question=(
            f"Summarize the gist of {agent_name}'s most recent "
            + 'voluntary actions. Do not speculate about their motives. '
            + 'Just straightforwardly describe what they did most recently.'
        ),
        max_tokens=1000,
        terminators=(),
    )
    what_effect_it_had = what_they_did_chain_of_thought.open_question(
        question=(
            f"If any, what consequences did {agent_name}'s "
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
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    justification_chain_of_thought.statement(component_states)
    justification_chain_of_thought.statement(
        f'The current time: {self._clock_now()}.')
    justification_chain_of_thought.statement(
        f'{agent_name}\'s latest voluntary action: {what_they_did}')
    justification_chain_of_thought.statement(
        f'The effect of {agent_name}\'s voluntary action (if any): ' +
        f'{what_effect_it_had}')
    audiences_str = ', '.join(self._audiences[:-1])
    audiences_str += f', and {self._audiences[-1]}'
    _ = justification_chain_of_thought.open_question(
        question=(
            f'How would {agent_name} justify their actions to all the '
            + f'following audiences: {audiences_str}?'
        ),
        max_tokens=2000,
        terminators=(),
    )
    most_salient_justification = justification_chain_of_thought.open_question(
        question=(
            f"Given {agent_name}'s current situation, which "
            + 'justification is most salient to them? Describe the action '
            + 'itself, as well as some reasons why, and to whom, it can be '
            + 'justified. Feel free to blend justifications crafted for '
            + 'different audiences.'
        ),
        answer_prefix=f'{agent_name} ',
        max_tokens=1000,
        terminators=(),
    )
    result = (
        f'[thought] {agent_name} {most_salient_justification}')
    memory.add(result, metadata={})

    display_chain = concat_interactive_documents(
        what_they_did_chain_of_thought, justification_chain_of_thought)

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': display_chain.view().text().splitlines(),
    })

    return result

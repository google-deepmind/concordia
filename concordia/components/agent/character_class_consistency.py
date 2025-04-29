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

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import datetime
import types

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging

DEFAULT_PRE_ACT_KEY = 'Character class selection'

DEFAULT_CHARACTER_CLASSES = (
    'barbarian - mighty warriors who are powered by primal forces of the multiverse that manifest as Rage',
    'bard - expert at inspiring others, soothing hurts, disheartening foes, and creating illusions',
    'cleric - can reach out to the divine magic of the Outer Planes and channel it to bolster people and battle foes',
    'druid - call on the forces of nature, harnessing magic to heal, transform into animals, and wield elemental destruction',
    'fighter - share an unparalleled prowess with weapons and armor, and are well acquainted with death, both meting it out and defying it',
    'monk - focus their internal resevoirs of power to create extraordinary',
    'paladin - live on the front lines of the cosmic struggle, united by their oaths against the forces of annihilation',
    'ranger - hones with deadly focus and harness primal powers to protect the world from the ravages of monsters and tyrants',
    'rogue - have a knack for finding the solution to just about any problem, prioritizing subte strikes over brute strengths',
    'sorcerer - harness and channel raw, roiling power of innate magic that is stamped into their very being',
    'warlock - quest for knowledge that lies hidden in the fabric of the multiverse, piecing together arcane secrets to bolster their own power',
    'wizard - cast spells of explosive fire, arcing lightning, subtle deception, and spectacular transformations',
)


def concat_interactive_documents(
    doc_a: interactive_document.InteractiveDocument,
    doc_b: interactive_document.InteractiveDocument,
) -> interactive_document.InteractiveDocument:
  """Concatenates two interactive documents. Returns a copy."""
  copied_doc = doc_a.copy()
  copied_doc.extend(doc_b.contents())
  return copied_doc


class CharacterClassConsistency(action_spec_ignored.ActionSpecIgnored):
  """Choose the character class for a specific player and track the consistency of the
  character based on class traits plus general character traits."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      character_classes: Sequence[str] = DEFAULT_CHARACTER_CLASSES,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the SelectCharacterClass component.

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve recent memories.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      character classes: character classes from the D&D 5e player handbook.
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
    self._character_classes = character_classes

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    # Determine how agent should take an actions based on its class.
    what_class_they_are_chain_of_thought = interactive_document.InteractiveDocument(
        self._model)
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join(
        [mem.text for mem in memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve)]
    )
    what_class_they_are_chain_of_thought.statement(
        f'Memories of {agent_name}:\n{mems}')
    what_class_they_are_chain_of_thought.statement(
        f'The current time: {self._clock_now()}.')
    what_they_did = what_class_they_are_chain_of_thought.open_question(
        question=(
            f"Expand on the {agent_name}'s character "
            + 'based on the motives of someone of their character class. '
            + f"Combine this description with {agent_name}'s personality traits."
        ),
        max_tokens=1000,
        terminators=(),
    )
    what_effect_it_had = what_class_they_are_chain_of_thought.open_question(
        question=(
            f"If any, what consequences did {agent_name}'s "
            + 'character class motivations have? Only consider effects '
            + f'that have already occurred (before {self._clock_now()}).'
        ),
        max_tokens=1000,
        terminators=(),
    )
    # Now consider how to justify the voluntary actions for all audiences.
    class_consistency_chain_of_thought = interactive_document.InteractiveDocument(
        self._model)
    component_states = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    class_consistency_chain_of_thought.statement(component_states)
    class_consistency_chain_of_thought.statement(
        f'The current time: {self._clock_now()}.')
    class_consistency_chain_of_thought.statement(
        f'{agent_name}\'s latest action: {what_they_did}')
    class_consistency_chain_of_thought.statement(
        f'The effect of {agent_name}\'s action (if any): ' +
        f'{what_effect_it_had}')
    audiences_str = ', '.join(self._audiences[:-1])
    audiences_str += f', and {self._audiences[-1]}'
    _ = class_consistency_chain_of_thought.open_question(
        question=(
            f'How would {agent_name} justify their actions '
            + f'based on the features of their character class: {audiences_str}?'
        ),
        max_tokens=2000,
        terminators=(),
    )
    most_salient_justification = class_consistency_chain_of_thought.open_question(
        question=(
            f"Given {agent_name}'s current situation, which "
            + 'aspect of their character is most salient to them? Describe the action '
            + 'itself, as well as some reasons why relating to their character'
            + 'it can be justfied. Feel free to blend justifications based on different '
            + 'character aspects.'
        ),
        answer_prefix=f'{agent_name} ',
        max_tokens=1000,
        terminators=(),
    )
    result = (
        f'[thought] {agent_name} {most_salient_justification}')
    memory.add(result, metadata={})

    display_chain = concat_interactive_documents(
        what_class_they_are_chain_of_thought, class_consistency_chain_of_thought)

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': display_chain.view().text().splitlines(),
    })

    return result

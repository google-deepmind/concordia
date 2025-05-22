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

from collections.abc import Mapping, Sequence
import random
import types

from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging


class PersonRepresentation(action_spec_ignored.ActionSpecIgnored):
  """Represent other characters in the simulated world."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      additional_questions: Sequence[str] = (),
      num_memories_to_retrieve: int = 100,
      cap_number_of_detected_people: int = 10,
      pre_act_key: str = 'Person representation',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to represent other people in the simulated world.

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      additional_questions: sequence of additional questions to ask about each
        player in the simulation.
      num_memories_to_retrieve: The number of memories to retrieve.
      cap_number_of_detected_people: The maximum number of people that can be
        represented.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._additional_questions = additional_questions
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._logging_channel = logging_channel
    self._cap_number_of_detected_people = cap_number_of_detected_people

    self._names_detected = []

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join([
        mem.text
        for mem in memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve * 2
        )
    ])

    find_people_prompt = interactive_document.InteractiveDocument(self._model)
    find_people_prompt.statement(
        f'Recent observations of {agent_name}:\n{mems}')
    people_str = find_people_prompt.open_question(
        question=('Create a comma-separated list containing all the proper '
                  'names of people mentioned in the observations above. For '
                  'example if the observations mention Julie, Michael, '
                  'Bob Skinner, and Francis then produce the list '
                  '"Julie,Michael,Bob Skinner,Francis".'),
        question_label='Exercise',)
    # Ignore leading and trailing whitespace in detected names
    self._names_detected.extend(  #  pytype: disable=attribute-error
        [name.strip() for name in people_str.split(',')])
    # Prevent adding duplicates
    self._names_detected = list(set(self._names_detected))
    # Prevent adding too many names, forgetting some if there are too many
    if len(self._names_detected) > self._cap_number_of_detected_people:
      self._names_detected = random.sample(self._names_detected,
                                           self._cap_number_of_detected_people)

    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(f'Considerations:\n{component_states}\n')

    associative_scorer = legacy_associative_memory.RetrieveAssociative(
        use_recency=True,
        use_importance=True,
        add_time=True,
        sort_by_time=True,
    )

    person_respresentations = []
    prompt_copies_to_log = []
    for person_name in self._names_detected:
      if not person_name:
        continue
      if person_name == agent_name:
        continue
      query = f'{person_name}'
      memories_list = [mem.text for mem in memory.retrieve(
          query=query,
          scoring_fn=associative_scorer,
          limit=self._num_memories_to_retrieve) if person_name in mem.text]
      if not memories_list:
        continue
      new_prompt = prompt.copy()
      memories = '\n'.join(memories_list)
      new_prompt.statement(f'Observed behavior and speech of {person_name}:'
                           f'\n{memories}\n')
      question = ('Taking note of all the information above, '
                  'write a descriptive paragraph capturing the character of '
                  f'{person_name} in sufficient detail for a skilled actor to '
                  'play their role convincingly. Include personality traits,  '
                  'accents, styles of speech, conversational quirks, topics '
                  'they frequently bring up, salient or unusual beliefs, and '
                  'any other relevant details.')
      person_description = new_prompt.open_question(
          f'{question}\n',
          max_tokens=350,
          terminators=('\n\n',),
          question_label='Exercise',
          answer_prefix=f'{person_name} is ',
      )
      person_representation = f'{person_name} is {person_description}'
      for question in self._additional_questions:
        additional_result = new_prompt.open_question(
            question,
            max_tokens=200,
            terminators=('\n',),
            question_label='Exercise',
            answer_prefix=f'{person_name} is ',
        )
        person_representation = (f'{person_representation}\n    '
                                 f'{person_name} is {additional_result}')

      person_respresentations.append(person_representation + '\n***')
      prompt_copies_to_log.append(new_prompt.view().text())

    result = '\n'.join(person_respresentations)

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Name detection chain of thought': (
            find_people_prompt.view().text().splitlines()),
        'Names detected so far': self._names_detected,
        'Components chain of thought': prompt.view().text().splitlines(),
        'Full chain of thought': (
            '\n***\n'.join(prompt_copies_to_log).splitlines()),
    })

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      return {
          'names_detected': self._names_detected,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    with self._lock:
      self._names_detected = state['names_detected']

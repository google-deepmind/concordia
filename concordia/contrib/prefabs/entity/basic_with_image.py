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

"""A prefab entity that generates both text and image outputs."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import image_text_act_component
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

_DEFAULT_OBSERVATION_HISTORY_LENGTH = 1_000_000
_DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH = 25
_DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH = 5


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab entity that generates both text and image outputs in JSON."""

  description: str = (
      'An entity based on the basic prefab that produces structured JSON '
      'output with both text and image fields. Supports image_first, '
      'text_first, and choice modes for generation ordering.'
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Alice',
          'goal': '',
          'randomize_choices': True,
          'prefix_entity_name': True,
          'image_model': None,
          'image_mode': 'choice',
          'image_prompt_question': None,
          'image_from_text_question': None,
          'observation_history_length': _DEFAULT_OBSERVATION_HISTORY_LENGTH,
          'situation_perception_history_length': (
              _DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH
          ),
          'self_perception_history_length': (
              _DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH
          ),
          'person_by_situation_history_length': (
              _DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH
          ),
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    entity_name = self.params.get('name', 'Alice')
    entity_goal = self.params.get('goal', '')
    randomize_choices = self.params.get('randomize_choices', True)
    prefix_entity_name = self.params.get('prefix_entity_name', True)
    image_model_instance = self.params.get('image_model', None)
    image_mode = self.params.get('image_mode', 'choice')
    observation_history_length = self.params.get(
        'observation_history_length', _DEFAULT_OBSERVATION_HISTORY_LENGTH
    )
    situation_perception_history_length = self.params.get(
        'situation_perception_history_length',
        _DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH,
    )
    self_perception_history_length = self.params.get(
        'self_perception_history_length',
        _DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH,
    )
    person_by_situation_history_length = self.params.get(
        'person_by_situation_history_length',
        _DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH,
    )

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = 'Instructions'
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nInstructions',
    )

    observation_to_memory_key = 'Observation'
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = agent_components.observation.LastNObservations(
        history_length=observation_history_length,
        pre_act_label=(
            '\nEvents so far (ordered from least recent to most recent)'
        ),
    )

    situation_perception_key = 'SituationPerception'
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            num_memories_to_retrieve=situation_perception_history_length,
            pre_act_label=(
                f'\nQuestion: What situation is {entity_name} in right now?'
                '\nAnswer'
            ),
        )
    )
    self_perception_key = 'SelfPerception'
    self_perception = (
        agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            num_memories_to_retrieve=self_perception_history_length,
            components=[
                situation_perception_key,
            ],
            pre_act_label=(
                f'\nQuestion: What kind of person is {entity_name}?\nAnswer'
            ),
        )
    )

    person_by_situation_key = 'PersonBySituation'
    person_by_situation = agent_components.question_of_recent_memories.PersonBySituation(
        model=model,
        num_memories_to_retrieve=person_by_situation_history_length,
        components=[
            self_perception_key,
            situation_perception_key,
        ],
        pre_act_label=(
            f'\nQuestion: What would a person like {entity_name} do in '
            'a situation like this?\nAnswer'
        ),
    )

    if entity_goal:
      goal_key = 'Goal'
      overarching_goal = agent_components.constant.Constant(
          state=entity_goal, pre_act_label='\nGoal'
      )
    else:
      goal_key = None
      overarching_goal = None

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        self_perception_key: self_perception,
        situation_perception_key: situation_perception,
        person_by_situation_key: person_by_situation,
        observation_key: observation,
        memory_key: memory,
    }

    component_order = list(components_of_agent.keys())

    if overarching_goal is not None:
      components_of_agent[goal_key] = overarching_goal
      component_order.insert(1, goal_key)

    image_prompt_kwargs = {}
    ipq = self.params.get('image_prompt_question', None)
    if ipq:
      image_prompt_kwargs['image_prompt_question'] = ipq
    iftq = self.params.get('image_from_text_question', None)
    if iftq:
      image_prompt_kwargs['image_from_text_question'] = iftq

    act_component = image_text_act_component.ImageTextActComponent(
        model=model,
        image_model=image_model_instance,
        image_mode=image_mode,
        component_order=component_order,
        randomize_choices=randomize_choices,
        prefix_entity_name=prefix_entity_name,
        **image_prompt_kwargs,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent

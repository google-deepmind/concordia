# Copyright 2024 DeepMind Technologies Limited.
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

"""A prefab for a 'rational' agent (logic of consequence).

This is a deliberate like-for-like counterpart to ``basic.py`` (the March &
Olsen logic of appropriateness actor). The two prefabs share the same
infrastructure -- same memory handling, same observation history, the same goal
handling, the same constants, and the same act component configuration -- and
differ only in their decision chain:

  - basic:    situation -> self -> "what would a person like me do?"
  - rational: situation -> available options -> "which option is best?"
"""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Mirrors the constants in basic.py. The decision chain is parallel:
#   situation (25)  ->  available options (1M)  ->  best option (5)
# corresponds to basic's
#   situation (25)  ->  self (1M)               ->  person-by-situation (5)
_DEFAULT_OBSERVATION_HISTORY_LENGTH = 1_000_000
_DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH = 25
_DEFAULT_AVAILABLE_OPTIONS_HISTORY_LENGTH = 1_000_000
_DEFAULT_BEST_OPTION_HISTORY_LENGTH = 5


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab for a 'rational' agent (logic of consequence)."""

  description: str = (  # pyrefly: ignore[bad-override]
      'An entity that makes decisions by asking '
      '"What situation am I in right now?", "Which options are available to '
      'me?", and "Of those options, which is best?"'
  )
  params: Mapping[str, str] = dataclasses.field(  # pyrefly: ignore[bad-assignment]
      default_factory=lambda: {
          'name': 'Rational Agent',
          'goal': '',
          'randomize_choices': True,
          'prefix_entity_name': True,
          'observation_history_length':
              _DEFAULT_OBSERVATION_HISTORY_LENGTH,
          'situation_perception_history_length': (
              _DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH
          ),
          'available_options_history_length': (
              _DEFAULT_AVAILABLE_OPTIONS_HISTORY_LENGTH
          ),
          'best_option_history_length': (
              _DEFAULT_BEST_OPTION_HISTORY_LENGTH
          ),
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the rational agent.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      A rational entity agent.
    """
    entity_name = self.params.get('name', 'Rational Agent')
    entity_goal = self.params.get('goal', '')
    randomize_choices = self.params.get('randomize_choices', True)
    prefix_entity_name = self.params.get('prefix_entity_name', True)
    observation_history_length = self.params.get(
        'observation_history_length',
        _DEFAULT_OBSERVATION_HISTORY_LENGTH,
    )
    situation_perception_history_length = self.params.get(
        'situation_perception_history_length',
        _DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH,
    )
    available_options_history_length = self.params.get(
        'available_options_history_length',
        _DEFAULT_AVAILABLE_OPTIONS_HISTORY_LENGTH,
    )
    best_option_history_length = self.params.get(
        'best_option_history_length',
        _DEFAULT_BEST_OPTION_HISTORY_LENGTH,
    )

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = 'Instructions'
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nInstructions',
    )

    observation_to_memory_key = 'ObservationToMemory'
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY)
    observation = agent_components.observation.LastNObservations(
        history_length=observation_history_length,  # pyrefly: ignore[bad-argument-type]
        pre_act_label=(
            '\nEvents so far (ordered from least recent to most recent)'
        ),
    )

    # Create the goal component early so perception components can reference it.
    if entity_goal:
      goal_key = 'Goal'
      overarching_goal = agent_components.constant.Constant(
          state=entity_goal, pre_act_label='\nGoal'
      )
    else:
      goal_key = None
      overarching_goal = None

    # When a goal is set, include it in the perception components so the
    # intermediate reasoning chain (not just the final action prompt)
    # explicitly considers the agent's goal.
    goal_components = [goal_key] if goal_key else []

    situation_perception_key = 'SituationPerception'
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            num_memories_to_retrieve=situation_perception_history_length,
            components=goal_components,
            pre_act_label=(
                f'\nQuestion: What situation is {entity_name} in right now?'
                '\nAnswer'
            ),
        )
    )

    options_perception_key = 'AvailableOptionsPerception'
    options_perception = (
        agent_components.question_of_recent_memories.AvailableOptionsPerception(
            model=model,
            num_memories_to_retrieve=available_options_history_length,
            components=goal_components + [
                situation_perception_key,
            ],
            pre_act_label=(
                f'\nQuestion: Which options are available to {entity_name} '
                'right now?\nAnswer'
            ),
        )
    )

    best_option_perception_key = 'BestOptionPerception'
    best_option_perception = (
        agent_components.question_of_recent_memories.BestOptionPerception(
            model=model,
            num_memories_to_retrieve=best_option_history_length,
            components=goal_components + [
                situation_perception_key,
                options_perception_key,
            ],
            pre_act_label=(
                f'\nQuestion: Of the options available to {entity_name}, '
                f'which choice of action or strategy is best for {entity_name} '
                'to take right now?\nAnswer'
            ),
        )
    )

    components_of_agent = {
        instructions_key: instructions,
        observation_to_memory_key: observation_to_memory,
        situation_perception_key: situation_perception,
        options_perception_key: options_perception,
        best_option_perception_key: best_option_perception,
        observation_key: observation,
        memory_key: memory,
    }

    component_order = list(components_of_agent.keys())

    if overarching_goal is not None:
      components_of_agent[goal_key] = overarching_goal  # pyrefly: ignore[unsupported-operation]
      # Place goal after the instructions.
      component_order.insert(1, goal_key)  # pyrefly: ignore[bad-argument-type]

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
        randomize_choices=randomize_choices,  # pyrefly: ignore[bad-argument-type]
        prefix_entity_name=prefix_entity_name,  # pyrefly: ignore[bad-argument-type]
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
        measurements=self.params.get('measurements'),  # pyrefly: ignore[bad-argument-type]
    )

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

"""A prefab implementing an entity with a minimal set of components."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

DEFAULT_INSTRUCTIONS_COMPONENT_KEY = 'Instructions'
DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL = '\nInstructions'
DEFAULT_GOAL_COMPONENT_KEY = 'Goal'


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab implementing an entity with a minimal set of components."""

  description: str = (
      'An entity that has a minimal set of components and is configurable by'
      ' the user. The initial set of components manage memory, observations,'
      ' and instructions. If goal is specified, the entity will have a goal '
      'constant component.'
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Alice',
          'goal': '',
          # A custom instruction to use instead of the default instructions.
          'custom_instructions': '',
          'extra_components': {},
          # A mapping from component name to the index at which to insert it
          # in the component order. If not specified, the extra components
          # will be inserted at the end of the component order.
          'extra_components_index': {},

      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent.

    Args:
      model: The language model to use.
      memory_bank: The agent's memory_bank object.

    Returns:
      An entity.
    """

    agent_name = self.params.get('name', 'Alice')

    custom_instructions = self.params.get('custom_instructions', None)
    if custom_instructions is not None:
      instructions = agent_components.constant.Constant(
          state=custom_instructions,
          pre_act_label=DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL,
      )
    else:
      instructions = agent_components.instructions.Instructions(
          agent_name=agent_name,
          pre_act_label=DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL,
      )

    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_label = '\nObservation'
    observation = agent_components.observation.LastNObservations(
        history_length=100, pre_act_label=observation_label
    )

    components_of_agent = {
        DEFAULT_INSTRUCTIONS_COMPONENT_KEY: instructions,
        'observation_to_memory': observation_to_memory,
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: (
            observation
        ),
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
            agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
        ),
    }

    component_order = list(components_of_agent.keys())

    # Add the extra components to the end of the component order.
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})

    # Check that extra_components_index is a dict.
    if not isinstance(extra_components_index, dict) or not isinstance(
        extra_components, dict
    ):
      raise ValueError(
          'extra_components_index and extra_components must be dict. Got'
          f' {type(extra_components_index)} and {type(extra_components)}'
      )

    if extra_components:
      if not extra_components_index:
        extra_components_index = {
            component_name: -1
            for component_name in extra_components.keys()
        }
      for component_name, index in extra_components_index.items():
        components_of_agent[component_name] = extra_components[component_name]
        component_order.insert(index, component_name)

    if self.params.get('goal', ''):
      goal_key = DEFAULT_GOAL_COMPONENT_KEY
      goal = agent_components.constant.Constant(
          state=self.params.get('goal', ''),
          pre_act_label='Overarching goal',
      )
      components_of_agent[goal_key] = goal
      # Place goal after the instructions.
      component_order.insert(1, goal_key)

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent


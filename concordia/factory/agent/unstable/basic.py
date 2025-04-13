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

"""A factory implementing the three key questions actor."""

from collections.abc import Callable
import json

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory
from concordia.associative_memory.unstable import formative_memories
from concordia.clocks import game_clock
from concordia.components.agent import unstable as agent_components
from concordia.language_model import language_model
from concordia.typing.unstable import entity_component
from concordia.utils import measurements as measurements_lib
import numpy as np


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    clock: game_clock.MultiIntervalClock | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory_bank: The agent's memory_bank object.
    clock: DEPRECATED. The clock argument is ignored by this factory.

  Returns:
    An agent.
  """
  del clock
  agent_name = config.name

  measurements = measurements_lib.Measurements()

  instructions_key = 'Instructions'
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      pre_act_label='\nInstructions',
      logging_channel=measurements.get_channel(instructions_key).on_next,
  )

  observation_to_memory_key = 'Observation'
  observation_to_memory = agent_components.observation.ObservationToMemory()

  observation_key = (
      agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY)
  observation = agent_components.observation.LastNObservations(
      history_length=100,
      pre_act_label=(
          '\nEvents so far (ordered from least recent to most recent)'
      ),
      logging_channel=measurements.get_channel(observation_key).on_next,
  )

  situation_perception_key = 'SituationPerception'
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          pre_act_label=(
              f'\nQuestion: What situation is {agent_name} in right now?'
              '\nAnswer'
          ),
          logging_channel=measurements.get_channel(
              situation_perception_key
          ).on_next,
      )
  )
  self_perception_key = 'SelfPerception'
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      pre_act_label=f'\nQuestion: What kind of person is {agent_name}?\nAnswer',
      logging_channel=measurements.get_channel(self_perception_key).on_next,
  )

  person_by_situation_key = 'PersonBySituation'
  person_by_situation = (
      agent_components.question_of_recent_memories.PersonBySituation(
          model=model,
          components={
              self_perception_key: self_perception.get_pre_act_label(),
              situation_perception_key: (
                  situation_perception.get_pre_act_label()),
          },
          pre_act_label=(
              f'\nQuestion: What would a person like {agent_name} do in '
              'a situation like this?\nAnswer'),
          logging_channel=measurements.get_channel(
              person_by_situation_key).on_next,
      )
  )
  relevant_memories_key = 'RelevantMemories'
  relevant_memories = (
      agent_components.all_similar_memories.AllSimilarMemories(
          model=model,
          components={
              situation_perception_key: (
                  situation_perception.get_pre_act_label()
              ),
          },
          num_memories_to_retrieve=10,
          pre_act_label='\nRecalled memories and observations',
          logging_channel=measurements.get_channel(
              relevant_memories_key
          ).on_next,
      )
  )

  if config.goal:
    goal_key = 'Goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_label='\nGoal',
        logging_channel=measurements.get_channel(goal_key).on_next)
  else:
    goal_key = None
    overarching_goal = None

  components_of_agent = {
      instructions_key: instructions,
      observation_to_memory_key: observation_to_memory,
      relevant_memories_key: relevant_memories,
      self_perception_key: self_perception,
      situation_perception_key: situation_perception,
      person_by_situation_key: person_by_situation,
      observation_key: observation,
      agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
          agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
      ),
  }

  component_order = list(components_of_agent.keys())

  if overarching_goal is not None:
    components_of_agent[goal_key] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_key)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      component_order=component_order,
      logging_channel=measurements.get_channel('Act').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent


def save_to_json(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
) -> str:
  """Saves an agent to JSON data.

  This function saves the agent's state to a JSON string, which can be loaded
  afterwards with `rebuild_from_json`. The JSON data
  includes the state of the agent's context components, act component, memory,
  agent name and the initial config. The clock, model and embedder are not
  saved and will have to be provided when the agent is rebuilt. The agent must
  be in the `READY` phase to be saved.

  Args:
    agent: The agent to save.

  Returns:
    A JSON string representing the agent's state.

  Raises:
    ValueError: If the agent is not in the READY phase.
  """

  if agent.get_phase() != entity_component.Phase.READY:
    raise ValueError('The agent must be in the `READY` phase to be saved.')

  data = {
      component_key: agent.get_component(component_key).get_state()
      for component_key in agent.get_all_context_components()
  }

  data['act_component'] = agent.get_act_component().get_state()

  config = agent.get_config()
  if config is not None:
    data['agent_config'] = config.to_dict()

  return json.dumps(data)


def rebuild_from_json(
    json_data: str,
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Rebuilds an agent from JSON data."""

  data = json.loads(json_data)

  new_agent_memory_bank = basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=embedder,
  )

  if 'agent_config' not in data:
    raise ValueError('The JSON data does not contain the agent config.')
  agent_config = formative_memories.AgentConfig.from_dict(
      data.pop('agent_config')
  )

  agent = build_agent(
      config=agent_config,
      model=model,
      memory_bank=new_agent_memory_bank,
  )

  for component_key in agent.get_all_context_components():
    agent.get_component(component_key).set_state(data.pop(component_key))

  agent.get_act_component().set_state(data.pop('act_component'))

  assert not data, f'Unused data {sorted(data)}'
  return agent

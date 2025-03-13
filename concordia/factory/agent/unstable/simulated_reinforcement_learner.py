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

"""An Actor Factory."""

from concordia.agents.unstable import entity_agent_with_logging
from concordia.components.agent import unstable as agent_components
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib


def build_agent(
    *,
    reward: str,
    model: language_model.LanguageModel,
    history_length: int = 100,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    reward: Reward variable the actor tries to maximize e.g. "money".
    model: The language model to use.
    history_length: The number of previous observations to keep.

  Returns:
    An actor that simulates reinforcement learning.
  """

  measurements = measurements_lib.Measurements()

  instructions_key = 'instructions'
  instructions = agent_components.constant.Constant(
      state='You are simulating a reinforcement learning agent.',
      pre_act_key='\nInstructions',
      logging_channel=measurements.get_channel(instructions_key).on_next,
  )

  reward_key = 'goal'
  reward_component = agent_components.constant.Constant(
      state=reward,
      pre_act_key='\nReward variable to maximize',
      logging_channel=measurements.get_channel(reward_key).on_next,
  )

  memory_component_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_NAME
  memory_component = agent_components.memory.ListMemory(memory_bank=[])

  observation_to_memory_key = 'observation_to_memory'
  observation_to_memory = agent_components.observation.ObservationToMemory()

  history_key = '\nHistory of previous observations'
  history_component = agent_components.observation.LastNObservations(
      history_length=history_length,
      pre_act_key=history_key,
      logging_channel=measurements.get_channel(history_key).on_next,
  )

  observation_key = (
      agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_NAME
  )
  observation_component = (
      agent_components.observation.ObservationsSinceLastPreAct(
          pre_act_key='\nLatest observation',
          logging_channel=measurements.get_channel(observation_key).on_next,
      )
  )

  components_of_agent = {
      instructions_key: instructions,
      reward_key: reward_component,
      observation_to_memory_key: observation_to_memory,
      memory_component_key: memory_component,
      observation_key: observation_component,
      history_key: history_component,
  }

  act_component_key = 'act_component'
  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      component_order=list(components_of_agent.keys()),
      logging_channel=measurements.get_channel(act_component_key).on_next,
  )

  actor = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='',
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return actor

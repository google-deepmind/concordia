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

"""A Generic Agent Factory. This is temporary code. Do not use it."""

import datetime

from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components.agent import v2 as agent_components
from concordia.language_model import language_model
from concordia.typing import component


def get_instructions(agent_name: str) -> component.Component:
  """Get role playing instructions for the agent."""
  instructions = agent_components.constant.Constant(
      state=(
          f'The instructions for how to play the role of {agent_name} are as '
          'follows. This is a social science experiment studying how well you '
          f'play the role of a character named {agent_name}. The experiment '
          'is structured as a tabletop roleplaying game (like dungeons and '
          'dragons). However, in this case it is a serious social science '
          'experiment and simulation. The goal is to be realistic. It is '
          f'important to play the role of a person like {agent_name} as '
          f'accurately as possible, i.e., by responding in ways that you think '
          f'it is likely a person like {agent_name} would respond, and taking '
          f'into account all information about {agent_name} that you have. '
          'Always use third-person limited perspective.'
      ))
  return instructions


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent.EntityAgent:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  instructions = get_instructions(agent_name=agent_name)

  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      memory=memory
  )

  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      memory=memory,
      agent_name=agent_name,
      components={'summary of recent obervations': observation},
      clock_now=clock.now,
      num_memories_to_retrieve=10,
  )

  act_component = agent_components.legacy_act_component.ActComponent(
      model=model,
      clock=clock,
  )

  components_of_agent = {
      'Role playing instructions': instructions,
      'Observation': observation,
      'Recalled memories and observations': relevant_memories,
  }
  if config.goal:
    overarching_goal = agent_components.constant.Constant(
        state=config.goal)
    components_of_agent['Overarching goal'] = overarching_goal

  agent = entity_agent.EntityAgent(
      agent_name=agent_name,
      act_component=act_component,
      components=components_of_agent,
  )

  return agent

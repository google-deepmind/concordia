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
from concordia.memory_bank import legacy_associative_memory
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

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  instructions = get_instructions(agent_name=agent_name)

  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      verbose=True,
  )
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      verbose=True,
  )
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
  )
  identity_characteristics = agent_components.identity.IdentityWithoutPreAct(
      model=model,
      verbose=True,
  )
  self_perception = agent_components.self_perception.SelfPerception(
      model=model,
      components={'identity_characteristics': identity_characteristics},
      verbose=True,
  )
  situation_perception = (
      agent_components.situation_perception.SituationPerception(
          model=model,
          components={'Observation': observation,
                      'Summary of recent observations': observation_summary},
          clock_now=clock.now,
          verbose=True,
      )
  )
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={'Summary of recent observations': observation_summary,
                  'The current date/time is': time_display},
      num_memories_to_retrieve=10,
  )

  components_of_agent = {
      'Role playing instructions': instructions,
      'Observation': observation,
      'Summary of recent observations': observation_summary,
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer':
          self_perception,
      (f'\nQuestion: What kind of situation is {agent_name} in '
       'right now?\nAnswer'): situation_perception,
      'Current time': time_display,
      'Recalled memories and observations': relevant_memories,
  }
  component_order = list(components_of_agent.keys())
  components_of_agent['Identity'] = identity_characteristics
  components_of_agent['__memory__'] = (
      agent_components.memory_component.MemoryComponent(raw_memory))
  if config.goal:
    key = 'Overarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal)
    components_of_agent[key] = overarching_goal
    component_order.insert(1, key)  # Place the goal after the instructions.

  act_component = agent_components.legacy_act_component.ActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
  )

  agent = entity_agent.EntityAgent(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
  )

  return agent

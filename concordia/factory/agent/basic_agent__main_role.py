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

"""A Generic Agent Factory."""

from collections.abc import Sequence
import datetime

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import component

DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'


def get_instructions(agent_name: str) -> component.Component:
  """Get role playing instructions for the agent."""
  instructions = generic_components.constant.ConstantComponent(
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
      ),
      name='role playing instructions\n')
  return instructions


def get_time_display_component(
    clock: game_clock.MultiIntervalClock,
    name: str = 'Current time',
) -> component.Component:
  """Get a component that displays the current time."""
  return generic_components.report_function.ReportFunction(
      name=name,
      function=clock.current_time_interval_str,
  )


def get_current_observation_component(
    agent_name: str,
    clock: game_clock.MultiIntervalClock,
    agent_memory: associative_memory.AssociativeMemory) -> component.Component:
  """Component that displays current observations and adds them to memory."""
  return agent_components.observation.Observation(
      agent_name=agent_name,
      clock_now=clock.now,
      memory=agent_memory,
      timeframe=clock.get_step_size(),
      component_name='current observations',
  )


def get_somatic_state_component(
    model: language_model.LanguageModel,
    agent_name: str,
    clock: game_clock.MultiIntervalClock,
    agent_memory: associative_memory.AssociativeMemory) -> component.Component:
  """Component that reports the agent's current somatic state."""
  return agent_components.somatic_state.SomaticState(
      model=model,
      memory=agent_memory,
      agent_name=agent_name,
      clock_now=clock.now,
  )


def get_summary_obs_component(
    model: language_model.LanguageModel,
    agent_name: str,
    clock: game_clock.MultiIntervalClock,
    agent_memory: associative_memory.AssociativeMemory,
    components: Sequence[component.Component],
) -> component.Component:
  """Component that reports a summary of the agent's current observations."""
  return agent_components.observation.ObservationSummary(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      memory=agent_memory,
      components=list(components),
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      component_name='summary of observations',
  )


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> basic_agent.BasicAgent:
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
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  instructions = get_instructions(agent_name=agent_name)
  time = get_time_display_component(clock=clock)

  overarching_goal = generic_components.constant.ConstantComponent(
      state=config.goal, name='overarching goal')

  current_obs = get_current_observation_component(
      agent_name=agent_name,
      clock=clock,
      agent_memory=memory,
  )

  summary_obs = get_summary_obs_component(
      model=model,
      agent_name=agent_name,
      clock=clock,
      agent_memory=memory,
      components=[current_obs],
  )

  identity_characteristics = agent_components.identity.SimIdentity(
      model=model,
      memory=memory,
      agent_name=agent_name,
      name='identity',
      clock_now=clock.now,
  )
  self_perception = agent_components.self_perception.SelfPerception(
      name=f'\nQuestion: What kind of person is {agent_name}?\nAnswer',
      model=model,
      memory=memory,
      agent_name=agent_name,
      components=[identity_characteristics],
      clock_now=clock.now,
  )
  situation_perception = (
      agent_components.situation_perception.SituationPerception(
          name=(
              f'\nQuestion: What kind of situation is {agent_name} in '
              + 'right now?\nAnswer'
          ),
          model=model,
          memory=memory,
          agent_name=agent_name,
          components=[current_obs, summary_obs],
          clock_now=clock.now,
      )
  )
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      name='relevant memories',
      model=model,
      memory=memory,
      agent_name=agent_name,
      components=[summary_obs],
      clock_now=clock.now,
      num_memories_to_retrieve=10,
  )

  person_by_situation = (
      agent_components.person_by_situation.PersonBySituation(
          name=(
              f'\nQuestion: What would a person like {agent_name} do in a '
              + 'situation like this?\nAnswer'
          ),
          model=model,
          memory=memory,
          agent_name=agent_name,
          clock_now=clock.now,
          components=[self_perception, situation_perception],
      )
  )

  plan = agent_components.plan.SimPlan(
      model=model,
      memory=memory,
      agent_name=agent_name,
      clock_now=clock.now,
      components=[overarching_goal,
                  relevant_memories,
                  self_perception,
                  situation_perception,
                  person_by_situation],
      name=(f'Question: What is {agent_name}\'s plan?\nAnswer'),
      goal=person_by_situation,
      horizon=DEFAULT_PLANNING_HORIZON,
  )

  sequential = generic_components.sequential.Sequential(
      name='information',
      components=[
          time,
          current_obs,
          relevant_memories,
          self_perception,
          situation_perception,
          person_by_situation,
          plan,
      ]
  )

  agent = basic_agent.BasicAgent(
      model=model,
      agent_name=agent_name,
      clock=clock,
      components=[instructions,
                  overarching_goal,
                  sequential],
      update_interval=update_time_interval,
  )

  return agent

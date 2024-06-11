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
from concordia.factory.agent import basic_agent__main_role
from concordia.language_model import language_model
from concordia.typing import component


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
    additional_components: Sequence[component.Component] = (),
) -> basic_agent.BasicAgent:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.
    additional_components: Additional components to add to the agent.

  Returns:
    An agent.
  """
  if config.extras.get('main_character', False):
    raise ValueError('This function is meant for a supporting character '
                     'but it was called on a main character.')

  agent_name = config.name

  instructions = basic_agent__main_role.get_instructions(agent_name)

  time = generic_components.report_function.ReportFunction(
      name='Current time',
      function=clock.current_time_interval_str,
  )

  current_obs = agent_components.observation.Observation(
      agent_name=agent_name,
      clock_now=clock.now,
      memory=memory,
      timeframe=clock.get_step_size(),
      component_name='current observations',
  )
  somatic_state = agent_components.somatic_state.SomaticState(
      model=model,
      memory=memory,
      agent_name=agent_name,
      clock_now=clock.now,
  )
  summary_obs = agent_components.observation.ObservationSummary(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      memory=memory,
      components=[current_obs],
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      component_name='summary of observations',
  )
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      name='relevant memories',
      model=model,
      memory=memory,
      agent_name=agent_name,
      components=[summary_obs, somatic_state],
      clock_now=clock.now,
      num_memories_to_retrieve=10,
  )
  self_perception = agent_components.self_perception.SelfPerception(
      name=f'\nQuestion: What kind of person is {agent_name}\nAnswer',
      model=model,
      memory=memory,
      agent_name=agent_name,
      clock_now=clock.now,
  )
  situation_perception = (
      agent_components.situation_perception.SituationPerception(
          name=(
              f'Question: What kind of situation is {agent_name} in '
              + 'right now?\nAnswer '
          ),
          model=model,
          memory=memory,
          agent_name=agent_name,
          components=[
              current_obs, somatic_state, summary_obs, relevant_memories],
          clock_now=clock.now,
      )
  )
  person_by_situation = (
      agent_components.person_by_situation.PersonBySituation(
          name=(
              f'Question: What would a person like {agent_name} do in a '
              + 'situation like this?\nAnswer '
          ),
          model=model,
          memory=memory,
          agent_name=agent_name,
          clock_now=clock.now,
          components=[self_perception, situation_perception],
      )
  )

  sequential = generic_components.sequential.Sequential(
      name='information',
      components=[
          time,
          current_obs,
          summary_obs,
          relevant_memories,
          self_perception,
          situation_perception,
          person_by_situation,
          *additional_components,
      ]
  )

  agent = basic_agent.BasicAgent(
      model=model,
      agent_name=agent_name,
      clock=clock,
      components=[instructions,
                  sequential],
      update_interval=update_time_interval
  )

  return agent

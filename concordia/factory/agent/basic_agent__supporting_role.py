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
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.factory.agent import basic_agent__main_role
from concordia.language_model import language_model
from concordia.metrics import common_sense_morality
from concordia.metrics import goal_achievement
from concordia.metrics import opinion_of_others
from concordia.typing import component
from concordia.utils import measurements as measurements_lib


def build_agent(
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
    time_step: datetime.timedelta,
    blank_memory_factory: blank_memories.MemoryFactory,
    formative_memory_factory: formative_memories.FormativeMemoryFactory,
    agent_config: formative_memories.AgentConfig,
    player_names: list[str],
    custom_components: Sequence[component.Component] | None = None,
    measurements: measurements_lib.Measurements | None = None,
    debug: bool = False) -> tuple[basic_agent.BasicAgent,
                                  associative_memory.AssociativeMemory]:
  """Build an agent."""
  if agent_config.extras.get('main_character', False):
    raise ValueError('This function is meant for a supporting character '
                     'but it was called on a main character.')

  agent_name = agent_config.name
  custom_components = custom_components or []

  if debug:
    mem = blank_memory_factory.make_blank_memory()
  else:
    mem = formative_memory_factory.make_memories(agent_config)

  instructions = basic_agent__main_role.get_instructions(agent_name)

  time = generic_components.report_function.ReportFunction(
      name='Current time',
      function=clock.current_time_interval_str,
  )

  current_obs = agent_components.observation.Observation(
      agent_name=agent_name,
      clock_now=clock.now,
      memory=mem,
      timeframe=clock.get_step_size(),
      component_name='current observations',
  )
  somatic_state = agent_components.somatic_state.SomaticState(
      model=model,
      memory=mem,
      agent_name=agent_name,
      clock_now=clock.now,
  )
  summary_obs = agent_components.observation.ObservationSummary(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      memory=mem,
      components=[current_obs, somatic_state],
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      component_name='summary of observations',
  )

  self_perception = agent_components.self_perception.SelfPerception(
      name=f'answer to what kind of person is {agent_name}',
      model=model,
      memory=mem,
      agent_name=agent_name,
      clock_now=clock.now,
  )
  situation_perception = (
      agent_components.situation_perception.SituationPerception(
          name=(
              f'answer to what kind of situation is {agent_name} in '
              + 'right now'
          ),
          model=model,
          memory=mem,
          agent_name=agent_name,
          components=[current_obs, somatic_state, summary_obs],
          clock_now=clock.now,
      )
  )
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      name='relevant memories',
      model=model,
      memory=mem,
      agent_name=agent_name,
      components=[summary_obs, self_perception],
      clock_now=clock.now,
      num_memories_to_retrieve=10,
      verbose=True,
    )

  type_specific_components = [relevant_memories,
                              self_perception,
                              situation_perception]
  for custom_component in custom_components:
    type_specific_components.append(custom_component)

  if debug:
    type_specific_components = []
  agent = basic_agent.BasicAgent(
      model=model,
      agent_name=agent_name,
      clock=clock,
      verbose=False,
      components=[instructions,
                  time,
                  current_obs,
                  *type_specific_components],
      update_interval=time_step
  )

  goal_metric = goal_achievement.GoalAchievementMetric(
      model=model,
      player_name=agent_name,
      player_goal=agent_config.goal,
      clock=clock,
      name='Goal Achievement',
      measurements=measurements,
      channel='goal_achievement',
      verbose=False,
  )
  morality_metric = common_sense_morality.CommonSenseMoralityMetric(
      model=model,
      player_name=agent_name,
      clock=clock,
      name='Morality',
      verbose=False,
      measurements=measurements,
      channel='common_sense_morality',
  )
  reputation_metric = opinion_of_others.OpinionOfOthersMetric(
      model=model,
      player_name=agent_name,
      player_names=player_names,
      context_fn=agent.state,
      clock=clock,
      name='Opinion',
      verbose=False,
      measurements=measurements,
      channel='opinion_of_others',
      question='What is {opining_player}\'s opinion of {of_player}?',
  )
  agent.add_component(reputation_metric)
  agent.add_component(goal_metric)
  agent.add_component(morality_metric)

  for extra_memory in agent_config.extras['player_specific_memories']:
    mem.add(f'{extra_memory}', tags=['initial_player_specific_memory'])
  return agent, mem

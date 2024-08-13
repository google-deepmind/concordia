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

"""A factory implementing a simulation of a user for a product."""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Unused (but required by the interface for now)

  Returns:
    An agent.
  """
  del update_time_interval
  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
  measurements = measurements_lib.Measurements()

  instructions = components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )
  observation_label = '\nObservation'
  observation = components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = '\nSummary of recent observations'
  observation_summary = components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  time_display = components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )
  options_perception_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    options_perception_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None
  options_perception_components.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
  })
  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = (
      components.question_of_recent_memories.AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
      )
  )
  identity_label = '\nIdentity characteristics'
  identity_characteristics = (
      components.question_of_query_associated_memories.IdentityWithoutPreAct(
          model=model,
          logging_channel=measurements.get_channel(
              'IdentityWithoutPreAct'
          ).on_next,
          pre_act_key=identity_label,
      )
  )
  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer'
  )
  self_perception = components.question_of_recent_memories.SelfPerception(
      model=model,
      components={_get_class_name(identity_characteristics): identity_label},
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )
  situation_perception_label = (
      f'\nQuestion: What kind of situation is {agent_name} in '
      'right now?\nAnswer')
  situation_perception = (
      components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
              _get_class_name(options_perception): options_perception_label,
          },
          clock_now=clock.now,
          pre_act_key=situation_perception_label,
          logging_channel=measurements.get_channel(
              'SituationPerception'
          ).on_next,
      )
  )
  person_by_situation_label = (
      f'\nQuestion: What would a person like {agent_name} do in '
      'a situation like this?\nAnswer')
  person_by_situation = (
      components.question_of_recent_memories.PersonBySituation(
          model=model,
          components={
              _get_class_name(self_perception): self_perception_label,
              _get_class_name(situation_perception): situation_perception_label,
          },
          clock_now=clock.now,
          pre_act_key=person_by_situation_label,
          logging_channel=measurements.get_channel('PersonBySituation').on_next,
      )
  )
  reflection_label = '\nReflection'
  reflection = (
      components.justify_recent_voluntary_actions.JustifyRecentVoluntaryActions(
          model=model,
          components={_get_class_name(self_perception): self_perception_label},
          clock_now=clock.now,
          pre_act_key=reflection_label,
          logging_channel=measurements.get_channel(
              'JustifyRecentVoluntaryActions').on_next,
      )
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      observation,
      reflection,
      observation_summary,
      relevant_memories,
      options_perception,
      self_perception,
      situation_perception,
      person_by_situation,
      time_display,

      # Components that do not provide pre_act context.
      identity_characteristics,
  )
  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          components.memory_component.MemoryComponent(raw_memory))
  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent

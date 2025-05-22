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

from collections.abc import Mapping
import datetime
import types

from concordia.agents.deprecated import entity_agent_with_logging
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import formative_memories
from concordia.clocks import game_clock
from concordia.components.agent import deprecated as agent_components
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.utils.deprecated import measurements as measurements_lib


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
    fixed_response_by_call_to_action: Mapping[
        str, str] = types.MappingProxyType({}),
    additional_components: Mapping[
        entity_component.ComponentName,
        entity_component.ContextComponent,
    ] = types.MappingProxyType({}),
    search_in_prompt: bool = False,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.
    fixed_response_by_call_to_action: A mapping from call to action to fixed
      response.
    additional_components: Additional components to add to the agent.
    search_in_prompt: Optionally, search for the target string in the entire
        prompt (i.e. the context), not just the call to action. Off by default.

  Returns:
    An agent.
  """
  del update_time_interval
  if config.extras.get('main_character', False):
    raise ValueError('This function is meant for a supporting character '
                     'but it was called on a main character.')

  agent_name = config.name
  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
  measurements = measurements_lib.Measurements()

  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  somatic_state_label = '\nSensations and feelings'
  somatic_state = agent_components.question_of_query_associated_memories.SomaticStateWithoutPreAct(
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('SomaticState').on_next,
      pre_act_key=somatic_state_label,
  )
  observation_summary_label = '\nSummary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      components={_get_class_name(somatic_state): somatic_state_label},
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = (
      agent_components.all_similar_memories.AllSimilarMemoriesWithoutPreAct(
          model=model,
          components={
              _get_class_name(observation_summary): observation_summary_label,
              _get_class_name(somatic_state): somatic_state_label,
              _get_class_name(time_display): 'The current date/time is',
          },
          num_memories_to_retrieve=10,
          pre_act_key=relevant_memories_label,
          logging_channel=measurements.get_channel(
              'AllSimilarMemoriesWithoutPreAct'
          ).on_next,
      )
  )
  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer')
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )
  situation_perception_label = (
      f'\nQuestion: What kind of situation is {agent_name} in '
      'right now?\nAnswer')
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(somatic_state): somatic_state_label,
              _get_class_name(observation_summary): observation_summary_label,
              _get_class_name(relevant_memories): relevant_memories_label,
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
      agent_components.question_of_recent_memories.PersonBySituation(
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

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      observation,
      observation_summary,
      self_perception,
      situation_perception,
      person_by_situation,

      # Components that do not provide pre_act context.
      somatic_state,
      relevant_memories,
  )

  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          agent_components.memory_component.MemoryComponent(raw_memory))
  components_of_agent.update(additional_components)

  act_component = agent_components.puppet_act_component.PuppetActComponent(
      model=model,
      clock=clock,
      logging_channel=measurements.get_channel('ActComponent').on_next,
      fixed_responses=fixed_response_by_call_to_action,
      search_in_prompt=search_in_prompt,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent

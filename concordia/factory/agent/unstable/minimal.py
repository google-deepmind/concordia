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

"""An Agent Factory."""

from collections.abc import Callable
import json

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory
from concordia.associative_memory.unstable import formative_memories
from concordia.clocks import game_clock
from concordia.components.agent import unstable as agent_components
from concordia.language_model import language_model
from concordia.typing.unstable import entity_component
import numpy as np


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


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
    clock: The clock to use.

  Returns:
    An agent.
  """

  agent_name = config.name

  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name
  )

  if clock:
    time_display = agent_components.report_function.ReportFunction(
        function=clock.current_time_interval_str,
        pre_act_label='\nCurrent time',
    )
  else:
    time_display = None

  observation_to_memory = agent_components.observation.ObservationToMemory(
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.LastNObservations(
      history_length=100,
      pre_act_label=observation_label,
  )

  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_label=goal_label,
    )
  else:
    goal_label = None
    overarching_goal = None

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      observation_to_memory,
  )
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }
  list_memory_bank = list(memory_bank.get_data_frame()['text'])
  components_of_agent[agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY] = (
      agent_components.memory.ListMemory(memory_bank=list_memory_bank)
  )
  components_of_agent[
      agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
  ] = observation

  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  if time_display is not None:
    components_of_agent[time_display.get_pre_act_label()] = time_display
    # Place time display after the instructions.
    component_order.insert(1, time_display.get_pre_act_label())

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
      component_name: agent.get_component(component_name).get_state()
      for component_name in agent.get_all_context_components()
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
    clock: game_clock.MultiIntervalClock | None = None,
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
      clock=clock,
  )

  for component_name in agent.get_all_context_components():
    agent.get_component(component_name).set_state(data.pop(component_name))

  agent.get_act_component().set_state(data.pop('act_component'))

  assert not data, f'Unused data {sorted(data)}'
  return agent

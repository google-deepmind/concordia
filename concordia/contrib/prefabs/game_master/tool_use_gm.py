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

"""GameMaster prefab that enables tool use via MCP protocol."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import next_acting
from concordia.components.game_master import terminate as terminate_components
from concordia.contrib.components.game_master import mcp_tool_executor
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMasterWithMCPTools(prefab_lib.Prefab):
  """A prefab game master that supports MCP tool use."""

  description: str = (
      'A game master that enables agents to use external tools via MCP.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'GameMaster with Tool Support',
          'mcp_server_command': '',
          'mcp_server_args': [],
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the GameMaster entity with MCP tool support."""

    name = self.params.get('name', 'GameMaster with Tool Support')
    mcp_server_command = self.params.get('mcp_server_command', '')
    mcp_server_args = self.params.get('mcp_server_args', [])
    player_names = [entity.name for entity in self.entities]

    # Memory component
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Instructions
    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

    # Tool instructions as a constant
    tool_instructions_key = 'tool_instructions'
    tool_instructions = actor_components.constant.Constant(
        state=(
            'Agents in this simulation have access to external tools via MCP.'
            ' When an agent requests external information, execute the'
            ' appropriate tool and inject the result into observations.'
        ),
        pre_act_label='Tool Support Instructions',
    )

    # Observation components
    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1000,
    )

    # Relevant memories
    relevant_memories_key = 'relevant_memories'
    relevant_memories = actor_components.all_similar_memories.AllSimilarMemories(
        model=model,
        components=[observation_key],
        num_memories_to_retrieve=10,
        pre_act_label='Relevant memories',
    )

    # MCP Tool executor
    tool_executor_key = 'tool_executor'
    tool_executor_component = mcp_tool_executor.MCPToolExecutor(
        model=model,
        mcp_server_command=mcp_server_command,
        mcp_server_args=list(mcp_server_args),
        memory_component_key=memory_component_key,
        pre_act_label='Tool Executor',
    )

    # Event resolution (act component)
    player_characters_key = 'player_characters'
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label='Story so far',
    )

    next_actor_key = next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = next_acting.NextActingAllEntities(
        player_names=player_names,
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    default_action_spec = entity_lib.ActionSpec(
        call_to_action='What does {name} do next?',
        output_type=entity_lib.OutputType.FREE,
        options=(),
        tag='action',
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=default_action_spec
    )

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components=[
            instructions_key,
            player_characters_key,
            relevant_memories_key,
            display_events_key,
        ],
    )

    terminate_key = terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY
    terminate_component = terminate_components.NeverTerminate()

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=[],
        components=[
            instructions_key,
            tool_instructions_key,
            player_characters_key,
            relevant_memories_key,
            display_events_key,
        ],
    )

    component_order = [
        instructions_key,
        tool_instructions_key,
        player_characters_key,
        observation_key,
        observation_to_memory_key,
        relevant_memories_key,
        display_events_key,
        memory_component_key,
        tool_executor_key,
        make_observation_key,
        next_actor_key,
        next_action_spec_key,
        event_resolution_key,
        terminate_key,
    ]

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    components_of_game_master = {
        instructions_key: instructions,
        tool_instructions_key: tool_instructions,
        player_characters_key: player_characters,
        observation_key: observation,
        observation_to_memory_key: observation_to_memory,
        relevant_memories_key: relevant_memories,
        display_events_key: display_events,
        memory_component_key: memory_component,
        tool_executor_key: tool_executor_component,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
        terminate_key: terminate_component,
    }

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )

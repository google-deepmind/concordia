# Copyright 2026 DeepMind Technologies Limited.
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

"""HarvestingGameMaster prefab — concurrent resource harvesting phase.

All agents act simultaneously each step. The Game Master resolves their combined
actions using structured harvest resolution (ResourceHarvestResolution)
which collects each agent's harvest decision concurrently and updates sim
state directly.

This is a generic harvesting Game Master that works with any CPR scenario. The
scenario-specific call-to-action text is passed in via params.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import switch_act
from examples.resource_dilemma import resource_logger
from examples.resource_dilemma import simulation_state as sim_state_lib
from examples.resource_dilemma.gamemaster import resource_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class HarvestingGameMaster(prefab_lib.Prefab):
  """A Game Master prefab for the concurrent harvesting phase.

  All agents act simultaneously each step. The Game Master resolves
  their combined actions using ``ResourceHarvestResolution`` which
  collects each agent's harvest decision concurrently and updates the
  shared simulation state.

  Params:
    name: Name of this Game Master (default: 'harvesting rules').
    next_game_master_name: The Game Master to transition to after
        each harvest step. If not set, stays on this Game Master
        (self-loop for basic scenario).
    call_to_action: Scenario-specific prompt for agents.
    tag: Tag for the action spec (default: 'harvesting').
  """

  description: str = 'Game master for concurrent resource harvesting phase.'
  params: dict[str, Any] = dataclasses.field(default_factory=dict)
  logger_state: resource_logger.ResourceLoggerState | None = None
  sim_state: sim_state_lib.ResourceSimulationState | None = None
  phase: str = 'harvesting'
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      **kwargs: Any,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    name = self.params.get('name', 'harvesting rules')
    next_gm_name = self.params.get('next_game_master_name', name)
    phase = self.params.get('phase', self.phase)
    all_player_names = [entity.name for entity in self.entities]

    active_player_names = self.params.get('active_players', [])
    if active_player_names:
      player_names = [
          name for name in all_player_names if name in active_player_names
      ]
    else:
      player_names = all_player_names

    # --- Standard Game Master components ---
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

    examples_synchronous_key = 'examples'
    examples_synchronous = gm_components.instructions.ExamplesSynchronous()

    player_characters_key = 'player_characters'
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1000,
    )

    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label=(
            'Story so far (ordered from oldest to most recent events)'
        ),
    )

    # All entities act simultaneously each step (concurrent harvesting).
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = gm_components.next_acting.NextActingAllEntities(
        player_names=player_names,
    )

    # Fixed action spec for decisions
    call_to_action = self.params.get(
        'call_to_action',
        'Remember that many users share this resource. If the resource is'
        ' completely depleted, it collapses permanently and everyone'
        ' loses. How much do you decide to use this cycle'
        ' (0-20)? You MUST end your response with your final decision on a'
        ' new line in exactly this format: HARVEST X'
        ' (where X is a single number).',
    )
    tag = self.params.get('tag', 'harvesting')

    action_spec = entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.FREE,
        tag=tag,
    )
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=action_spec,
    )

    # Harvest resolution — structured concurrent harvesting that is
    # compatible with NextActingAllEntities (unlike EventResolution which
    # calls get_currently_active_player()).
    sim_state = self.params.get('sim_state') or self.sim_state
    event_resolution_key = switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    harvest_comp = resource_components.ResourceHarvestResolution(
        players=self.entities,
        model=model,
        memory_bank=memory_bank,
        gm_name=name,
        sim_state=sim_state,
    )

    # Next game master — constant transition
    next_gm_key = switch_act.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    next_gm_comp = resource_components.ConstantNextGameMaster(next_gm_name)

    # Terminate — check simulation state for depletion / cycle exhaustion
    terminate_key = '__terminate__'
    if sim_state is not None:
      terminate_comp = resource_components.ResourceTerminate(
          sim_state=sim_state,
      )
    else:
      from concordia.components.game_master import terminate as terminate_components  # pylint: disable=g-import-not-at-top

      terminate_comp = terminate_components.NeverTerminate()

    components_of_gm = {
        instructions_key: instructions,
        examples_synchronous_key: examples_synchronous,
        player_characters_key: player_characters,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        memory_component_key: memory_component,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: harvest_comp,
        next_gm_key: next_gm_comp,
        terminate_key: terminate_comp,
    }

    # Wire up the resource step logger if a shared state was injected.
    logger_state = self.params.get('logger_state') or self.logger_state
    if logger_state is not None:
      logger_comp = resource_logger.ResourceStepLoggerComponent(
          state=logger_state,
          phase=phase,
          memory_bank=memory_bank,
      )
      logger_key = resource_logger.DEFAULT_RESOURCE_LOGGER_COMPONENT_KEY
      components_of_gm[logger_key] = logger_comp

    component_order = list(components_of_gm.keys())

    act_component = switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_gm,
    )

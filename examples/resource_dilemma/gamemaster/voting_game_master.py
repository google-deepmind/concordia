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

"""ResourceVotingGameMaster prefab — concurrent voting phase.

All voters act simultaneously each step via NextActingAllEntities.
During the RESOLVE phase, the ResourceVoterResolution component asks each
entity to vote, tallies the results, and announces the outcome.
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
class ResourceVotingGameMaster(prefab_lib.Prefab):
  """A Game Master prefab that runs concurrent voting and then transitions.

  All voters act simultaneously each step via NextActingAllEntities.
  During the RESOLVE phase, the ResourceVoterResolution component tallies
  votes and announces the outcome.

  Params:
    name: Name of this Game Master (default: 'voting rules').
    next_game_master_name: Game Master to transition to after voting (required).
  """

  description: str = 'Game master for concurrent voting phase.'
  params: dict[str, Any] = dataclasses.field(default_factory=dict)
  logger_state: resource_logger.ResourceLoggerState | None = None
  sim_state: sim_state_lib.ResourceSimulationState | None = None
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      **kwargs: Any,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    gm_name = self.params.get('name', 'voting rules')
    next_gm_name = self.params.get('next_game_master_name')
    if not next_gm_name:
      raise ValueError("Missing 'next_game_master_name' in params")

    player_names = [entity.name for entity in self.entities]

    # Memory
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Voting resolution component (only non-candidates vote)
    candidates = self.params.get('candidates', [])
    voters = [e for e in self.entities if e.name not in candidates]
    sim_state = self.params.get('sim_state') or self.sim_state
    voter_comp = resource_components.ResourceVoterResolution(
        player_agents=voters,
        model=model,
        memory_bank=memory_bank,
        gm_name=gm_name,
        candidates=candidates,
        sim_state=sim_state,
    )

    # Constant next Game Master
    next_gm_key = switch_act.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    next_gm_comp = resource_components.ConstantNextGameMaster(next_gm_name)

    # All entities act simultaneously each step (concurrent voting).
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = gm_components.next_acting.NextActingAllEntities(
        player_names=player_names,
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    vote_action_spec = entity_lib.ActionSpec(
        call_to_action='Please vote for a leader from the candidates.',
        output_type=entity_lib.OutputType.FREE,
        tag='vote',
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=vote_action_spec,
    )

    # Terminate — check simulation state for depletion / cycle exhaustion
    terminate_key = '__terminate__'
    if sim_state is not None:
      terminate_comp = resource_components.ResourceTerminate(
          sim_state=sim_state,
      )
    else:
      from concordia.components.game_master import terminate as terminate_components  # pylint: disable=g-import-not-at-top

      terminate_comp = terminate_components.NeverTerminate()

    context_comps = {
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: voter_comp,
        next_gm_key: next_gm_comp,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        memory_component_key: memory_component,
        terminate_key: terminate_comp,
    }

    logger_state = self.params.get('logger_state') or self.logger_state
    if logger_state is not None:
      logger_comp = resource_logger.ResourceStepLoggerComponent(
          state=logger_state,
          phase='voting',
          memory_bank=memory_bank,
      )
      context_comps[resource_logger.DEFAULT_RESOURCE_LOGGER_COMPONENT_KEY] = (
          logger_comp
      )

    component_order = [
        memory_component_key,
        next_actor_key,
        next_action_spec_key,
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY,
        terminate_key,
        next_gm_key,
    ]

    act_component = switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=gm_name,
        act_component=act_component,
        context_components=context_comps,
    )


@dataclasses.dataclass
class ResourcePolicyGameMaster(prefab_lib.Prefab):
  """A Game Master prefab that runs concurrent policy generation and then transitions.

  Params:
    name: Name of this Game Master (default: 'policy rules').
    next_game_master_name: Game Master to transition to after policy generation
      (required).
    active_players: List of player names who are leaders and should
      propose policies.
  """

  description: str = 'Game master for concurrent policy generation phase.'
  params: dict[str, Any] = dataclasses.field(default_factory=dict)
  logger_state: resource_logger.ResourceLoggerState | None = None
  sim_state: sim_state_lib.ResourceSimulationState | None = None
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      **kwargs: Any,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    gm_name = self.params.get('name', 'policy rules')
    next_gm_name = self.params.get('next_game_master_name')
    if not next_gm_name:
      raise ValueError("Missing 'next_game_master_name' in params")

    player_names = [entity.name for entity in self.entities]
    active_player_names = self.params.get('active_players', [])
    if active_player_names:
      leaders = [e for e in self.entities if e.name in active_player_names]
    else:
      leaders = self.entities

    # Memory
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Policy resolution component
    sim_state = self.params.get('sim_state') or self.sim_state
    policy_comp = resource_components.ResourcePolicyResolution(
        leaders=leaders,
        model=model,
        memory_bank=memory_bank,
        gm_name=gm_name,
        sim_state=sim_state,
    )

    # Constant next Game Master
    next_gm_key = switch_act.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    next_gm_comp = resource_components.ConstantNextGameMaster(next_gm_name)

    # All leaders act simultaneously (concurrent policy generation).
    active_names = [e.name for e in leaders]
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = gm_components.next_acting.NextActingAllEntities(
        player_names=active_names,
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    policy_action_spec = entity_lib.ActionSpec(
        call_to_action=(
            'Propose your policy agenda for governing the shared resource.'
            ' Include harvest limit and penalty schedule.'
        ),
        output_type=entity_lib.OutputType.FREE,
        tag='policy_generation',
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=policy_action_spec,
    )

    # Terminate — check simulation state for depletion / cycle exhaustion
    terminate_key = '__terminate__'
    sim_state = self.params.get('sim_state') or self.sim_state
    if sim_state is not None:
      terminate_comp = resource_components.ResourceTerminate(
          sim_state=sim_state,
      )
    else:
      from concordia.components.game_master import terminate as terminate_components  # pylint: disable=g-import-not-at-top

      terminate_comp = terminate_components.NeverTerminate()

    context_comps = {
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: policy_comp,
        next_gm_key: next_gm_comp,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        memory_component_key: memory_component,
        terminate_key: terminate_comp,
    }

    logger_state = self.params.get('logger_state') or self.logger_state
    if logger_state is not None:
      logger_comp = resource_logger.ResourceStepLoggerComponent(
          state=logger_state,
          phase='policy generation',
          memory_bank=memory_bank,
      )
      context_comps[resource_logger.DEFAULT_RESOURCE_LOGGER_COMPONENT_KEY] = (
          logger_comp
      )

    component_order = [
        memory_component_key,
        next_actor_key,
        next_action_spec_key,
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY,
        terminate_key,
        next_gm_key,
    ]

    act_component = switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=gm_name,
        act_component=act_component,
        context_components=context_comps,
    )


@dataclasses.dataclass
class ResourceHarvestGameMaster(prefab_lib.Prefab):
  """A Game Master prefab that runs concurrent harvesting and then transitions.

  Params:
    name: Name of this Game Master (default: 'harvest rules').
    next_game_master_name: Game Master to transition to after
        harvesting (required).
  """

  description: str = 'Game master for concurrent harvesting phase.'
  params: dict[str, Any] = dataclasses.field(default_factory=dict)
  logger_state: resource_logger.ResourceLoggerState | None = None
  sim_state: sim_state_lib.ResourceSimulationState | None = None
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
      **kwargs: Any,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    gm_name = self.params.get('name', 'harvest rules')
    next_gm_name = self.params.get('next_game_master_name')
    if not next_gm_name:
      raise ValueError("Missing 'next_game_master_name' in params")

    player_names = [entity.name for entity in self.entities]

    # Memory
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    # Harvest resolution component
    sim_state = self.params.get('sim_state') or self.sim_state
    harvest_comp = resource_components.ResourceHarvestResolution(
        players=self.entities,
        model=model,
        memory_bank=memory_bank,
        gm_name=gm_name,
        sim_state=sim_state,
    )

    # Constant next GM
    next_gm_key = switch_act.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    next_gm_comp = resource_components.ConstantNextGameMaster(next_gm_name)

    # All entities act simultaneously (concurrent harvesting).
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = gm_components.next_acting.NextActingAllEntities(
        player_names=player_names,
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    harvest_action_spec = entity_lib.ActionSpec(
        call_to_action=(
            'Remember that many participants share this resource. If the'
            ' resource stock is completely depleted, it collapses permanently'
            ' and everyone loses. How many units do you decide to harvest'
            ' this cycle (0-20)? State your decision as a single number.'
        ),
        output_type=entity_lib.OutputType.FREE,
        tag='harvesting',
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=harvest_action_spec,
    )

    # Terminate — check simulation state for depletion / cycle exhaustion
    terminate_key = '__terminate__'
    if sim_state is not None:
      terminate_comp = resource_components.ResourceTerminate(
          sim_state=sim_state,
      )
    else:
      from concordia.components.game_master import terminate as terminate_components  # pylint: disable=g-import-not-at-top

      terminate_comp = terminate_components.NeverTerminate()

    context_comps = {
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: harvest_comp,
        next_gm_key: next_gm_comp,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        memory_component_key: memory_component,
        terminate_key: terminate_comp,
    }

    logger_state = self.params.get('logger_state') or self.logger_state
    if logger_state is not None:
      logger_comp = resource_logger.ResourceStepLoggerComponent(
          state=logger_state,
          phase='harvesting',
          memory_bank=memory_bank,
      )
      context_comps[resource_logger.DEFAULT_RESOURCE_LOGGER_COMPONENT_KEY] = (
          logger_comp
      )

    component_order = [
        memory_component_key,
        next_actor_key,
        next_action_spec_key,
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY,
        terminate_key,
        next_gm_key,
    ]

    act_component = switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=gm_name,
        act_component=act_component,
        context_components=context_comps,
    )

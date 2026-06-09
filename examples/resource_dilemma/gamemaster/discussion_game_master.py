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

"""DiscussionGameMaster prefab — sequential community discussion.

The Game Master picks who speaks next (game_master_choice acting
order). Entities speak freely using speech action specs. The Game
Master decides when discussion is sufficient and transitions to the
next phase (typically voting or harvesting).

This is a generic discussion Game Master that works with any CPR scenario.
The discussion topic text can be customised via params.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate as terminate_components
from examples.resource_dilemma import resource_logger
from examples.resource_dilemma import simulation_state as sim_state_lib
from examples.resource_dilemma.gamemaster import resource_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class DiscussionGameMaster(prefab_lib.Prefab):
  """A Game Master prefab for sequential community discussion.

  The Game Master picks who speaks next (game_master_choice acting
  order). Entities speak freely using speech action specs. The Game
  Master decides when discussion is sufficient and transitions to the
  next phase.

  Params:
    name: Name of this Game Master (default: 'community meeting').
    next_game_master_name: Game Master to transition to when discussion ends.
    acting_order: One of 'game_master_choice', 'fixed', 'random'.
    turn_limit: Maximum discussion turns before auto-transitioning
    (default: 10).
    discussion_topic: Custom discussion topic text (optional).
  """

  description: str = 'Game master for sequential community discussion.'
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
    name = self.params.get('name', 'community meeting')
    next_gm_name = self.params.get('next_game_master_name', 'harvesting rules')
    acting_order = self.params.get('acting_order', 'game_master_choice')
    player_names = [entity.name for entity in self.entities]

    # --- Standard components ---
    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

    default_topic = (
        'The community members are gathered for a meeting. Everyone can'
        ' hear each other. They should discuss the events of the past'
        ' cycle, recall any active policies, identify who broke the rules'
        ' and who followed them, discuss whether the resource was overused,'
        ' and suggest ways to improve. They should also respond to each'
        ' other in sequence and address any relevant statements already'
        ' made by others. Proposals for new rules must specify a usage'
        ' limit and/or a penalty schedule. Use memories of past'
        ' interactions and observations to support your points.'
    )
    discussion_topic_text = self.params.get('discussion_topic', default_topic)

    discussion_topic_key = 'discussion_topic'
    discussion_topic = actor_components.constant.Constant(
        discussion_topic_text,
        pre_act_label='Current Objective',
    )

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
        pre_act_label='Community Discussion (speak to contribute):',
    )

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components=[
            observation_component_key,
            display_events_key,
        ],
    )

    send_events_key = '__send_events_to_players__'
    send_events = gm_components.event_resolution.SendEventToRelevantPlayers(
        model=model,
        player_names=player_names,
        make_observation_component_key=make_observation_key,
    )

    # Acting order — who speaks next
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    if acting_order == 'fixed':
      next_actor = gm_components.next_acting.NextActingInFixedOrder(
          sequence=player_names,
      )
    elif acting_order == 'game_master_choice':
      next_actor = gm_components.next_acting.NextActing(
          model=model,
          player_names=player_names,
          components=[
              instructions_key,
              discussion_topic_key,
              player_characters_key,
              display_events_key,
          ],
      )
    elif acting_order == 'random':
      next_actor = gm_components.next_acting.NextActingInRandomOrder(
          player_names=player_names,
      )
    else:
      raise ValueError(f'Unsupported acting order: {acting_order}')

    # Speech action spec
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=entity_lib.DEFAULT_SPEECH_ACTION_SPEC,
    )

    # Event resolution — pass-through (speech doesn't need transformation)
    identity_step = event_resolution_components.RemoveSpecificText(
        substring_to_remove='Putative event to resolve:  '
    )
    event_resolution_key = switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=(identity_step,),
        notify_observers=False,
    )

    # Next game master — LLM decides when discussion is complete
    next_gm_key = switch_act.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    map_gm_names_to_choices = {
        next_gm_name: 'Yes, end discussion and proceed.',
        name: 'No, the community still needs to discuss.',
    }
    turn_limit = self.params.get('turn_limit', 10)
    sim_state = self.params.get('sim_state') or self.sim_state
    next_gm = resource_components.TurnLimitedNextGameMaster(
        limit=turn_limit,
        model=model,
        call_to_action=(
            '**DISCUSSION CHECK:** The community meeting is for deliberation'
            ' about resource management rules. Before moving on, ensure:'
            '\n- Have members had a chance to share their views?'
            '\n- Have any proposals been clearly stated?'
            '\n- Have other members responded to the proposals?'
            '\n- Have there been at least 2-3 meaningful exchanges between'
            ' different members?'
            '\nDo NOT end discussion prematurely. Only proceed if the'
            ' discussion feels naturally concluded or a clear proposal has'
            ' emerged. Is the community ready to proceed?'
        ),
        map_game_master_names_to_choices=map_gm_names_to_choices,
        components=[
            instructions_key,
            discussion_topic_key,
            player_characters_key,
            display_events_key,
        ],
    )

    # Terminate — check simulation state for depletion / cycle exhaustion
    terminate_key = '__terminate__'
    if sim_state is not None:
      terminate_comp = resource_components.ResourceTerminate(
          sim_state=sim_state,
          phase='discussion',
      )
    else:
      terminate_comp = terminate_components.NeverTerminate()

    components_of_gm = {
        instructions_key: instructions,
        discussion_topic_key: discussion_topic,
        examples_synchronous_key: examples_synchronous,
        player_characters_key: player_characters,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        memory_component_key: memory_component,
        make_observation_key: make_observation,
        send_events_key: send_events,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
        next_gm_key: next_gm,
        terminate_key: terminate_comp,
    }

    logger_state = self.params.get('logger_state') or self.logger_state
    if logger_state is not None:
      logger_comp = resource_logger.ResourceStepLoggerComponent(
          state=logger_state,
          phase='discussion',
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

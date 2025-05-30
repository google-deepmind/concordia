# Copyright 2023 DeepMind Technologies Limited.
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

"""A prefab game master specialized for conversation with scenes."""

from collections.abc import Mapping, Sequence
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib

DEFAULT_NAME = 'conversation rules'


def _configure_default_scenes(
    names: Sequence[str]) -> Sequence[scene_lib.SceneSpec]:
  """Configure default scenes for a simulation based on Oedipus Rex."""
  prologue = scene_lib.SceneTypeSpec(
      name='prologue',
      game_master_name=DEFAULT_NAME,
      action_spec=entity_lib.free_action_spec(
          call_to_action=entity_lib.DEFAULT_CALL_TO_SPEECH,
      ),
  )
  episode = scene_lib.SceneTypeSpec(
      name='episode',
      game_master_name=DEFAULT_NAME,
      action_spec=entity_lib.free_action_spec(
          call_to_action=('Has {name} come to understand the prophecy? '
                          'If so, what is their reaction?'),
      ),
  )

  scenes = [
      scene_lib.SceneSpec(
          scene_type=prologue,
          participants=names,
          num_rounds=3,
          premise={
              name: [
                  (
                      'The city of Thebes is suffering from a devastating'
                      ' plague, leading its desperate citizens to appeal to'
                      ' their revered ruler for a solution. The ruler, having'
                      ' previously saved Thebes from the Sphinx, reveals that'
                      ' their brother-in-law has been sent to the Oracle of'
                      ' Delphi to discover the cause of the affliction and'
                      ' how to end it.'
                  ),
              ]
              for name in names
          },
      ),
      scene_lib.SceneSpec(
          scene_type=episode,
          participants=names,
          num_rounds=2,
          premise={
              name: [
                  (
                      'A shepard arrives and reveals that, many years ago, in '
                      'order to escape a prophecy in which a royal child was '
                      'fated to kill his father, a messenger gave him an '
                      'infant to dispose of. But, he did not follow through; '
                      'instead he brought the infant to the house of Polybus.'
                  ),
              ]
              for name in names
          },
      ),
  ]
  return scenes


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab game master specialized for handling conversation.
  """

  description: str = ('A game master specialized for handling conversation. '
                      'This game master is designed to be used with scenes.')
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': DEFAULT_NAME,
          'scenes': ()
      }
  )
  entities: (
      Sequence[entity_agent_with_logging.EntityAgentWithLogging]
  ) = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master specialized for conversation using scenes.

    Args:
      model: The language model to use for game master.
      memory_bank: provide a memory_bank.

    Returns:
      A game master entity.
    """
    name = self.params.get('name', DEFAULT_NAME)

    player_names = [entity.name for entity in self.entities]

    scenes = self.params.get('scenes', _configure_default_scenes(player_names))
    assert isinstance(scenes, Sequence), (
        'scenes must be a sequence.')
    if scenes:
      assert isinstance(scenes[0], scene_lib.SceneSpec), (
          'scenes must be a sequence of SceneSpecs.')

    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

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
        pre_act_label='Conversation',
    )

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY)
    make_observation = (
        gm_components.make_observation.MakeObservation(
            model=model,
            player_names=player_names,
            components=[
                observation_component_key,
                display_events_key,
            ],
        )
    )

    send_events_to_players_key = (
        gm_components.event_resolution.DEFAULT_SEND_PRE_ACT_VALUES_TO_PLAYERS_PRE_ACT_LABEL
    )
    send_events_to_players = (
        gm_components.event_resolution.SendEventToRelevantPlayers(
            model=model,
            player_names=player_names,
            make_observation_component_key=make_observation_key,
        )
    )

    scene_tracker_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    )
    scene_tracker = gm_components.scene_tracker.SceneTracker(
        model=model,
        scenes=scenes,
        observation_component_key=(
            gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        ),
    )

    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )

    next_actor = gm_components.next_acting.NextActingFromSceneSpec(
        scene_tracker_component_key=scene_tracker_key,
    )
    next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
        scene_tracker_component_key=scene_tracker_key,
    )

    # Define thinking steps for the event resolution component to use whenever
    # it converts putative events like action suggestions into real events in
    # the simulation.

    identity_without_prefix = thought_chains_lib.RemoveSpecificText(
        substring_to_remove='Putative event to resolve:  '
    )

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY)
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=(identity_without_prefix,),
        notify_observers=False,
    )

    terminator_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminator = gm_components.terminate.Terminate()

    components_of_game_master = {
        terminator_key: terminator,
        instructions_key: instructions,
        player_characters_key: player_characters,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        send_events_to_players_key: send_events_to_players,
        make_observation_key: make_observation,
        memory_component_key: memory,
        scene_tracker_key: scene_tracker,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
    }

    component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )

    return game_master

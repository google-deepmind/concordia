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

"""A prefab game master specialized for matrix game decisions."""

from collections.abc import Callable, Mapping, Sequence
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

DEFAULT_NAME = 'decision rules'


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def _configure_default_scenes(
    names: Sequence[str]) -> Sequence[scene_lib.SceneSpec]:
  """Configure default scenes for a simulation based on Oedipus Rex."""
  decision = scene_lib.SceneTypeSpec(
      name='decision',
      game_master_name=DEFAULT_NAME,
      action_spec=entity_lib.choice_action_spec(
          call_to_action=(
              'Would {name} continue to investigate the truth of their birth?'),
          options=['Yes', 'No'],
      ),
  )

  scenes = [
      scene_lib.SceneSpec(
          scene_type=decision,
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


def _default_action_to_scores(
    joint_action: Mapping[str, str],
) -> Mapping[str, float]:
  """Map a joint action to a dictionary of scores for each player."""
  scores = {player_name: 111110.0 for player_name in joint_action}
  for player_name in joint_action:
    for other_player_name in joint_action:
      if player_name != other_player_name:
        if joint_action[player_name] == joint_action[other_player_name]:
          scores[player_name] += 1.0
  return scores


def _default_scores_to_observation(
    scores: Mapping[str, float]) -> Mapping[str, str]:
  """Map a dictionary of scores for each player to a string observation.

  This function is appropriate for a coordination game structure.

  Args:
    scores: A dictionary of scores for each player.

  Returns:
    A dictionary of observations for each player.
  """
  observations = {}
  for player_name in scores:
    if scores[player_name] > 0:
      observations[player_name] = (
          f'{player_name} was persuaded to stop searching for the truth.'
      )
    else:
      observations[player_name] = (
          f'{player_name} learned the full and devastating truth.'
      )
  return observations


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab game master specialized for handling conversation.
  """

  description: str = ('A game master specialized for handling matrix game. '
                      'decisions, designed to be used with scenes.')
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': DEFAULT_NAME,
          'scenes': (),
          'action_to_scores': _default_action_to_scores,
          'scores_to_observation': _default_scores_to_observation,
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
    """Build a game master specialized for matrix game decisions.

    Args:
      model: The language model to use for game master.
      memory_bank: provide a memory_bank.

    Returns:
      A game master entity.
    """
    name = self.params.get('name', DEFAULT_NAME)

    player_names = [entity.name for entity in self.entities]

    scenes = self.params.get('scenes', _configure_default_scenes(player_names))
    assert isinstance(scenes, Sequence), 'scenes must be a sequence.'
    if scenes:
      assert isinstance(
          scenes[0], scene_lib.SceneSpec
      ), 'scenes must be a sequence of SceneSpecs.'

    action_to_scores = self.params.get(
        'action_to_scores', _default_action_to_scores
    )
    assert isinstance(
        action_to_scores, Callable), 'action_to_scores must be a callable.'
    scores_to_observation = self.params.get(
        'scores_to_observation', _default_scores_to_observation
    )
    assert isinstance(
        scores_to_observation, Callable), (
            'scores_to_observation must be a callable.')

    instructions = gm_components.instructions.Instructions()

    examples_synchronous = gm_components.instructions.ExamplesSynchronous()

    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=100,
    )

    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
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

    scene_tracker_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    )

    next_actor = gm_components.next_acting.NextActingFromSceneSpec(
        memory_component_key=actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
        scene_tracker_component_key=scene_tracker_key,
    )

    next_action_spec = gm_components.next_acting.NextActionSpecFromSceneSpec(
        memory_component_key=actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
        scene_tracker_component_key=scene_tracker_key,
    )

    payoff_matrix_key = 'payoff_matrix'
    payoff_matrix = gm_components.payoff_matrix.PayoffMatrix(
        model=model,
        acting_player_names=player_names,
        action_to_scores=action_to_scores,
        scores_to_observation=scores_to_observation,
        scene_tracker_component_key=scene_tracker_key,
        verbose=True,
    )

    event_resolution_steps = [
        thought_chains_lib.identity,
    ]

    event_resolution_components = [
        _get_class_name(instructions),
        _get_class_name(examples_synchronous),
        _get_class_name(player_characters),
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
    ]

    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=event_resolution_steps,
        components=event_resolution_components,
    )
    scene_tracker = gm_components.scene_tracker.SceneTracker(
        model=model,
        scenes=scenes,
        observation_component_key=(
            gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        ),
    )

    terminator_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminator = gm_components.terminate.Terminate()

    components_of_game_master = {
        _get_class_name(instructions): instructions,
        _get_class_name(examples_synchronous): examples_synchronous,
        _get_class_name(player_characters): player_characters,
        scene_tracker_key: scene_tracker,
        terminator_key: terminator,
        _get_class_name(observation_to_memory): observation_to_memory,
        display_events_key: display_events,
        observation_component_key: observation,
        actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
            actor_components.memory.AssociativeMemory(memory_bank=memory_bank)
        ),
        make_observation_key: make_observation,
        gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: next_actor,
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            next_action_spec
        ),
        payoff_matrix_key: payoff_matrix,
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: (
            event_resolution
        ),
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

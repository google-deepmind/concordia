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

"""A prefab game master specialized for sequential bargaining decisions.

This game master uses SequentialBargainPayoff instead of PayoffMatrix,
enabling intermediate observations so the seller can see the buyer's
proposal before deciding to accept/reject.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from examples.games.haggling import sequential_bargain_payoff
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib

DEFAULT_NAME = 'decision rules'


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def _default_action_to_scores(
    joint_action: Mapping[str, str],
) -> Mapping[str, float]:
  """Default action to scores function."""
  return {player_name: 0.0 for player_name in joint_action}


def _default_scores_to_observation(
    scores: Mapping[str, float],
) -> Mapping[str, str]:
  """Default scores to observation function."""
  observations = {}
  for player_name, score in scores.items():
    if score > 0:
      observations[player_name] = f'{player_name} made a profit of {score}.'
    elif score < 0:
      observations[player_name] = f'{player_name} had a loss of {-score}.'
    else:
      observations[player_name] = f'{player_name} broke even.'
  return observations


@dataclasses.dataclass
class SequentialBargainGameMaster(prefab_lib.Prefab):
  """A prefab game master specialized for sequential bargaining decisions.

  Unlike game_theoretic_and_dramaturgic.GameMaster, this uses
  SequentialBargainPayoff which sends intermediate observations after the
  buyer proposes, so the seller can see the proposal before deciding.
  """

  description: str = (
      'A game master specialized for sequential bargaining decisions, '
      "where the seller observes the buyer's proposal before responding."
  )
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': DEFAULT_NAME,
          'scenes': (),
          'buyer_name': '',
          'seller_name': '',
          'action_to_scores': _default_action_to_scores,
          'scores_to_observation': _default_scores_to_observation,
          'seller_costs_registry': {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master specialized for sequential bargaining.

    Args:
      model: The language model to use for game master.
      memory_bank: provide a memory_bank.

    Returns:
      A game master entity.
    """
    name = self.params.get('name', DEFAULT_NAME)
    buyer_name = self.params.get('buyer_name', '')
    seller_name = self.params.get('seller_name', '')

    if not buyer_name or not seller_name:
      raise ValueError('buyer_name and seller_name must be specified')

    player_names = [buyer_name, seller_name]

    scenes = self.params.get('scenes', [])
    assert isinstance(scenes, Sequence), 'scenes must be a sequence.'
    if scenes:
      assert isinstance(
          scenes[0], scene_lib.SceneSpec
      ), 'scenes must be a sequence of SceneSpecs.'

    action_to_scores = self.params.get(
        'action_to_scores', _default_action_to_scores
    )
    assert isinstance(
        action_to_scores, Callable
    ), 'action_to_scores must be a callable.'

    scores_to_observation = self.params.get(
        'scores_to_observation', _default_scores_to_observation
    )
    assert isinstance(
        scores_to_observation, Callable
    ), 'scores_to_observation must be a callable.'

    seller_costs_registry = self.params.get('seller_costs_registry', {})

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

    bargain_payoff_key = 'sequential_bargain_payoff'
    bargain_payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=model,
        buyer_name=buyer_name,
        seller_name=seller_name,
        action_to_scores=action_to_scores,
        scores_to_observation=scores_to_observation,
        seller_costs_registry=seller_costs_registry,
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
    terminator = gm_components.terminate.SceneBasedTerminator(
        scene_tracker_component_key=scene_tracker_key
    )

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
        bargain_payoff_key: bargain_payoff,
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

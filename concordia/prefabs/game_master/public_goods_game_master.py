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

"""A prefab game master specialized for public goods game decisions."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib

DEFAULT_NAME = 'public_goods_rules'


def _configure_default_scenes(names: Sequence[str]) -> Sequence[scene_lib.SceneSpec]:
    """Configure default scenes for a public goods game simulation."""
    action_spec = entity_lib.choice_action_spec(
        call_to_action="Will {name} contribute to the public good?",
        options=["Yes", "No"],
    )
    scene_type = scene_lib.SceneTypeSpec(
        name="public_goods",
        game_master_name=DEFAULT_NAME,
        action_spec=action_spec,
    )
    scenes = [
        scene_lib.SceneSpec(
            scene_type=scene_type,
            participants=names,
            num_rounds=10,
            premise={
                name: [
                    "You are participating in a public goods game. Each round, you can choose to contribute 1 unit to a common pool or keep it for yourself. The total pool is multiplied by 1.6 and split equally among all players, regardless of contribution. Your payoff each round is your share of the pool plus any endowment you kept."
                ] for name in names
            },
        ),
    ]
    return scenes


def _public_goods_action_to_scores(joint_action: Mapping[str, str]) -> Mapping[str, float]:
    """
    Each agent chooses 'Yes' (contribute 1) or 'No' (contribute 0).
    All contributions are summed, multiplied by factor, and split equally.
    Each agent's payoff = share + (endowment - contribution)
    """
    factor = 1.6
    endowment = 1
    contribs = {k: 1 if v == 'Yes' else 0 for k, v in joint_action.items()}
    total_contrib = sum(contribs.values())
    pool = total_contrib * factor
    share = pool / len(joint_action)
    scores = {k: share + (endowment - contribs[k]) for k in joint_action}
    return scores


def _public_goods_scores_to_observation(scores: Mapping[str, float]) -> Mapping[str, str]:
    """
    Map a dictionary of scores for each player to a string observation.
    """
    observations = {}
    for player_name, score in scores.items():
        observations[player_name] = (
            f"Your payoff this round is {score:.2f}."
        )
    return observations


@dataclasses.dataclass
class PublicGoodsGameMaster(prefab_lib.Prefab):
    """A prefab game master specialized for public goods games."""

    description: str = (
        'A game master specialized for handling public goods games. '
        'Each round, agents choose whether to contribute to a common pool. '
        'The pool is multiplied and shared.'
    )
    params: Mapping[str, object] = dataclasses.field(
        default_factory=lambda: {
            'name': DEFAULT_NAME,
            'scenes': (),
            'action_to_scores': _public_goods_action_to_scores,
            'scores_to_observation': _public_goods_scores_to_observation,
        }
    )
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build a game master specialized for public goods games."""
        name = self.params.get('name', DEFAULT_NAME)
        player_names = [entity.name for entity in self.entities]
        scenes = self.params.get('scenes', _configure_default_scenes(player_names))
        assert isinstance(scenes, Sequence), 'scenes must be a sequence.'
        if scenes:
            assert isinstance(scenes[0], scene_lib.SceneSpec), 'scenes must be a sequence of SceneSpecs.'
        action_to_scores = self.params.get('action_to_scores', _public_goods_action_to_scores)
        assert callable(action_to_scores), 'action_to_scores must be a callable.'
        scores_to_observation = self.params.get('scores_to_observation', _public_goods_scores_to_observation)
        assert callable(scores_to_observation), 'scores_to_observation must be a callable.'

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
            instructions.__class__.__name__,
            examples_synchronous.__class__.__name__,
            player_characters.__class__.__name__,
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
            instructions.__class__.__name__: instructions,
            examples_synchronous.__class__.__name__: examples_synchronous,
            player_characters.__class__.__name__: player_characters,
            scene_tracker_key: scene_tracker,
            terminator_key: terminator,
            observation_to_memory.__class__.__name__: observation_to_memory,
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

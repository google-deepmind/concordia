# Copyright 2025 DeepMind Technologies Limited.
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

"""Prefab for a generic psychology experiment Game Master."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib


def _get_class_name(object_: object) -> str:
  """Returns the class name of an object."""
  return object_.__class__.__name__


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A generic Game Master prefab for psychology experiments.

  This GM is configured by providing custom classes for experiment-specific
  observation generation and action specification.
  """

  description: str = (
      "A generic Game Master that administers a psychology experiment defined "
      "by custom observation and action specification components."
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          "name": "ExperimenterGM",
          "scenes": (),
          "experiment_component_class": None,
          "experiment_component_init_kwargs": {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build the generic psychology experiment Game Master.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity representing the Game Master.

    Raises:
      ValueError: If required experiment component classes are not provided.
    """
    name = self.params["name"]

    experiment_component_class = self.params.get("experiment_component_class")
    experiment_component_init_kwargs = self.params.get(
        "experiment_component_init_kwargs", {}
    )

    if not experiment_component_class:
      raise ValueError("experiment_component_class parameter must be provided.")

    player_names = [entity.name for entity in self.entities]

    scenes = self.params["scenes"]
    assert isinstance(scenes, Sequence), "scenes must be a sequence."
    if scenes:
      assert isinstance(
          scenes[0], scene_lib.SceneSpec
      ), "scenes must be a sequence of SceneSpecs."

    # Common GM components
    instructions = gm_components.instructions.Instructions()
    examples_synchronous = gm_components.instructions.ExamplesSynchronous()
    player_characters_key = "player_characters"
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
    display_events_key = "display_events"
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
    )
    terminator_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminator = gm_components.terminate.Terminate()

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

    # Component for determining the next acting player
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY

    next_actor = gm_components.next_acting.NextActingFromSceneSpec(
        memory_component_key=actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
        scene_tracker_component_key=scene_tracker_key,
    )

    # Instantiate custom experiment components
    experiment_component = experiment_component_class(
        acting_player_names=player_names, **experiment_component_init_kwargs
    )

    # Event resolution
    event_resolution_steps = [thought_chains_lib.identity]
    event_resolution_components = [
        _get_class_name(instructions),
        _get_class_name(examples_synchronous),
        player_characters_key,
    ]
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=event_resolution_steps,
        components=event_resolution_components,
    )

    # Assemble all components for the Game Master
    components_of_game_master = {
        _get_class_name(instructions): instructions,
        _get_class_name(examples_synchronous): examples_synchronous,
        player_characters_key: player_characters,
        scene_tracker_key: scene_tracker,
        terminator_key: terminator,
        _get_class_name(observation_to_memory): observation_to_memory,
        display_events_key: display_events,
        observation_component_key: observation,
        actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: (
            actor_components.memory.AssociativeMemory(memory_bank=memory_bank)
        ),
        # Use the custom experiment component for making observations
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            experiment_component
        ),
        next_actor_key: next_actor,
        # Use the custom action spec provider for the next action spec
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            experiment_component
        ),
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: (
            event_resolution
        ),
    }

    component_order = list(components_of_game_master.keys())

    # The SwitchAct component directs the GM's actions
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


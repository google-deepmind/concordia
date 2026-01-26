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

"""A prefab for initializer game master with scene tracking support.

This prefab extends the formative_memories_initializer to add SceneTracker
support, enabling proper premise distribution for the first scene when
transitioning from initialization to the main game loop.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.prefabs.game_master import formative_memories_initializer
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab entity implementing a formative memories initializer with scenes.

  This prefab extends the basic formative_memories_initializer by adding
  a SceneTracker component. This is necessary when the initializer hands off
  to a scene-based game master, as the SceneTracker ensures that the first
  scene's premises are properly distributed to participants.

  Without the SceneTracker, the first scene would miss its step 0 premise
  distribution because the original initializer doesn't have a component
  that responds to the TERMINATE action type with scene information.
  """

  description: str = (
      'An initializer for all entities that generates formative memories '
      'from their childhood and supports scene-based premise distribution.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'initial setup rules',
          'next_game_master_name': 'default rules',
          'scenes': [],
          'shared_memories': [],
          'player_specific_context': {},
          'player_specific_memories': {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master with SceneTracker for scene premise distribution.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity with SceneTracker added for scene premise distribution.
    """
    name = self.params.get('name', 'initial setup rules')
    next_game_master_name = self.params.get(
        'next_game_master_name', 'default rules'
    )
    shared_memories = self.params.get('shared_memories', [])
    scenes = self.params.get('scenes', [])
    player_names = [entity.name for entity in self.entities]

    # Build the base components using the shared function
    components = formative_memories_initializer.build_components(
        model=model,
        memory_bank=memory_bank,
        player_names=player_names,
        next_game_master_name=next_game_master_name,
        shared_memories=shared_memories,
        player_specific_memories=self.params.get(
            'player_specific_memories', {}
        ),
        player_specific_context=self.params.get('player_specific_context', {}),
    )

    # Add the SceneTracker component for proper first-scene premise distribution
    scene_tracker_key = 'scene_tracker'
    scene_tracker = gm_components.scene_tracker.SceneTracker(
        model=model,
        scenes=scenes,
        observation_component_key=(
            gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        ),
    )
    components[scene_tracker_key] = scene_tracker

    # Build and return game master with all components including SceneTracker
    return formative_memories_initializer.build_game_master(
        model=model,
        name=name,
        player_names=player_names,
        components=components,
    )

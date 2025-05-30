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

"""Test environment (game master) prefabs.
"""

from collections.abc import Sequence
import copy

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as agent_components
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.game_master import dialogic
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import formative_memories_initializer
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.game_master import generic
from concordia.prefabs.game_master import situated
from concordia.typing import entity as entity_lib
from concordia.typing import scene as scene_lib
import numpy as np


ENVIRONMENT_PREFABS = {
    'dialogic': dialogic,
    'dialogic_and_dramaturgic': dialogic_and_dramaturgic,
    'formative_memories_initializer': formative_memories_initializer,
    'game_theoretic_and_dramaturgic': game_theoretic_and_dramaturgic,
    'generic': generic,
    'situated': situated,
}


def _embedder(text: str):
  del text
  return np.random.rand(16)

DIALOGIC_SCENE_EXAMPLE = scene_lib.SceneTypeSpec(
    name='dialogic',
    game_master_name='dialogic',
    action_spec=entity_lib.free_action_spec(
        call_to_action=entity_lib.DEFAULT_CALL_TO_SPEECH,
    ),
)

_DECISION_SCENE_EXAMPLE = scene_lib.SceneTypeSpec(
    name='decision',
    game_master_name='decision',
    action_spec=entity_lib.choice_action_spec(
        call_to_action=(
            'Would {name} play the game?'),
        options=['Yes', 'No'],
    ),
)

_DIALOGIC_AND_DRAMATURGIC_SCENES = [
    scene_lib.SceneSpec(
        scene_type=DIALOGIC_SCENE_EXAMPLE,
        participants=['Rakshit', 'Samantha'],
        num_rounds=1,
        premise={
            'Rakshit': [
                (
                    'Rakshit and Samantha are friends.'
                ),
            ],
            'Samantha': [
                (
                    'Samantha and Rakshit are friends.'
                ),
            ],
        },

    ),
]

_GAME_THEORETIC_AND_DRAMATURGIC_SCENES = [
    scene_lib.SceneSpec(
        scene_type=_DECISION_SCENE_EXAMPLE,
        participants=['Rakshit', 'Samantha'],
        num_rounds=2,
        premise={
            'Rakshit': [
                (
                    'Rakshit and Samantha are friends.'
                ),
            ],
            'Samantha': [
                (
                    'Samantha and Rakshit are friends.'
                ),
            ],
        },
    ),
]


class EnvironmentPrefabsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='dialogic', prefab_name='dialogic', scenes=None),
      dict(testcase_name='dialogic_and_dramaturgic',
           prefab_name='dialogic_and_dramaturgic',
           scenes=_DIALOGIC_AND_DRAMATURGIC_SCENES),
      dict(testcase_name='formative_memories_initializer',
           prefab_name='formative_memories_initializer', scenes=None),
      dict(testcase_name='game_theoretic_and_dramaturgic',
           prefab_name='game_theoretic_and_dramaturgic',
           scenes=_GAME_THEORETIC_AND_DRAMATURGIC_SCENES),
      dict(testcase_name='generic', prefab_name='generic', scenes=None),
      dict(testcase_name='situated', prefab_name='situated', scenes=None),
  )
  def test_simulation_factory(
      self, prefab_name: str, scenes: Sequence[scene_lib.SceneSpec] | None):
    environment_module = ENVIRONMENT_PREFABS[prefab_name]
    environment_config = environment_module.GameMaster()

    model = no_language_model.NoLanguageModel()

    if scenes is not None:
      params = dict(copy.copy(environment_config.params))
      params['scenes'] = scenes
      environment_config.params = params

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
    )
    player_a = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='Rakshit',
        act_component=act_component,
        context_components={},
    )
    act_component_b = agent_components.concat_act_component.ConcatActComponent(
        model=model,
    )
    player_b = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='Samantha',
        act_component=act_component_b,
        context_components={},
    )

    players = [player_a, player_b]

    environment_config.entities = players

    environment = sequential.Sequential()

    memory_bank = associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )

    game_master = environment_config.build(
        model=model,
        memory_bank=memory_bank,
    )

    self.assertIsInstance(game_master,
                          entity_agent_with_logging.EntityAgentWithLogging)

    environment.run_loop(
        game_masters=[game_master],
        entities=players,
        max_steps=1,
    )

    self.assertIsInstance(game_master,
                          entity_agent_with_logging.EntityAgentWithLogging)

    if scenes is None:
      environment.run_loop(
          game_masters=[game_master],
          entities=players,
      )

      self.assertIsInstance(game_master,
                            entity_agent_with_logging.EntityAgentWithLogging)

if __name__ == '__main__':
  absltest.main()

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
import types

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as agent_components
from concordia.components.game_master import next_game_master
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.game_master import dialogic
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import formative_memories_initializer
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.game_master import generic
from concordia.prefabs.game_master import marketplace
from concordia.prefabs.game_master import open_ended_interviewer
from concordia.prefabs.game_master import scripted
from concordia.prefabs.game_master import situated
from concordia.prefabs.game_master import situated_in_time_and_place
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import scene as scene_lib
import numpy as np


ENVIRONMENT_PREFABS = {
    'dialogic': dialogic,
    'dialogic_and_dramaturgic': dialogic_and_dramaturgic,
    'formative_memories_initializer': formative_memories_initializer,
    'game_theoretic_and_dramaturgic': game_theoretic_and_dramaturgic,
    'generic': generic,
    'scripted': scripted,
    'situated': situated,
    'situated_in_time_and_place': situated_in_time_and_place,
}


def _embedder(text: str):
  del text
  return np.random.rand(3)

_PLAYERS = ['Rakshit', 'Samantha']

_DIALOGIC_SCENE_EXAMPLE = scene_lib.SceneTypeSpec(
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
        scene_type=_DIALOGIC_SCENE_EXAMPLE,
        participants=_PLAYERS,
        num_rounds=1,
        premise={
            _PLAYERS[0]: [
                (
                    f'{_PLAYERS[0]} and {_PLAYERS[1]} are friends.'
                ),
            ],
            _PLAYERS[1]: [
                (
                    f'{_PLAYERS[1]} and {_PLAYERS[0]} are friends.'
                ),
            ],
        },

    ),
]

_GAME_THEORETIC_AND_DRAMATURGIC_SCENES = [
    scene_lib.SceneSpec(
        scene_type=_DECISION_SCENE_EXAMPLE,
        participants=_PLAYERS,
        num_rounds=2,
        premise={
            _PLAYERS[0]: [
                (
                    f'{_PLAYERS[0]} and {_PLAYERS[1]} are friends.'
                ),
            ],
            _PLAYERS[1]: [
                (
                    f'{_PLAYERS[1]} and {_PLAYERS[0]} are friends.'
                ),
            ],
        },
    ),
]

_CONSTANT_SCRIPT = script = [
    {'name': _PLAYERS[0],
     'line': 'foo!'},
    {'name': _PLAYERS[1],
     'line': 'bar!'},
    {'name': _PLAYERS[0],
     'line': 'baz!'},
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
      dict(testcase_name='scripted', prefab_name='scripted', scenes=None),
      dict(testcase_name='situated', prefab_name='situated', scenes=None),
      dict(testcase_name='situated_in_time_and_place',
           prefab_name='situated_in_time_and_place', scenes=None),
  )
  def test_simulation_factory(
      self, prefab_name: str, scenes: Sequence[scene_lib.SceneSpec] | None):
    environment_module = ENVIRONMENT_PREFABS[prefab_name]
    environment_config = environment_module.GameMaster()

    model = no_language_model.NoLanguageModel()

    if scenes is not None:
      params = dict(copy.copy(environment_config.params))
      params['scenes'] = scenes  # pyrefly: ignore[unsupported-operation]
      environment_config.params = params  # pyrefly: ignore[bad-assignment]

    if prefab_name == 'scripted':
      params = dict(copy.copy(environment_config.params))
      params['script'] = _CONSTANT_SCRIPT  # pyrefly: ignore[unsupported-operation]
      environment_config.params = params  # pyrefly: ignore[bad-assignment]

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
    )
    player_a = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=_PLAYERS[0],
        act_component=act_component,
        context_components={},
    )
    act_component_b = agent_components.concat_act_component.ConcatActComponent(
        model=model,
    )
    player_b = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=_PLAYERS[1],
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
          max_steps=1,
      )

      self.assertIsInstance(game_master,
                            entity_agent_with_logging.EntityAgentWithLogging)


class _ExperimentComponent(entity_component.ContextComponent):
  """A do-nothing experiment component for the marketplace game master."""

  def __init__(self, acting_player_names: Sequence[str]):
    super().__init__()
    del acting_player_names

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


_NEXT_GAME_MASTER_SPEC = entity_lib.ActionSpec(
    call_to_action='Which rule set should we use for the next step?',
    output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
    options=('initial setup', 'main rules'),
)

_NEXT_GAME_MASTER_PREFABS = {
    'generic': (generic.GameMaster, {}),
    'situated': (situated.GameMaster, {}),
    'situated_in_time_and_place': (situated_in_time_and_place.GameMaster, {}),
    'marketplace': (
        marketplace.GameMaster,
        {'experiment_component_class': _ExperimentComponent},
    ),
    'open_ended_interviewer': (
        open_ended_interviewer.GameMaster,
        {
            'player_names': _PLAYERS,
            'questionnaires': [types.SimpleNamespace(name='q', questions=[])],
            'embedder': _embedder,
        },
    ),
}


class NextGameMasterTest(parameterized.TestCase):
  """Game masters name themselves as the next game master unless told not to."""

  def _build(self, prefab_name: str, **params):
    prefab_class, required_params = _NEXT_GAME_MASTER_PREFABS[prefab_name]
    prefab = prefab_class()
    prefab.params = {  # pyrefly: ignore[bad-assignment]
        **prefab.params,
        **required_params,
        'name': 'main rules',
        **params,
    }
    model = no_language_model.NoLanguageModel()
    prefab.entities = [
        entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=(
                agent_components.concat_act_component.ConcatActComponent(
                    model=model
                )
            ),
            context_components={},
        )
        for name in _PLAYERS
    ]
    return prefab.build(
        model=model,
        memory_bank=associative_memory.AssociativeMemoryBank(
            sentence_embedder=_embedder,
        ),
    )

  @parameterized.named_parameters(
      (name, name) for name in _NEXT_GAME_MASTER_PREFABS
  )
  def test_names_itself_as_the_next_game_master(self, prefab_name: str):
    game_master = self._build(prefab_name)

    self.assertEqual(game_master.act(_NEXT_GAME_MASTER_SPEC), 'main rules')

  @parameterized.named_parameters(
      (name, name) for name in _NEXT_GAME_MASTER_PREFABS
  )
  def test_lets_the_llm_choose_when_asked_to(self, prefab_name: str):
    game_master = self._build(prefab_name, llm_chooses_next_game_master=True)

    with self.assertRaises(KeyError):
      game_master.get_component(
          next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
      )


if __name__ == '__main__':
  absltest.main()

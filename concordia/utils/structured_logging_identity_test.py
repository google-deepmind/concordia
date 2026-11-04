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

"""Stock simulation and raw-frame identity roundtrips in structured logs."""

import contextlib
import io
import json

from absl.testing import absltest
from absl.testing import parameterized
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal
from concordia.prefabs.game_master import async_social_media
from concordia.prefabs.simulation import generic
from concordia.typing import prefab
from concordia.utils import structured_logging
import numpy as np


class _ScriptedForumModel(no_language_model.NoLanguageModel):

  def __init__(self, author, title):
    super().__init__()
    self._author = author
    self._title = title

  def sample_text(self, prompt: str, **kwargs) -> str:
    del prompt, kwargs
    return json.dumps({
        'action': 'post',
        'author': self._author,
        'title': self._title,
        'content': 'Synthetic fixture',
    })


class StructuredLogIdentityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'ordinary_names_with_entity_in_content',
          'Alice',
          'forum_rules',
          'Entity records',
      ),
      ('bracketed_player', 'Analyst [A]', 'forum_rules', 'Fixture'),
      ('game_master_with_entity_in_name', 'Alice', 'Entity rules', 'Fixture'),
      ('nested_wrapper_name', 'Entity [A]', 'forum_rules', 'Fixture'),
      ('player_named_entity', 'Entity', 'forum_rules', 'Fixture'),
      ('ordinary_space_names', 'Alex Smith', 'Forum rules', 'Fixture'),
  )
  def test_simulation_play_preserves_actor_and_game_master_in_json(
      self, player, game_master, title
  ):
    config = prefab.Config(
        prefabs={
            'player': minimal.Entity(),
            'forum': async_social_media.GameMaster(),
        },
        instances=[
            prefab.InstanceConfig(
                prefab='player',
                role=prefab.Role.ENTITY,
                params={'name': player},
            ),
            prefab.InstanceConfig(
                prefab='forum',
                role=prefab.Role.GAME_MASTER,
                params={'name': game_master},
            ),
        ],
        default_premise='',
        default_max_steps=1,
    )
    simulation = generic.Simulation(
        config,
        model=_ScriptedForumModel(player, title),
        embedder=lambda _: np.ones(3),
    )
    with contextlib.redirect_stdout(io.StringIO()):
      result = simulation.play(max_steps=1)
    restored = structured_logging.SimulationLog.from_json(result.to_json())
    actors = restored.get_entries_by_entity(player)
    self.assertLen(actors, 1)
    self.assertEqual(actors[0].entry_type, 'entity')
    self.assertEqual(actors[0].component_name, 'entity_action')
    masters = restored.get_entries_by_entity(game_master)
    self.assertLen(masters, 1)
    self.assertEqual(masters[0].entry_type, 'step')
    self.assertEqual(masters[0].component_name, 'game_master')
    self.assertEqual(set(restored.get_entity_names()), {player, game_master})
    self.assertEqual(restored.get_steps(), [1])

  @parameterized.parameters(
      'A]B', ' Analyst [A] ', 'Team Entity [A]', 'Entity [nested]'
  )
  def test_outer_wrapper_is_removed_once_without_normalizing_name(self, name):
    log = structured_logging.SimulationLog.from_raw_log([{
        'Step': 3,
        f'Entity [{name}]': {'__act__': {'Value': 'waits'}},
        'thread': name,
    }])
    restored = structured_logging.SimulationLog.from_json(log.to_json())
    self.assertEqual(restored.get_entity_names(), [name])
    entries = restored.get_entries_by_entity(name)
    self.assertLen(entries, 1)
    self.assertEqual(entries[0].entry_type, 'entity')
    self.assertEqual(
        restored.reconstruct_value(entries[0].deduplicated_data)['thread'], name
    )

  @parameterized.named_parameters(
      ('legacy_with_thread', 'Analyst [A]', 'Analyst [A]'),
      ('legacy_without_thread', '', 'Unknown'),
  )
  def test_legacy_bare_entity_keeps_thread_fallback(
      self, thread, expected_name
  ):
    log = structured_logging.SimulationLog.from_raw_log([{
        'Step': 1,
        'Entity': {'__act__': {'Value': 'waits'}},
        'thread': thread,
    }])
    self.assertEqual(log.get_entity_names(), [expected_name])
    self.assertEqual(log.entries[0].entry_type, 'entity')


if __name__ == '__main__':
  absltest.main()

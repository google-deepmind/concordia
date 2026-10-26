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

"""Tests for adding and removing entities in the generic simulation."""

from absl.testing import absltest
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal as minimal_entity
from concordia.prefabs.game_master import generic as generic_gm
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import prefab as prefab_lib
import numpy as np


def _embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


def _entity_config(name: str) -> prefab_lib.InstanceConfig:
  return prefab_lib.InstanceConfig(
      prefab='minimal_entity',
      role=prefab_lib.Role.ENTITY,
      params={**minimal_entity.Entity().params, 'name': name},
  )


def _simulation() -> simulation_lib.Simulation:
  config = prefab_lib.Config(
      prefabs={
          'minimal_entity': minimal_entity.Entity(),
          'generic_gm': generic_gm.GameMaster(),
      },
      instances=[
          _entity_config('Alice'),
          _entity_config('Bob'),
          prefab_lib.InstanceConfig(
              prefab='generic_gm',
              role=prefab_lib.Role.GAME_MASTER,
              params={**generic_gm.GameMaster().params, 'name': 'rules'},
          ),
      ],
  )
  return simulation_lib.Simulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
      embedder=_embedder,
  )


class RemoveEntityTest(absltest.TestCase):

  def test_removed_entity_leaves_the_simulation_and_its_checkpoints(self):
    sim = _simulation()
    entities = sim.entities
    bob = next(e for e in sim.entities if e.name == 'Bob')

    removed = sim.remove_entity('Bob')

    self.assertIs(removed, bob)
    self.assertIs(sim.entities, entities)  # Changed in place.
    self.assertEqual([e.name for e in sim.entities], ['Alice'])
    self.assertIsNone(sim.get_entity_prefab_config('Bob'))
    self.assertEqual(list(sim.make_checkpoint_data()['entities']), ['Alice'])

  def test_removing_an_unknown_entity_raises(self):
    sim = _simulation()
    with self.assertRaises(KeyError):
      sim.remove_entity('Carol')

  def test_a_removed_name_can_join_again(self):
    sim = _simulation()
    sim.remove_entity('Bob')
    sim.add_entity(_entity_config('Bob'))
    self.assertEqual(sorted(e.name for e in sim.entities), ['Alice', 'Bob'])


if __name__ == '__main__':
  absltest.main()

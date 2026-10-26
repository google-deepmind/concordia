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

"""Tests of births and deaths for the generic simulation."""

from absl.testing import absltest
from concordia.contrib.prefabs.simulation import population as population_lib
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal as minimal_entity
from concordia.prefabs.game_master import generic as generic_gm
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import prefab as prefab_lib
import numpy as np


def _embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


def _params(name: str) -> dict[str, object]:
  return {**minimal_entity.Entity().params, 'name': name}


def _simulation() -> simulation_lib.Simulation:
  config = prefab_lib.Config(
      prefabs={
          'minimal_entity': minimal_entity.Entity(),
          'generic_gm': generic_gm.GameMaster(),
      },
      instances=[
          prefab_lib.InstanceConfig(
              prefab='minimal_entity',
              role=prefab_lib.Role.ENTITY,
              params=_params('Alice'),
          ),
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


class PopulationTest(absltest.TestCase):

  def test_a_spawned_entity_joins_and_a_retired_one_leaves_on_record(self):
    sim = _simulation()
    population = population_lib.Population(sim, prefab='minimal_entity')

    population.spawn(_params('Bob'))
    self.assertEqual(sorted(e.name for e in sim.entities), ['Alice', 'Bob'])

    population.retire('Alice')

    self.assertEqual([e.name for e in sim.entities], ['Bob'])
    (record,) = population.archive.records
    self.assertEqual(record['name'], 'Alice')
    self.assertEqual(record['params']['name'], 'Alice')
    self.assertIn('context_components', record['state'])
    self.assertEqual(list(sim.make_checkpoint_data()['entities']), ['Bob'])

  def test_retiring_an_unknown_name_changes_nothing(self):
    sim = _simulation()
    population = population_lib.Population(sim, prefab='minimal_entity')

    population.retire('Carol')

    self.assertEqual([e.name for e in sim.entities], ['Alice'])
    self.assertEmpty(population.archive.records)

  def test_the_archive_keeps_the_latest_and_sinks_them_all(self):
    sunk = []
    archive = population_lib.Archive(keep=2, sink=sunk.append)

    for i in range(5):
      archive.add({'name': f'p{i}'})

    self.assertEqual([r['name'] for r in archive.records], ['p3', 'p4'])
    self.assertEqual([r['name'] for r in sunk], [f'p{i}' for i in range(5)])

  def test_a_keep_of_none_keeps_nothing(self):
    archive = population_lib.Archive(keep=0)
    archive.add({'name': 'p0'})
    self.assertEmpty(archive.records)


if __name__ == '__main__':
  absltest.main()

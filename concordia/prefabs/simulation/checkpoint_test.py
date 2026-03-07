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

"""Tests for simulation checkpoint save and restore."""

import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal as minimal_entity
from concordia.prefabs.game_master import generic as generic_gm
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib
from concordia.utils import measurements as measurements_lib
import numpy as np


def _embedder(text: str):
  del text
  return np.random.rand(3)


def _make_config(measurements=None):
  entity_a_params = dict(minimal_entity.Entity().params)
  entity_a_params['name'] = 'Alice'
  if measurements is not None:
    entity_a_params['measurements'] = measurements

  entity_b_params = dict(minimal_entity.Entity().params)
  entity_b_params['name'] = 'Bob'
  if measurements is not None:
    entity_b_params['measurements'] = measurements

  gm_params = dict(generic_gm.GameMaster().params)
  gm_params['name'] = 'default_rules'
  if measurements is not None:
    gm_params['measurements'] = measurements

  config = prefab_lib.Config(
      prefabs={
          'minimal_entity': minimal_entity.Entity(),
          'generic_gm': generic_gm.GameMaster(),
      },
      instances=[
          prefab_lib.InstanceConfig(
              prefab='minimal_entity',
              role=prefab_lib.Role.ENTITY,
              params=entity_a_params,
          ),
          prefab_lib.InstanceConfig(
              prefab='minimal_entity',
              role=prefab_lib.Role.ENTITY,
              params=entity_b_params,
          ),
          prefab_lib.InstanceConfig(
              prefab='generic_gm',
              role=prefab_lib.Role.GAME_MASTER,
              params=gm_params,
          ),
      ],
      default_premise='Test premise.',
      default_max_steps=1,
  )
  return config


def _checkpoint_roundtrip(sim):
  checkpoint_data = sim.make_checkpoint_data()
  with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'checkpoint.json')
    with open(path, 'w') as f:
      json.dump(checkpoint_data, f)
    with open(path) as f:
      loaded = json.load(f)
  sim.load_from_checkpoint(loaded)
  return loaded


class CheckpointTest(parameterized.TestCase):

  def test_checkpoint_structure(self):
    config = _make_config()
    model = no_language_model.NoLanguageModel()
    sim = simulation_lib.Simulation(
        config=config,
        model=model,
        embedder=_embedder,
    )
    checkpoint_data = sim.make_checkpoint_data()
    self.assertIn('entities', checkpoint_data)
    self.assertIn('game_masters', checkpoint_data)
    self.assertIn('raw_log', checkpoint_data)
    self.assertIn('Alice', checkpoint_data['entities'])
    self.assertIn('Bob', checkpoint_data['entities'])
    self.assertIn('default_rules', checkpoint_data['game_masters'])

  def test_roundtrip_without_measurements(self):
    config = _make_config()
    model = no_language_model.NoLanguageModel()
    sim = simulation_lib.Simulation(
        config=config,
        model=model,
        embedder=_embedder,
    )

    original_entity_names = {e.name for e in sim.entities}
    original_gm_names = {gm.name for gm in sim.game_masters}

    _checkpoint_roundtrip(sim)

    restored_entity_names = {e.name for e in sim.entities}
    restored_gm_names = {gm.name for gm in sim.game_masters}
    self.assertEqual(original_entity_names, restored_entity_names)
    self.assertEqual(original_gm_names, restored_gm_names)

  def test_measurements_dropped_by_json_serializer(self):
    reactive = async_measurements_lib.ReactiveMeasurements()
    config = _make_config(measurements=reactive)
    model = no_language_model.NoLanguageModel()
    sim = simulation_lib.Simulation(
        config=config,
        model=model,
        embedder=_embedder,
    )
    checkpoint_data = sim.make_checkpoint_data()

    for entity_name, entity_data in checkpoint_data['entities'].items():
      self.assertNotIn(
          'measurements',
          entity_data['entity_params'],
          f'{entity_name} params should not contain measurements after'
          ' JSON serialization (it is not JSON-serializable)',
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='standard_measurements',
          measurements_factory=measurements_lib.Measurements,
      ),
      dict(
          testcase_name='reactive_measurements',
          measurements_factory=async_measurements_lib.ReactiveMeasurements,
      ),
  )
  def test_checkpoint_roundtrip_restores_measurements(
      self,
      measurements_factory,
  ):
    measurements = measurements_factory()
    config = _make_config(measurements=measurements)
    model = no_language_model.NoLanguageModel()
    sim = simulation_lib.Simulation(
        config=config,
        model=model,
        embedder=_embedder,
    )

    for entity in sim.entities:
      self.assertIsInstance(entity.measurements, type(measurements))

    _checkpoint_roundtrip(sim)

    for entity in sim.entities:
      self.assertIsInstance(
          entity.measurements,
          type(measurements),
          f'{entity.name} must have {type(measurements).__name__} after'
          ' restore',
      )
    for gm in sim.game_masters:
      self.assertIsInstance(
          gm.measurements,
          type(measurements),
          f'{gm.name} must have {type(measurements).__name__} after restore',
      )

  def test_reactive_capture_works_after_restore(self):
    reactive = async_measurements_lib.ReactiveMeasurements()
    config = _make_config(measurements=reactive)
    model = no_language_model.NoLanguageModel()
    sim = simulation_lib.Simulation(
        config=config,
        model=model,
        embedder=_embedder,
    )

    _checkpoint_roundtrip(sim)

    entity = sim.entities[0]
    measurements = entity.measurements
    if isinstance(measurements, async_measurements_lib.ReactiveMeasurements):
      with measurements.capture(entity.name) as captured:
        measurements.publish_datum(
            'test_channel',
            {'Key': 'test', 'Value': 'hello'},
            capture_key=entity.name,
        )
      self.assertIn('test_channel', captured)
      self.assertEqual(captured['test_channel']['Value'], 'hello')


if __name__ == '__main__':
  absltest.main()

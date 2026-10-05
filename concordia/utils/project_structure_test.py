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

"""Registered structural authoring contracts, without simulation execution."""

import copy
import json
from unittest import mock

from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import project_config
from concordia.utils import project_test_support as fixtures
from concordia.utils import simulation_server
import pytest


@pytest.fixture(autouse=True)
def prohibit_execution():
  with (
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError,
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError,
      ),
  ):
    yield


def expanded(registry):
  document = registry.default_document('builder-v1')
  player = copy.deepcopy(document['instances'][0])
  player['id'] = 'third-player'
  player['params']['name'] = 'Charlie 🎵'
  player['params']['custom_instructions'] = 'Literal "quotes"\n</script> & café'
  player['params']['goal'] = 'Listen carefully'
  gm = copy.deepcopy(document['instances'][2])
  gm['id'] = 'second-gm'
  gm['params']['name'] = 'Second room'
  gm['params']['next_game_master_name'] = 'conversation'
  document['instances'].extend([player, gm])
  document['instances'][2]['params']['name'] = 'Renamed conversation'
  document['instances'].reverse()
  return document


def test_structure_roundtrip_order_reference_and_real_components():
  registry = fixtures.builder_registry()
  document = expanded(registry)
  assert registry.loads(registry.dumps(document)) == document
  config = registry.to_config(document)
  assert [x.params['name'] for x in config.instances] == [
      x['params']['name'] for x in document['instances']
  ]
  assert (
      config.instances[0].params['next_game_master_name']
      == 'Renamed conversation'
  )
  simulation = fixtures.build(config)
  players = {
      player.name: fixtures.as_agent(player)
      for player in simulation.get_entities()
  }
  assert len(players) == 3
  assert (
      players['Charlie 🎵'].get_component('Goal').get_state()['state']
      == 'Listen carefully'
  )
  assert (
      players['Charlie 🎵'].get_component('Instructions').get_state()['state']
      == document['instances'][1]['params']['custom_instructions']
  )
  assert 'SelfPerception' in players['Bob'].get_all_context_components()
  assert (
      'SelfPerception' not in players['Charlie 🎵'].get_all_context_components()
  )
  choices = registry.inspector(document)['second-gm']['next_game_master_name'][
      'choices'
  ]
  assert {'value': 'conversation', 'label': 'Renamed conversation'} in choices
  catalog = registry.catalog(document)
  catalog[0]['instance']['params']['name'] = 'mutated'
  assert registry.catalog(document)[0]['instance']['params']['name'] == 'Alice'


@pytest.mark.parametrize(
    'mutation',
    [
        lambda d: d['instances'][0].update(prototype='os.system'),
        lambda d: d['instances'][0].update(prototype=[]),
        lambda d: d['instances'][0].update(prefab='os.system'),
        lambda d: d['instances'][0].update(role='game_master'),
        lambda d: d['instances'][0].update(id='<script>'),
        lambda d: d['instances'][0].update(id='x' * 129),
        lambda d: d['instances'][0].update(id='bob'),
        lambda d: d['instances'][0].update(constructor='eval'),
        lambda d: d['instances'][0]['params'].update(name='Bob'),
        lambda d: d['instances'][0]['params'].update(randomize_choices=1),
        lambda d: d['instances'][2]['params'].update(
            next_game_master_name='alice'
        ),
        lambda d: d['instances'][2]['params'].update(
            next_game_master_name='missing'
        ),
        lambda d: d['instances'][2]['params'].update(
            acting_order='unsupported'
        ),
        lambda d: d['instances'].pop(),
        lambda d: d.update(instances=[d['instances'][2]]),
        lambda d: d.update(instances=d['instances'] * 34),
        lambda d: d.update(schema_version=1),
    ],
)
def test_invalid_structure_is_atomic(mutation):
  registry = fixtures.builder_registry()
  initial = registry.default_document('builder-v1')
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      initial,
      mock.Mock(),
      integrated=True,
      preview=lambda config: fixtures.build(config).make_checkpoint_data(),
  )
  before = server.get_project()
  invalid = copy.deepcopy(initial)
  mutation(invalid)
  with pytest.raises((project_config.ValidationError, ValueError)):
    server.replace_project(json.dumps(invalid), before['revision'])
  assert server.get_project() == before
  assert server.simulation is None


def test_referenced_removal_and_stale_revision_preserve_saved_structure():
  registry = fixtures.builder_registry()
  document = expanded(registry)
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      document,
      mock.Mock(),
      integrated=True,
      preview=lambda config: fixtures.build(config).make_checkpoint_data(),
  )
  revision = server.get_project()['revision']
  removed = copy.deepcopy(document)
  removed['instances'] = [
      x for x in removed['instances'] if x['id'] != 'conversation'
  ]
  with pytest.raises(
      project_config.ValidationError, match='target instance ID'
  ):
    registry.normalize(removed)
  removed['instances'][0]['params']['next_game_master_name'] = 'second-gm'
  server.replace_project(registry.dumps(removed), revision)
  saved = server.get_project()
  with pytest.raises(ValueError, match='changed in another tab'):
    server.replace_project(registry.dumps(document), revision)
  assert server.get_project() == saved
  assert registry.loads(registry.dumps(saved['document'])) == removed


if __name__ == '__main__':
  raise SystemExit(pytest.main([__file__]))

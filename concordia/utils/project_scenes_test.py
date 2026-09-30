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

"""Scene and registered component authoring; all simulation execution blocked."""

import copy
import json
from unittest import mock

from concordia.components.agent import constant
from concordia.components.agent import observation
from concordia.components.game_master import scene_tracker
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import scene as scene_lib
from concordia.utils import project_config
from concordia.utils import project_test_support as fixtures
from concordia.utils import simulation_server
import pytest


@pytest.fixture(autouse=True)
def no_execution():
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('no simulation'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError('no model'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError('no model'),
      ),
  ):
    yield


def world():
  registry = fixtures.scene_registry()
  return registry, registry.default_document('scenes-v1')


def test_scene_specs_references_text_components_and_fresh_builds():
  registry, document = world()
  document['instances'][0]['params']['name'] = 'Renamed actor 🎵'
  document['instances'][2]['params']['name'] = 'Renamed GM'
  document['groups'][0]['name'] = 'Our group'
  document['scene_types'][0]['name'] = 'A renamed type'
  literal = '</script> "quoted" & café\n\nlast line\n'
  document['scene_types'][0]['premise'] = literal
  second = copy.deepcopy(document['scenes'][0])
  second.update(id='encore', name='Encore', num_rounds=3, premise='')
  document['scenes'].append(second)
  document['components'] = [
      dict(
          id='context',
          instance='alice',
          type='constant',
          name='Identity',
          params={'state': literal, 'pre_act_label': 'Identity'},
      ),
      dict(
          id='recent',
          instance='alice',
          type='recent-observations',
          name='Recent events',
          params={'history_length': 7, 'pre_act_label': 'Events'},
      ),
  ]
  saved = registry.loads(registry.dumps(document))
  assert saved == document
  first_config, second_config = registry.to_config(saved), registry.to_config(
      saved
  )
  scenes = first_config.instances[2].params['scenes']
  assert isinstance(scenes[0], scene_lib.SceneSpec)
  assert isinstance(scenes[0].scene_type, scene_lib.SceneTypeSpec)
  assert scenes[0].scene_type.game_master_name == 'Renamed GM'
  assert scenes[0].participants == ['Renamed actor 🎵', 'Bob']
  assert scenes[0].scene_type.default_premise['Renamed actor 🎵'] == [literal]
  assert scenes[0].premise is None
  assert scenes[1].premise['Renamed actor 🎵'] == ['']
  first, second = fixtures.build(first_config), fixtures.build(second_config)
  gm = first.get_game_masters()[0]
  tracker = gm.get_component('__next_game_master__')
  assert isinstance(tracker, scene_tracker.SceneTracker)
  assert set(tracker.get_participants()) == {'Renamed actor 🎵', 'Bob'}
  actor, other = first.get_entities()[0], second.get_entities()[0]
  context = actor.get_component('authored_context')
  recent = actor.get_component('authored_recent')
  assert isinstance(context, constant.Constant)
  assert isinstance(recent, observation.LastNObservations)
  assert recent.get_state()['history_length'] == 7
  assert (
      recent.get_state()['memory_component_key']
      in actor.get_all_context_components()
  )
  assert context.get_state()['state'] == literal
  assert context is not other.get_component('authored_context')
  context.set_state({'state': 'runtime only'})
  assert registry.loads(registry.dumps(document)) == saved
  assert other.get_component('authored_context').get_state()['state'] == literal
  assert list(first_config.instances[0].params['extra_components']) == [
      'authored_context',
      'authored_recent',
  ]


@pytest.mark.parametrize(
    'mutate',
    [
        lambda d: d['groups'][0]['participants'].append('missing'),
        lambda d: d['groups'][0]['participants'].append('conversation'),
        lambda d: d['groups'][0]['participants'].append('alice'),
        lambda d: d['groups'][0].update(participants=[]),
        lambda d: d['groups'][0].update(participants=['bob']),
        lambda d: d['scene_types'][0].update(game_master='alice'),
        lambda d: d['scene_types'][0].update(group='missing'),
        lambda d: d['scene_types'][0].update(
            game_master={'constructor': 'os.system'}
        ),
        lambda d: d['scene_types'][0].update(
            action_spec={'callable': 'anything'}
        ),
        lambda d: d['scenes'][0].update(scene_type='missing'),
        lambda d: d['scenes'][0].update(participants=[]),
        lambda d: d['scenes'][0].update(num_rounds=True),
        lambda d: d['scenes'][0].update(num_rounds=1001),
        lambda d: d['scenes'][0].update(start_time='2026-01-01'),
        lambda d: d['scenes'][0].update(premise={'constructor': 'eval'}),
        lambda d: d.update(scenes=[]),
        lambda d: d['scenes'].append(copy.deepcopy(d['scenes'][0])),
        lambda d: d['instances'].pop(0),
    ],
)
def test_invalid_scene_data_keeps_document_and_preview(mutate):
  registry, document = world()
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      document,
      mock.Mock(),
      integrated=True,
      preview=lambda c: fixtures.build(c).make_checkpoint_data(),
  )
  before = server.get_project()
  snapshot = server.operation_service.snapshot('developer')
  invalid = copy.deepcopy(document)
  mutate(invalid)
  with pytest.raises(project_config.ValidationError):
    server.replace_project(json.dumps(invalid), before['revision'])
  assert server.get_project() == before
  assert server.operation_service.snapshot('developer') == snapshot
  assert server.simulation is None


def test_callable_input_rejected_without_invocation():
  registry, document = world()
  callback = mock.Mock(side_effect=AssertionError('must not call'))
  document['scenes'][0]['premise'] = callback
  with pytest.raises(project_config.ValidationError):
    registry.normalize(document)
  callback.assert_not_called()


@pytest.mark.parametrize(
    'change',
    [
        {'type': 'os.system'},
        {'instance': 'missing'},
        {'instance': 'conversation'},
        {'params': {'history_length': True, 'pre_act_label': 'x'}},
        {'params': {'history_length': 0, 'pre_act_label': 'x'}},
        {'params': {'history_length': 1001, 'pre_act_label': 'x'}},
        {
            'params': {
                'history_length': 5,
                'pre_act_label': 'x',
                'constructor': 'eval',
            }
        },
        {'id': '../Instructions'},
        {'id': {}},
        {'type': []},
    ],
)
def test_rejected_component_data(change):
  registry, document = world()
  item = dict(
      id='recent',
      instance='alice',
      type='recent-observations',
      name='Recent events',
      params={'history_length': 5, 'pre_act_label': 'Events'},
  )
  item.update(change)
  document['components'] = [item]
  with pytest.raises(project_config.ValidationError):
    registry.normalize(document)


def test_fresh_editor_reopen_and_stale_revision():
  registry, document = world()
  first = simulation_server.SimulationServer(port=0)
  second = simulation_server.SimulationServer(port=0)
  for server in (first, second):
    server.configure_project(
        registry,
        document,
        mock.Mock(),
        integrated=True,
        preview=lambda c: fixtures.build(c).make_checkpoint_data(),
    )
  changed = copy.deepcopy(document)
  changed['scenes'][0]['premise'] = 'Literal override\n🎵'
  changed['scenes'][0]['num_rounds'] = 3
  first.replace_project(registry.dumps(changed), 0)
  exported = registry.dumps(first.get_project()['document'])
  second.replace_project(exported, 0)
  assert second.get_project()['document'] == changed
  saved = second.get_project()
  with pytest.raises(ValueError, match='another tab'):
    second.replace_project(registry.dumps(document), 0)
  assert second.get_project() == saved

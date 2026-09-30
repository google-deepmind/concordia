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

"""Registered component CRUD, prefab consumption and atomic saves; no execution."""

import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

from concordia.components.agent import constant
from concordia.components.agent import observation
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import project_components
from concordia.utils import project_config
from concordia.utils import project_test_support as fixtures
from concordia.utils import simulation_server
import pytest


@pytest.mark.parametrize(
    'module', ['project_components', 'project_config', 'project_scenes']
)
def test_public_modules_import_independently(module):
  # A shared pytest process can hide cycles after another module imported config.
  result = subprocess.run(
      [sys.executable, '-c', f'from concordia.utils import {module}'],
      cwd=Path(__file__).resolve().parents[2],
      capture_output=True,
      text=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr


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


def record(owner='alice', kind='constant', identifier='context'):
  params = {'state': 'literal </script> 🎵\n', 'pre_act_label': 'Identity'}
  if kind == 'recent-observations':
    params = {'history_length': 7, 'pre_act_label': 'Recent events'}
  return dict(
      id=identifier,
      instance=owner,
      type=kind,
      name='Display label',
      params=params,
  )


def server_for(registry, document):
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      document,
      mock.Mock(side_effect=AssertionError('no run')),
      integrated=True,
      preview=lambda c: fixtures.build(c).make_checkpoint_data(),
  )
  return server


@pytest.mark.parametrize(
    ('owner', 'kind'),
    [
        ('alice', 'constant'),
        ('alice', 'recent-observations'),
        ('bob', 'constant'),
        ('bob', 'recent-observations'),
        ('conversation', 'constant'),
    ],
)
def test_each_offered_prefab_consumes_parameters_and_order(owner, kind):
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  document['components'] = [
      record(owner, kind),
      record(owner, 'constant', 'second'),
  ]
  config = registry.to_config(document)
  built = fixtures.build(config)
  entities = {
      e.name: e for e in [*built.get_entities(), *built.get_game_masters()]
  }
  name = next(
      x['params']['name'] for x in document['instances'] if x['id'] == owner
  )
  entity = entities[name]
  component = entity.get_component('authored_context')
  assert isinstance(
      component,
      constant.Constant
      if kind == 'constant'
      else observation.LastNObservations,
  )
  state = component.get_state()
  for key, value in document['components'][0]['params'].items():
    assert state[key] == value
  if kind == 'recent-observations':
    assert state['memory_component_key'] in entity.get_all_context_components()
  order = entity.get_act_component().get_state()['component_order']
  assert order[-2:] == ['authored_context', 'authored_second']
  # Rename is a display label; stable keys, literal context and built-ins survive.
  document['components'][0]['name'] = 'Renamed 🎵'
  document['components'].reverse()
  again = fixtures.build(registry.to_config(document))
  other = next(
      e
      for e in [*again.get_entities(), *again.get_game_masters()]
      if e.name == name
  )
  assert other.get_act_component().get_state()['component_order'][-2:] == [
      'authored_second',
      'authored_context',
  ]
  assert other.get_component('authored_context') is not component
  assert other.get_component('authored_context').get_state() == state
  component.set_state(
      {'state': 'runtime change'}
      if kind == 'constant'
      else {'history_length': 3}
  )
  assert other.get_component('authored_context').get_state() == state
  assert registry.loads(registry.dumps(document)) == document


def test_crud_save_reopen_and_stale_requests_are_atomic():
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  server = server_for(registry, document)
  service = server.operation_service
  document['components'] = [
      record('bob'),
      record('bob', 'recent-observations', 'events'),
  ]
  server.replace_project(registry.dumps(document), 0)
  document['components'][0]['params']['state'] = 'configured\n\nexact ending\n'
  document['components'][0]['name'] = 'Renamed'
  document['components'].reverse()
  server.replace_project(registry.dumps(document), 1)
  reopened = server_for(
      registry, registry.loads(registry.dumps(server.get_project()['document']))
  )
  assert reopened.get_project()['document'] == document
  before = service.snapshot('developer')
  stale = dict(
      operation='project.save',
      arguments={'text': registry.dumps(document), 'revision': 0},
      revision=service.revision,
      references=copy.deepcopy(service.references),
      retry_key='stale-components',
  )
  with pytest.raises(operation_service.OperationError, match='another tab'):
    service.dispatch('developer', stale)
  assert service.snapshot('developer') == before
  document['components'].pop(0)
  server.replace_project(registry.dumps(document), 2)
  assert server.get_project()['document']['components'][0]['id'] == 'context'
  document['components'].clear()
  server.replace_project(registry.dumps(document), 3)
  assert (
      'authored_context'
      not in server.operation_service.snapshot('developer')['result'][
          'definition'
      ]['entities']['entity_1']['component_info']['context_components']
  )
  assert server.simulation is None


@pytest.mark.parametrize(
    'change',
    [
        {'type': 'os.system'},
        {'type': []},
        {'instance': 'missing'},
        {'instance': 'conversation', 'type': 'recent-observations'},
        {'id': '../Instructions'},
        {'id': ''},
        {'id': {}},
        {'name': ''},
        {'name': False},
        {'params': {'state': 42, 'pre_act_label': 'x'}},
        {'params': {'state': {'constructor': 'eval'}, 'pre_act_label': 'x'}},
        {'params': {'state': 'x', 'pre_act_label': 'x', 'import': 'os'}},
        {'params': {'state': 'x'}},
        {'constructor': 'Constant'},
    ],
)
def test_invalid_component_saves_leave_definition_and_preview_intact(change):
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  server = server_for(registry, document)
  before = server.operation_service.snapshot('developer')
  item = record()
  item.update(change)
  document['components'] = [item]
  with pytest.raises(project_config.ValidationError):
    server.replace_project(json.dumps(document), 0)
  assert server.operation_service.snapshot('developer') == before
  assert server.get_project()['revision'] == 0


@pytest.mark.parametrize(
    'records',
    [
        lambda: [record(), record()],
        lambda: [record(identifier=str(i)) for i in range(101)],
    ],
)
def test_duplicate_ids_and_component_limit(records):
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  document['components'] = records()
  with pytest.raises(project_config.ValidationError):
    registry.normalize(document)


def test_callable_parameter_rejected_and_registration_dependencies_checked():
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  callback = mock.Mock(side_effect=AssertionError('must not invoke'))
  item = record()
  item['params']['state'] = callback
  document['components'] = [item]
  with pytest.raises(project_config.ValidationError):
    registry.to_config(document)
  callback.assert_not_called()
  types = project_components.standard_types(('alice', 'bob'), ('alice', 'bob'))
  types['constant'] = dataclasses.replace(
      types['constant'], dependencies=('missing_memory',)
  )
  with pytest.raises(
      project_config.ValidationError, match='missing prefab dependencies'
  ):
    project_components.validate_registration(
        fixtures.scene_config(), ('alice', 'bob', 'conversation'), types
    )
  with pytest.raises(
      project_config.ValidationError, match='unsupported component prefab host'
  ):
    project_components.validate_registration(
        fixtures.make_config(),
        ('alice', 'bob', 'conversation'),
        project_components.standard_types(('alice',), ('conversation',)),
    )


def test_component_only_template_and_legacy_compatibility():
  registry = project_config.Registry({
      'components-only': project_config.Template(
          factory=fixtures.make_config,
          instance_ids=('alice', 'bob', 'conversation'),
          editable_instances=True,
          component_types=project_components.standard_types(
              ('alice', 'bob'), ('alice', 'bob')
          ),
      )
  })
  document = registry.default_document('components-only')
  assert document['schema_version'] == 3
  assert 'scenes' not in document
  document['components'] = [record('bob')]
  assert registry.loads(registry.dumps(document)) == document
  assert (
      fixtures.build(registry.to_config(document))
      .get_entities()[1]
      .get_component('authored_context')
      .get_state()['state']
      == document['components'][0]['params']['state']
  )
  for old_registry, key in [
      (fixtures.registry(), fixtures.TEMPLATE_KEY),
      (fixtures.builder_registry(), 'builder-v1'),
  ]:
    old = old_registry.default_document(key)
    assert 'components' not in old
    assert old_registry.component_catalog(old) == []
    assert old_registry.loads(old_registry.dumps(old)) == old
    with pytest.raises(project_config.ValidationError):
      old_registry.normalize(dict(old, components=[]))


def test_basic_prefab_factory_order_and_rejected_collisions():
  config = fixtures.scene_registry().to_config(
      fixtures.scene_registry().default_document('scenes-v1')
  )
  simulation = fixtures.build(config)
  memory_bank = mock.Mock()
  policy = mock.Mock()
  factory = mock.Mock(return_value=policy)
  prefab = basic.Entity(
      params={
          'name': 'Player',
          'extra_components': {'custom': constant.Constant('context')},
      }
  )
  entity = prefab.build(
      no_language_model.NoLanguageModel(),
      memory_bank,
      act_component_factory=factory,
  )
  assert entity.get_act_component() is policy
  assert factory.call_args.args[0][-1] == 'custom'
  assert 'SelfPerception' in factory.call_args.args[0]
  assert (
      simulation.get_entities()[1]
      .get_act_component()
      .get_state()['component_order']
  )
  for extras, indices in [
      ({'__memory__': constant.Constant('bad')}, {}),
      ({'custom': constant.Constant('x')}, {'other': 0}),
      ({'custom': constant.Constant('x')}, {'custom': True}),
  ]:
    prefab.params = {
        'extra_components': extras,
        'extra_components_index': indices,
    }
    with pytest.raises(ValueError):
      prefab.build(no_language_model.NoLanguageModel(), memory_bank)


def test_authored_component_cannot_be_changed_through_runtime_edit():
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  document['components'] = [record()]
  server = server_for(registry, document)
  adapter = server._project_editor
  before = server.operation_service.snapshot('developer')
  with mock.patch.object(adapter, 'state', return_value='paused'):
    with pytest.raises(ValueError, match='Only registered Instructions/Goal'):
      adapter.edit({
          'instance_id': 'alice',
          'component': 'authored_context',
          'value': 'runtime text',
      })
  assert server.operation_service.snapshot('developer') == before
  assert server.get_project()['document'] == document

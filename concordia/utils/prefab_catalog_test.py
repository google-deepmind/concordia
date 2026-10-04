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

"""Installed prefab discovery and real guarded construction, without execution."""

import argparse
import copy
import dataclasses
import importlib
import json
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

from concordia.associative_memory import basic_associative_memory
from concordia.command_line_interface import concordia_session
from concordia.components.agent import constant
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import prefab_catalog
from concordia.utils import project_components
from concordia.utils import project_config
from concordia.utils import project_test_support as fixtures
from concordia.utils import session_commands
from concordia.utils import session_commands_test
from concordia.utils import session_draft
from concordia.utils import simulation_server
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def prohibit_execution():
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('no simulation'),
      ),
      mock.patch.object(
          simulation_server.SimulationServer,
          'start',
          side_effect=AssertionError('no listener'),
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


def memory():
  return basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=lambda _: np.zeros(3)
  )


def test_new_standard_and_contrib_modules_discovered_without_registration(
    tmp_path, monkeypatch
):
  for namespace in (
      'concordia.prefabs.entity',
      'concordia.contrib.prefabs.entity',
  ):
    folder = tmp_path / namespace
    folder.mkdir()
    (folder / 'catalog_probe.py').write_text(
        'from concordia.prefabs.entity import minimal\n'
        'class Entity(minimal.Entity):\n'
        '  description = "Installed test prefab"\n'
    )
    package = importlib.import_module(namespace)
    monkeypatch.setattr(package, '__path__', [*package.__path__, str(folder)])
  entries, unavailable = prefab_catalog.discover()
  assert not unavailable
  assert 'entity.catalog_probe.Entity' in entries
  assert 'contrib.entity.catalog_probe.Entity' in entries
  assert len(entries) == len(set(entries))
  assert (
      entries['entity.catalog_probe.Entity'].prefab
      is not prefab_catalog.discover()[0]['entity.catalog_probe.Entity'].prefab
  )
  module = importlib.import_module('concordia.prefabs.entity.catalog_probe')
  monkeypatch.setattr(
      module,
      'Second',
      type('Second', (module.Entity,), {'__module__': module.__name__}),
      raising=False,
  )
  expanded = prefab_catalog.discover()[0]
  assert 'entity.catalog_probe.Entity' in expanded
  assert 'entity.catalog_probe.Second' in expanded
  assert not any('_test' in key for key in entries)
  assert (
      'contrib.entity.conversations_with_ai_companions.HumanUserEntity'
      in entries
  )
  assert (
      'contrib.entity.conversations_with_ai_companions.AICompanionEntity'
      in entries
  )


def test_optional_dependency_reported_but_internal_import_bugs_propagate(
    monkeypatch,
):
  real = importlib.import_module

  def imported(name):
    if name == 'concordia.contrib.prefabs.entity.basic_with_image':
      raise ModuleNotFoundError(
          'optional missing', name='optional_image_package'
      )
    return real(name)

  monkeypatch.setattr(prefab_catalog.importlib, 'import_module', imported)
  entries, unavailable = prefab_catalog.discover()
  assert 'entity.minimal.Entity' in entries
  assert unavailable == [
      prefab_catalog.Unavailable(
          'concordia.contrib.prefabs.entity.basic_with_image',
          'optional_image_package',
      )
  ]

  def unavailable_package(name):
    if name == 'concordia.contrib.prefabs.game_master':
      raise ModuleNotFoundError(
          'optional missing', name='optional_forum_package'
      )
    return real(name)

  monkeypatch.setattr(
      prefab_catalog.importlib, 'import_module', unavailable_package
  )
  available, diagnostics = prefab_catalog.discover()
  assert 'entity.minimal.Entity' in available
  assert diagnostics == [
      prefab_catalog.Unavailable(
          'concordia.contrib.prefabs.game_master', 'optional_forum_package'
      )
  ]

  def broken(name):
    if name == 'concordia.contrib.prefabs.entity.basic_with_image':
      raise ModuleNotFoundError('internal missing', name='concordia.missing')
    return real(name)

  monkeypatch.setattr(prefab_catalog.importlib, 'import_module', broken)
  with pytest.raises(ModuleNotFoundError, match='internal missing'):
    prefab_catalog.discover()


@pytest.mark.parametrize(
    'key',
    [
        key
        for key, entry in prefab_catalog.discover()[0].items()
        if not entry.prefab.supports_extra_components
    ],
)
def test_every_unsupported_prefab_rejects_before_any_construction(key):
  prefab = prefab_catalog.discover()[0][key].prefab
  prefab.params = {
      **prefab.params,
      'extra_components': {'added': constant.Constant('hello')},
  }
  with pytest.raises(
      NotImplementedError, match='extra_components not implemented yet'
  ):
    prefab.build(mock.Mock(), mock.Mock())


@pytest.mark.parametrize(
    'key',
    [
        'entity.minimal.Entity',
        'entity.basic.Entity',
        'game_master.dialogic_and_dramaturgic.GameMaster',
        'game_master.generic.GameMaster',
        'game_master.situated.GameMaster',
        'contrib.game_master.space_ship.GameMaster',
        'contrib.entity.conversations_with_ai_companions.HumanUserEntity',
        'contrib.entity.conversations_with_ai_companions.AICompanionEntity',
    ],
)
def test_actual_supported_components_dependencies_order_and_freshness(key):
  definition = prefab_catalog.discover()[0][key].prefab

  def build():
    prefab = copy.deepcopy(definition)
    prefab.entities = [type('Player', (), {'name': 'Alice'})()]
    prefab.params = {
        **prefab.params,
        'extra_components': {
            'first': constant.Constant('one'),
            'second': constant.Constant('two'),
        },
        'extra_components_index': {'first': 100000, 'second': 100000},
        'extra_components_dependencies': {'first': ['__memory__']},
    }
    return prefab.build(
        mock.Mock(sample_text=mock.Mock(return_value='kitchen|room')), memory()
    )

  entity, other = build(), build()
  assert entity.get_component('first').get_state()['state'] == 'one'
  assert entity.get_component('first') is not other.get_component('first')
  assert '__memory__' in entity.get_all_context_components()
  assert list(entity.get_all_context_components())[-2:] == ['first', 'second']
  definition.entities = [type('Player', (), {'name': 'Alice'})()]
  definition.params = {
      **definition.params,
      'extra_components': {'added': constant.Constant('one')},
      'extra_components_dependencies': {'added': ['not_a_real_dependency']},
  }
  with pytest.raises(ValueError, match='missing prefab dependencies'):
    definition.build(
        mock.Mock(sample_text=mock.Mock(return_value='kitchen|room')), memory()
    )


def test_collision_indices_and_shared_objects_rejected():
  prefab = prefab_catalog.discover()[0]['entity.minimal.Entity'].prefab
  for params, message in [
      (
          {'extra_components': {'__memory__': constant.Constant('x')}},
          'replace built-in',
      ),
      (
          {
              'goal': 'goal',
              'extra_components': {'Goal': constant.Constant('x')},
          },
          'replace built-in',
      ),
      (
          {
              'extra_components': {'a': constant.Constant('x')},
              'extra_components_index': {'other': 0},
          },
          'same keys',
      ),
      (
          {
              'extra_components': {'a': constant.Constant('x')},
              'extra_components_index': {'a': True},
          },
          'integers',
      ),
  ]:
    prefab.params = params
    with pytest.raises(ValueError, match=message):
      prefab.build(no_language_model.NoLanguageModel(), memory())
  component = constant.Constant('x')
  prefab.params = {'extra_components': {'a': component, 'b': component}}
  with pytest.raises(ValueError, match='distinct component objects'):
    prefab.build(no_language_model.NoLanguageModel(), memory())


def test_shared_commands_build_discovered_prefab_and_preserve_saved_presets():
  registry = fixtures.scene_registry()
  saved = registry.default_document('scenes-v1')
  journal = {
      'document': copy.deepcopy(saved),
      'metadata': {
          'catalog': registry.catalog(saved),
          'component_catalog': registry.component_catalog(saved),
      },
  }
  for line in [
      'add instance game_master.generic.GameMaster --id Rules',
      'add component constant Rules --id reminder',
      'set components:reminder params.state \'"Keep time for discussion"\'',
      'duplicate Rules',
  ]:
    journal = session_draft.apply(journal, session_commands.parse(line))[
        'journal'
    ]
  document = journal['document']
  assert registry.loads(registry.dumps(document)) == document
  config = registry.to_config(document)
  built = fixtures.build(config)
  assert len(built.get_game_masters()) == 3
  assert (
      fixtures.as_agent(built.get_game_masters()[1])
      .get_component('authored_reminder')
      .get_state()['state']
      == 'Keep time for discussion'
  )
  assert registry.default_document('scenes-v1') == saved
  document['instances'][-1]['prefab'] = 'os.system'
  with pytest.raises(project_config.ValidationError, match='trusted template'):
    registry.normalize(document)


def test_browser_picker_and_typed_commands_share_discovered_catalog():
  browser_api = pytest.importorskip('playwright.sync_api')
  registry = fixtures.scene_registry()
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      registry.default_document('scenes-v1'),
      run_with_steps=mock.Mock(side_effect=AssertionError('no run')),
      integrated=True,
      preview=lambda config: fixtures.build(config).make_checkpoint_data(),
  )
  service = server.operation_service
  assert service is not None
  errors = []

  def route_request(route):
    path = urlsplit(route.request.url).path
    if path == '/':
      route.fulfill(content_type='text/html', body=server.html_content)
    elif path == '/api/state':
      route.fulfill(json=service.snapshot('developer'))
    elif path == '/api/dispatch':
      try:
        route.fulfill(
            json=service.dispatch('developer', route.request.post_data_json)
        )
      except operation_service.OperationError as error:
        route.fulfill(
            status=400,
            json={'error': {'code': error.code, 'message': str(error)}},
        )
    else:
      route.fulfill(status=404, body='No external network')

  with browser_api.sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    page.add_init_script('window.EventSource=class {close(){}};')
    page.route('**/*', route_request)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://localhost/')
    picker = page.get_by_role('combobox', name='Prefab or named preset')
    picker.select_option('game_master.generic.GameMaster')
    page.get_by_role('button', name='Add instance', exact=True).click()
    page.get_by_role('button', name='Save draft', exact=True).click()
    browser_api.expect(
        page.get_by_role('button', name='Save draft', exact=True)
    ).to_be_enabled()
    page.locator('[data-tab="log"]').click()
    command = page.get_by_role('textbox', name='Simulation log command')

    def send(line):
      command.fill(line)
      command.press('Enter')
      browser_api.expect(
          page.get_by_role('button', name='Send', exact=True)
      ).to_be_enabled()

    send(
        'add instance'
        ' contrib.entity.conversations_with_ai_companions.HumanUserEntity --id'
        ' Ship'
    )
    send('add component constant Ship --id reminder')
    send('set components:reminder params.state \'"Shared component text"\'')
    send('save')
    document = server.get_project()['document']
    assert any(
        item['prefab'] == 'game_master.generic.GameMaster'
        for item in document['instances']
    )
    assert any(item['id'] == 'Ship' for item in document['instances'])
    assert (
        document['components'][-1]['params']['state'] == 'Shared component text'
    )
    before = copy.deepcopy(server.get_project())
    send('add instance entity.rational.Entity --id Unsupported')
    send('add component constant Unsupported --id unsupported-note')
    send('save')
    browser_api.expect(
        page.get_by_role('log', name='Simulation log')
    ).to_contain_text('extra_components not implemented yet')
    assert (
        'not implemented yet'
        not in page.locator('#editor-toolbar').inner_text()
    )
    assert server.get_project() == before
    assert not errors
    browser.close()


def test_discovered_reference_ids_and_constructor_owned_defaults():
  registry = fixtures.scene_registry()
  document = registry.default_document('scenes-v1')
  catalog = registry.catalog(document)
  entry = next(
      x for x in catalog if x['key'] == 'game_master.dialogic.GameMaster'
  )
  assert entry['references'] == {'next_game_master_name': 'game_master'}
  assert entry['instance']['params']['next_game_master_name'] == 'conversation'
  journal = {'document': document, 'metadata': {'catalog': catalog}}
  journal = session_draft.apply(
      journal,
      session_commands.parse(
          'add instance game_master.dialogic.GameMaster --id Rules'
      ),
  )['journal']
  document = journal['document']
  document['instances'][2]['params']['name'] = 'Renamed conversation'
  config = registry.to_config(document)
  assert (
      config.instances[-1].params['next_game_master_name']
      == 'Renamed conversation'
  )
  assert {
      'value': 'conversation',
      'label': 'Renamed conversation',
  } in registry.inspector(document)['Rules']['next_game_master_name']['choices']
  document['instances'][-1]['params']['next_game_master_name'] = 'default rules'
  with pytest.raises(
      project_config.ValidationError, match='target instance ID'
  ):
    registry.normalize(document)
  entry = next(
      x for x in catalog if x['key'] == 'game_master.marketplace.GameMaster'
  )
  assert 'experiment_component_class' in entry['fixed_parameters']
  assert 'experiment_component_class' not in entry['instance']['params']


def test_missing_dependency_fails_shared_save_atomically():

  base = fixtures.scene_registry()._templates['scenes-v1']
  types = dict(base.component_types)
  types['constant'] = dataclasses.replace(
      types['constant'], dependencies=('missing',)
  )
  registry = project_config.Registry(
      {'scenes-v1': dataclasses.replace(base, component_types=types)}
  )
  document = registry.default_document('scenes-v1')
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      document,
      run_with_steps=mock.Mock(),
      integrated=True,
      preview=lambda config: fixtures.build(config).make_checkpoint_data(),
  )
  service = server.operation_service
  assert service is not None
  before = service.snapshot('developer')
  document['components'] = [
      dict(
          id='note',
          instance='alice',
          type='constant',
          name='Note',
          params={'state': 'hello', 'pre_act_label': 'Note'},
      )
  ]
  with pytest.raises(
      operation_service.OperationError, match='missing prefab dependencies'
  ):
    service.dispatch(
        'developer',
        dict(
            operation='project.save',
            arguments={'text': json.dumps(document), 'revision': 0},
            revision=service.revision,
            references=copy.deepcopy(service.references),
            retry_key='missing-dependency',
        ),
    )
  assert service.snapshot('developer') == before


def test_custom_recipe_restrictions_and_installed_name_collisions():
  base = fixtures.scene_registry()._templates['scenes-v1']
  types = dict(base.component_types)
  types['constant'] = dataclasses.replace(
      types['constant'], all_prefabs=False, prototypes=('alice',)
  )
  registry = project_config.Registry(
      {'scenes-v1': dataclasses.replace(base, component_types=types)}
  )
  document = registry.default_document('scenes-v1')
  assert next(
      x for x in registry.component_catalog(document) if x['key'] == 'constant'
  )['prototypes'] == ['alice']
  document['components'] = [
      dict(
          id='note',
          instance='bob',
          type='constant',
          name='Note',
          params={'state': 'x', 'pre_act_label': 'Note'},
      )
  ]
  with pytest.raises(project_config.ValidationError, match='compatible'):
    registry.normalize(document)

  def conflicting():
    config = fixtures.scene_config()
    config.prefabs = {
        **config.prefabs,
        'entity.minimal.Entity': config.prefabs['minimal'],
    }
    return config

  registry = project_config.Registry(
      {'scenes-v1': dataclasses.replace(base, factory=conflicting)}
  )
  with pytest.raises(project_config.ValidationError, match='collides'):
    registry.catalog(registry.default_document('scenes-v1'))


def test_reference_documentation_commands_use_real_shared_cli_preview(tmp_path):
  registry = fixtures.scene_registry()
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      registry.default_document('scenes-v1'),
      run_with_steps=mock.Mock(side_effect=AssertionError('no run')),
      integrated=True,
      preview=lambda config: fixtures.build(config).make_checkpoint_data(),
  )
  text = (Path(__file__).parents[1] / 'docs/editor-commands.md').read_text()
  block = (
      text.split('### Installed prefab discovery and component construction')[1]
      .split('```text\n')[1]
      .split('```')[0]
  )
  args = argparse.Namespace(
      url='http://fixture',
      timeout=1,
      line='',
      draft=tmp_path / 'draft.json',
      file=None,
      output=None,
  )
  with mock.patch.object(
      concordia_session.urllib.request,
      'urlopen',
      side_effect=session_commands_test.fake_http(server),
  ):
    for line in block.strip().splitlines():
      args.line = line
      concordia_session.friendly(args)
  document = server.get_project()['document']
  assert document['instances'][-1]['prefab'] == 'game_master.generic.GameMaster'
  assert (
      document['components'][-1]['params']['state']
      == 'Keep time for the discussion.'
  )
  assert server.get_project()['revision'] == 1

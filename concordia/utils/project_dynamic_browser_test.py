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

"""Generic component/HTML editor integration with intercepted network and no runs."""

import copy
import json
import threading
from unittest import mock
from urllib.parse import urlsplit

from concordia.components.game_master import scene_tracker
from concordia.contrib.components.game_master import forum
from concordia.environment.engines import asynchronous
from concordia.environment.engines import sequential
from concordia.environment.engines import simultaneous
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import project_test_support as fixtures
from concordia.utils import session_commands
from concordia.utils import session_draft
from concordia.utils import simulation_server
from concordia.utils.project_operations_test import eventually
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_execution():
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No simulation')
  ), mock.patch.object(
      simulation_server.SimulationServer,
      'start',
      side_effect=AssertionError('No listener'),
  ), mock.patch.object(
      no_language_model.NoLanguageModel,
      'sample_text',
      side_effect=AssertionError('No model'),
  ), mock.patch.object(
      no_language_model.NoLanguageModel,
      'sample_choice',
      side_effect=AssertionError('No model'),
  ):
    yield


def make_server():
  registry = fixtures.scene_registry()
  fs = forum.ForumState(player_names=['Alice', 'Bob'])
  fs.create_post(
      author='Alice', title='Robot alchemy', content='Careful experiments'
  )
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry,
      registry.default_document('scenes-v1'),
      integrated=True,
      run=mock.Mock(side_effect=AssertionError('No run')),
      preview=fixtures.build,
      viewers={
          'Forum fixture': fs.to_html,
          'Notebook': lambda: (
              '<html><style>h1{color:rgb(10, 20,'
              ' 30)}</style><h1>Notebook</h1><button'
              ' id="counter">0</button><script>counter.onclick=()=>counter.textContent=Number(counter.textContent)+1;</script></html>'
          ),
      },
  )
  return registry, server, fs


def test_generic_draft_cli_roundtrip_undo_validation_and_engine_identity():
  registry, server, _ = make_server()
  document = server.get_project()['document']
  assert not {'groups', 'scene_types', 'scenes'} & document.keys()
  metadata = server._project_editor.definition_view
  journal = {'document': document, 'metadata': metadata, 'selectedId': 'alice'}

  def command(line):
    nonlocal journal
    journal = session_draft.apply(journal, session_commands.parse(line))[
        'journal'
    ]

  command('state-field alice Instructions state \'"Listen carefully."\'')
  command(
      'state-field conversation __next_game_master__ scenes'
      ' \'[{"num_rounds":5}]\''
  )
  changed = copy.deepcopy(journal['document'])
  command('undo')
  assert 'conversation' not in journal['document']['dynamic_states']
  command('redo')
  assert journal['document'] == changed
  server.replace_project(json.dumps(changed), 0)
  assert server.get_project()['document'] == registry.loads(
      registry.dumps(changed)
  )
  entities = server._project_editor.definition_view['entities']
  assert (
      entities['entity_0']['component_info']['context_components'][
          'Instructions'
      ]['state']['state']
      == 'Listen carefully.'
  )
  assert (
      entities['entity_2']['component_info']['context_components'][
          '__next_game_master__'
      ]['state']['scenes'][0]['num_rounds']
      == 5
  )
  invalid = copy.deepcopy(changed)
  invalid['dynamic_states']['conversation']['__next_game_master__'][
      'scenes'
  ] = [{'num_rounds': 0}]
  with pytest.raises(
      ValueError, match='dynamic_states.conversation.*num_rounds'
  ):
    server.replace_project(json.dumps(invalid), 1)
  assert server.get_project()['document'] == changed
  # Reset remains available even when an optional component disappears.
  journal['document']['instances'][0]['params']['goal'] = 'A goal'
  server.replace_project(json.dumps(journal['document']), 1)
  journal['metadata'] = server._project_editor.definition_view
  command('state-field alice Goal state \'"Override goal"\'')
  journal['document']['instances'][0]['params']['goal'] = ''
  with pytest.raises(ValueError, match='Goal'):
    server.replace_project(json.dumps(journal['document']), 2)
  command('state-reset alice Goal')
  server.replace_project(json.dumps(journal['document']), 2)
  assert (
      'Goal'
      not in server._project_editor.definition_view['entities']['entity_0'][
          'component_info'
      ]['context_components']
  )
  for engine in [
      sequential.Sequential(),
      simultaneous.Simultaneous(),
      asynchronous.Asynchronous(),
  ]:
    simulation = generic.Simulation(
        config=registry.to_config(changed),
        model=no_language_model.NoLanguageModel(),
        embedder=lambda _: np.ones(8),
        engine=engine,
    )
    registry.apply_dynamic_states(changed, simulation)
    assert simulation.make_checkpoint_data()['engine'] == {
        'name': type(engine).__name__,
        'module': type(engine).__module__,
    }


def test_generic_runtime_edit_player_and_game_master_at_real_boundary():
  registry, server, _ = make_server()
  document = server.get_project()['document']
  editor = server._project_editor
  config = registry.to_config(document)
  editor.begin(config)
  simulation = fixtures.build(config)
  server.set_simulation(simulation)
  server._project_run = {'status': 'active'}
  controller = server.step_controller
  controller.pause()
  waiter = threading.Thread(target=controller.wait_for_step_permission)
  waiter.start()
  try:
    eventually(lambda: controller.at_pause_boundary)
    from concordia.utils.project_operations_test import dispatch

    for line in [
        'edit alice Instructions state \'"Runtime only"\'',
        'edit conversation __next_game_master__ scenes \'[{"num_rounds":7}]\'',
    ]:
      planned = session_commands.operation(
          session_commands.parse(line), editor.snapshot()
      )
      assert planned is not None
      operation, args = planned
      dispatch(editor, operation, args)
    assert (
        fixtures.as_agent(simulation.get_entities()[0])
        .get_component('Instructions')
        .get_state()['state']
        == 'Runtime only'
    )
    tracker = fixtures.as_agent(simulation.get_game_masters()[0]).get_component(
        '__next_game_master__', type_=scene_tracker.SceneTracker
    )
    before = json.loads(json.dumps(tracker.get_state()))
    assert before['scenes'][0]['num_rounds'] == 7
    with pytest.raises(operation_service.OperationError, match='num_rounds'):
      dispatch(
          editor,
          'runtime.edit_state',
          {
              'instance_id': 'conversation',
              'component': '__next_game_master__',
              'field': 'scenes',
              'value': '[{"num_rounds":0}]',
          },
      )
    assert tracker.get_state() == before
    with pytest.raises(operation_service.OperationError, match='not a dynamic'):
      dispatch(
          editor,
          'runtime.edit_state',
          {
              'instance_id': 'alice',
              'component': 'Instructions',
              'field': 'pre_act_label',
              'value': '"bad"',
          },
      )
    with pytest.raises(operation_service.OperationError, match='expected str'):
      dispatch(
          editor,
          'runtime.edit_state',
          {
              'instance_id': 'alice',
              'component': 'Instructions',
              'field': 'state',
              'value': '[]',
          },
      )
    assert server.get_project()['document'] == document
  finally:
    controller.stop()
    waiter.join(3)
  assert not waiter.is_alive()


def test_component_fields_and_generic_viewers_in_chromium(tmp_path):
  api = pytest.importorskip('playwright.sync_api')
  registry, server, fs = make_server()
  service = server.operation_service
  errors = []

  def intercept(route):
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

  with api.sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    page.set_default_timeout(5000)
    page.route('**/*', intercept)
    page.add_init_script('window.EventSource=class {close(){}};')
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://localhost/')
    expect = api.expect
    expect(page.locator('#editor-engine')).to_contain_text(
        'sequential.Sequential'
    )
    page.locator('[data-instance-id="alice"]').click()
    page.locator('#editor-alice-goal').fill('Keep this unsaved draft')
    for name in ['Notebook', 'Forum fixture', 'default', 'Notebook']:
      page.locator('#editor-viewer').select_option(name)
      if name == 'Notebook':
        frame = page.frame_locator('#editor-viewer-frame')
        expect(frame.locator('h1')).to_have_text('Notebook')
        frame.locator('#counter').click()
        expect(frame.locator('#counter')).to_have_text('1')
        assert (
            frame.locator('h1').evaluate('el=>getComputedStyle(el).color')
            == 'rgb(10, 20, 30)'
        )
      elif name == 'Forum fixture':
        expect(
            page.frame_locator('#editor-viewer-frame').locator('body')
        ).to_contain_text('Robot alchemy')
      else:
        expect(page.locator('.svg-container')).to_be_visible()
      expect(page.locator('#editor-alice-goal')).to_have_value(
          'Keep this unsaved draft'
      )
    page.locator('#editor-viewer').select_option('Forum fixture')
    fs.create_post(
        author='Bob', title='New result', content='A successful refresh'
    )
    page.get_by_role('button', name='Refresh viewer', exact=True).click()
    expect(
        page.frame_locator('#editor-viewer-frame').locator('body')
    ).to_contain_text('New result')
    command = page.get_by_role('textbox', name='Simulation log command')

    def send(line):
      expect(
          page.get_by_role('button', name='Save draft', exact=True)
      ).to_be_enabled()
      expect(
          page.get_by_role('button', name='Send', exact=True)
      ).to_be_enabled()
      command.fill(line)
      command.press('Enter')
      expect(
          page.get_by_role('button', name='Send', exact=True)
      ).to_be_enabled()

    send('viewer-refresh Notebook')
    expect(
        page.frame_locator('#editor-viewer-frame').locator('h1')
    ).to_have_text('Notebook')
    send('viewer default')
    send('viewer-refresh')
    expect(page.locator('.svg-container')).to_be_visible()
    page.get_by_role('button', name='Save draft', exact=True).click()
    expect(page.locator('#editor-status')).to_contain_text('saved definition')
    page.locator('[data-instance-id="alice"]').click()
    page.locator('#toggle_comp_Instructions').click()
    page.get_by_label('JSON value for Instructions.state', exact=True).check()
    page.locator('#dyn_Instructions_state').fill('null')
    page.locator(
        '.dynamic-save-btn[data-component="Instructions"][data-state-key="state"]'
    ).click()
    page.get_by_role('button', name='Save draft', exact=True).click()
    expect(page.locator('#console-output')).to_contain_text(
        'state: expected str'
    )
    page.get_by_role(
        'button', name='Reset Instructions state to prefab', exact=True
    ).click()
    send('save')
    page.locator('[data-instance-id="conversation"]').click()
    dynamic = page.locator(
        '.dynamic-save-btn[data-component="__next_game_master__"][data-state-key="scenes"]'
    )
    input_id = dynamic.get_attribute('data-input-id')
    page.locator('#toggle_comp___next_game_master__').click()
    page.locator('#' + input_id).fill('[{"num_rounds":0}]')
    dynamic.click()
    page.get_by_role('button', name='Save draft', exact=True).click()
    expect(page.locator('#console-output')).to_contain_text('num_rounds')
    assert server.get_project()['document']['dynamic_states'] == {}
    send(
        'state-field conversation __next_game_master__ scenes'
        ' \'[{"num_rounds":6}]\''
    )
    send('save')
    expect(page.locator('#editor-status')).to_contain_text('saved definition')
    assert server.get_project()['document']['dynamic_states']['conversation'][
        '__next_game_master__'
    ]['scenes'] == [{'num_rounds': 6}]
    page.get_by_role(
        'button', name='Reset __next_game_master__ scenes to prefab', exact=True
    ).click()
    send('save')
    expect(page.locator('#editor-status')).to_contain_text('saved definition')
    assert server.get_project()['document']['dynamic_states'] == {}
    local_html = tmp_path / 'local.html'
    local_html.write_text(
        '<h1>Local experiment</h1><a href="#details">Details</a><p'
        ' id="details">Local details</p>'
    )
    with page.expect_file_chooser() as chooser:
      page.get_by_role('button', name='Open viewer HTML', exact=True).click()
    chooser.value.set_files(local_html)
    frame = page.frame_locator('#editor-viewer-frame')
    expect(frame.locator('h1')).to_have_text('Local experiment')
    frame.get_by_role('link', name='Details').click()
    expect(frame.locator('#details')).to_be_visible()
    send('viewer default')
    expect(page.locator('.svg-container')).to_be_visible()

    # Bind an actual component graph and acknowledge a controller boundary,
    # without invoking Simulation.play or an engine loop.
    editor = server._project_editor
    config = registry.to_config(server.get_project()['document'])
    editor.begin(config)
    simulation = fixtures.build(config)
    server.set_simulation(simulation)
    server._project_run = {'status': 'active'}
    server.broadcast_entity_info(simulation.make_checkpoint_data())
    controller = server.step_controller
    controller.pause()
    waiter = threading.Thread(target=controller.wait_for_step_permission)
    waiter.start()
    try:
      eventually(lambda: controller.at_pause_boundary)
      page.reload()
      page.locator('#editor-mode').select_option('runtime')
      page.locator('[data-instance-id="conversation"]').click()
      page.locator('#toggle_comp___next_game_master__').click()
      dynamic = page.locator(
          '.dynamic-save-btn[data-component="__next_game_master__"][data-state-key="scenes"]'
      )
      input_id = dynamic.get_attribute('data-input-id')
      page.locator('#' + input_id).fill('[{"num_rounds":0}]')
      dynamic.click()
      expect(page.locator('#console-output')).to_contain_text('num_rounds')
      page.locator('#' + input_id).fill('[{"num_rounds":7}]')
      dynamic.click()
      expect(page.locator('#console-output')).to_contain_text(
          'runtime.edit_state completed'
      )
      tracker = fixtures.as_agent(
          simulation.get_game_masters()[0]
      ).get_component('__next_game_master__', type_=scene_tracker.SceneTracker)
      assert tracker._max_rounds == 7
      assert server.get_project()['document']['dynamic_states'] == {}
    finally:
      controller.stop()
      waiter.join(3)
    page.screenshot(path=str(tmp_path / 'generic-viewer.png'), full_page=True)
    assert not errors
    browser.close()


def test_cli_registered_viewer_exports_explicit_file(tmp_path):
  import argparse

  from concordia.command_line_interface import concordia_session
  from concordia.utils.session_commands_test import fake_http

  _, server, _ = make_server()
  destination = tmp_path / 'notebook.html'
  args = argparse.Namespace(
      url='http://fixture',
      timeout=1,
      line='viewer Notebook',
      draft=None,
      file=destination,
      output=None,
  )
  with mock.patch.object(
      concordia_session.urllib.request, 'urlopen', side_effect=fake_http(server)
  ):
    result = concordia_session.friendly(args)
    assert result['output'] == str(destination)
    assert '<h1>Notebook</h1>' in destination.read_text()
    args.line = 'viewer'
    args.file = None
    assert concordia_session.friendly(args)['available'] == [
        'default',
        'Forum fixture',
        'Notebook',
    ]
    args.line = 'viewer-refresh'
    with pytest.raises(ValueError, match='viewer-refresh NAME'):
      concordia_session.friendly(args)
    args.line = 'viewer-refresh Notebook'
    assert '<h1>Notebook</h1>' in concordia_session.friendly(args)['html']
    args.line = 'viewer-url javascript:alert(1)'
    with pytest.raises(ValueError, match='HTTP'):
      concordia_session.friendly(args)
    args.line = 'viewer Missing'
    with pytest.raises(ValueError, match='Unknown registered viewer'):
      concordia_session.friendly(args)
  assert server.get_project()['run']['status'] == 'not_started'


def test_dynamic_field_validation_belongs_to_component_not_current_value_type():
  from concordia.typing import entity_component

  class OptionalNumber(entity_component.ContextComponent):

    def __init__(self):
      self.value = None

    def get_state(self):
      return {'value': self.value}

    def get_dynamic_state(self):
      return self.get_state()

    def set_state(self, state):
      value = state.get('value', self.value)
      if value is not None and type(value) not in (int, float):
        raise ValueError('value: expected optional number')
      self.value = value

  config = fixtures.make_config()
  component = OptionalNumber()
  config.instances[0].params = {**config.instances[0].params, 'extra_components': {'OptionalNumber': component}}  # pyrefly: ignore[bad-assignment]
  simulation = fixtures.build(config)
  for value in [1, 2.5, None, 3]:
    simulation.set_component_dynamic_state(
        'Alice', 'OptionalNumber', 'value', value
    )
    assert (
        fixtures.as_agent(simulation.get_entities()[0])
        .get_component('OptionalNumber')
        .get_state()['value']
        == value
    )
  with pytest.raises(ValueError, match='optional number'):
    simulation.set_component_dynamic_state(
        'Alice', 'OptionalNumber', 'value', 'bad'
    )
  assert (
      fixtures.as_agent(simulation.get_entities()[0])
      .get_component('OptionalNumber')
      .get_state()['value']
      == 3
  )

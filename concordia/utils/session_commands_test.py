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

"""Shared command, local draft and log contracts without models or listeners."""

import argparse
import io
import json
import shlex
from unittest import mock
from urllib.parse import urlsplit

from concordia.command_line_interface import concordia_log
from concordia.command_line_interface import concordia_log_test
from concordia.command_line_interface import concordia_session
from concordia.utils import operation_service
from concordia.utils import project_run_steps_test
from concordia.utils import project_test_support
from concordia.utils import session_commands
from concordia.utils import session_draft
from concordia.utils import simulation_server
import pytest


@pytest.mark.parametrize(
    'line',
    [
        'run --steps nope',
        'run --shell x',
        'step 2',
        'log step 1',
        'log bogus --source current',
        'reload',
        'view bogus',
        'python x',
        '$(whoami)',
        'set x goal',
        'help extra',
        'discover extra',
        'watch extra',
        'call',
        'call x []',
        'call x nope',
        '',
    ],
)
def test_invalid_commands(line):
  with pytest.raises(ValueError):
    session_commands.parse(line)


def test_shared_control_mapping_and_run_resume_distinction():
  state = {
      'state': 'ready',
      'revision': 7,
      'run_limits': {'default_requested_steps': 10},
  }
  assert session_commands.operation(
      session_commands.parse('run --steps 12'), state
  ) == ('project.run', {'revision': 7, 'requested_steps': 12})
  assert session_commands.operation(session_commands.parse('step'), state) == (
      'runtime.step',
      {},
  )
  assert session_commands.parse('log step --source current 1')['kind'] == 'log'
  state['state'] = 'paused'
  with pytest.raises(ValueError, match='play'):
    session_commands.operation(session_commands.parse('run'), state)
  assert session_commands.operation(session_commands.parse('play'), state) == (
      'runtime.play',
      {},
  )


def test_node_draft_reuses_existing_actions_and_history():
  registry = project_test_support.scene_registry()
  document = registry.default_document('scenes-v1')
  journal = {
      'document': document,
      'selectedId': 'alice',
      'metadata': {
          'catalog': registry.catalog(document),
          'component_catalog': registry.component_catalog(document),
      },
  }
  original = json.loads(json.dumps(document))
  added = session_draft.apply(
      journal, session_commands.parse('duplicate alice')
  )
  journal = added['journal']
  assert len(journal['document']['instances']) == len(original['instances']) + 1
  assert journal['selectedId'] != 'alice'
  journal = session_draft.apply(journal, session_commands.parse('undo'))[
      'journal'
  ]
  assert journal['document'] == original
  journal = session_draft.apply(journal, session_commands.parse('redo'))[
      'journal'
  ]
  assert len(journal['document']['instances']) == len(original['instances']) + 1
  journal = session_draft.apply(
      journal,
      session_commands.parse(
          'set alice params.goal ' + shlex.quote(json.dumps('Quiet music'))
      ),
  )['journal']
  assert journal['document']['instances'][0]['params']['goal'] == 'Quiet music'
  with pytest.raises(ValueError, match='editable'):
    session_draft.apply(
        journal, session_commands.parse('set alice __proto__ {}')
    )


@pytest.mark.parametrize(
    'command,args',
    [
        ('overview', []),
        ('entities', []),
        ('actions', ['Alice']),
        ('context', ['Alice', '--step', '1']),
        ('step', ['1']),
        ('timeline', ['Alice']),
        ('search', ['hello']),
        ('memories', ['Alice']),
        ('components', ['--entity', 'Alice']),
        ('dump', []),
        ('bundle', []),
    ],
)
def test_all_existing_log_handlers_in_memory(command, args):
  log = concordia_log_test._create_sample_log()
  with mock.patch('builtins.open', side_effect=AssertionError('No filesystem')):
    result = concordia_log.analyze(log, command, args)
  assert result['text']
  if command in ('dump', 'bundle'):
    assert result['download']['content']


def fake_http(server):
  def request(req, **_kwargs):
    path = urlsplit(req.full_url).path
    service = server.operation_service
    if path == '/api/state':
      result = service.snapshot('developer')
    elif path == '/api/operations':
      result = service.discover('developer')
    else:
      result = service.dispatch('developer', json.loads(req.data))
    return io.BytesIO(json.dumps(result).encode())

  return request


def test_cli_local_draft_save_and_run_use_same_service(tmp_path):
  runner = mock.Mock()
  server = project_run_steps_test.editor(runner=runner)
  args = argparse.Namespace(
      url='http://fixture',
      timeout=1,
      line='set alice params.goal ' + shlex.quote(json.dumps('Quiet music')),
      draft=tmp_path / 'draft.json',
      file=None,
      output=None,
  )
  with mock.patch.object(
      concordia_session.urllib.request, 'urlopen', side_effect=fake_http(server)
  ):
    concordia_session.friendly(args)
    assert (
        server.get_project()['document']['instances'][0]['params']['goal']
        != 'Quiet music'
    )
    args.line = 'run --steps 3'
    with pytest.raises(ValueError, match='Unsaved'):
      concordia_session.friendly(args)
    args.line = "call project.reset '{}'"
    with pytest.raises(ValueError, match='Unsaved'):
      concordia_session.friendly(args)
    args.line = 'save'
    concordia_session.friendly(args)
    assert (
        server.get_project()['document']['instances'][0]['params']['goal']
        == 'Quiet music'
    )
    args.line = 'run --steps 3'
    concordia_session.friendly(args)
    project_run_steps_test.joined(server)
    assert runner.call_args.args[1] == 3


def test_service_command_errors_and_log_scope_are_actionable():
  server = project_run_steps_test.editor()
  service = server.operation_service
  with pytest.raises(operation_service.OperationError, match='Unknown command'):
    service.dispatch(
        'developer',
        {'operation': 'session.plan', 'arguments': {'line': 'exec foo'}},
    )
  values = {
      'command': 'overview',
      'source': 'current',
      'arguments': '[]',
      'imported': '',
  }
  with pytest.raises(operation_service.OperationError, match='No current'):
    service.dispatch(
        'developer', {'operation': 'log.query', 'arguments': values}
    )
  server.set_project_log(concordia_log_test._create_sample_log())
  result = service.dispatch(
      'developer', {'operation': 'log.query', 'arguments': values}
  )['result']
  assert result['source'] == 'current'
  assert 'Alice' in result['text']


def test_console_real_browser_single_sink_commands_and_draft_parity(tmp_path):
  browser_api = pytest.importorskip('playwright.sync_api')
  registry = project_test_support.scene_registry()
  server = simulation_server.SimulationServer(port=0)
  document = registry.default_document('scenes-v1')
  server.configure_project(
      registry,
      document,
      run_with_steps=mock.Mock(),
      integrated=True,
      title='Test editor · Fixture mock notice',
  )
  service = server.operation_service
  assert service is not None
  errors = []
  requests = []
  offline = [False]

  def route_request(route):
    path = urlsplit(route.request.url).path
    if offline[0]:
      route.abort('failed')
      return
    if path == '/':
      route.fulfill(content_type='text/html', body=server.html_content)
    elif path == '/api/state':
      route.fulfill(json=service.snapshot('developer'))
    elif path == '/api/operations':
      route.fulfill(json=service.discover('developer'))
    elif path == '/api/dispatch':
      body = route.request.post_data_json
      requests.append(body)
      try:
        route.fulfill(json=service.dispatch('developer', body))
      except operation_service.OperationError as error:
        route.fulfill(
            status=409,
            json={'error': {'code': error.code, 'message': str(error)}},
        )
    else:
      route.fulfill(status=404, body='No network')

  with (
      mock.patch.object(
          server, 'start', side_effect=AssertionError('No listener')
      ),
      browser_api.sync_playwright() as playwright,
  ):
    browser = playwright.chromium.launch()
    with browser.new_context(viewport={'width': 412, 'height': 915}) as context:
      context.add_init_script(
          'window.streamCount=0;window.EventSource=class'
          ' {constructor(){window.streamCount++;}close(){}};'
      )
      context.route('**/*', route_request)
      page = context.new_page()
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.goto('http://localhost/')
      page.locator('[data-tab="log"]').click()
      command = page.get_by_role('textbox', name='Simulation log command')
      output = page.get_by_role('log', name='Simulation log')
      browser_api.expect(
          page.get_by_role('button', name='Run', exact=True)
      ).to_be_enabled()

      def send(line):
        command.fill(line)
        command.press('Enter')
        browser_api.expect(
            page.get_by_role('button', name='Send', exact=True)
        ).to_be_enabled()

      send('discover')
      browser_api.expect(output).to_contain_text('session.plan')
      send('watch')
      send('watch')
      browser_api.expect(output).to_contain_text('Live stream already enabled')
      assert page.evaluate('window.streamCount') == 1
      send('call session.plan \'{"line":"help"}\'')
      browser_api.expect(output).to_contain_text('Files and draft/history')
      browser_api.expect(output).to_contain_text('Fixture mock notice')
      assert (
          'Fixture mock notice'
          not in page.locator('#editor-toolbar').inner_text()
      )
      send('help')
      browser_api.expect(output).to_contain_text('run [--steps N]')
      command.press('ArrowUp')
      browser_api.expect(command).to_have_value('help')
      command.press('ArrowDown')
      browser_api.expect(command).to_have_value('')
      before_ime = len(requests)
      assert (
          command.evaluate(
              'node=>node.dispatchEvent(new'
              " KeyboardEvent('keydown',{key:'Enter',isComposing:true,cancelable:true,bubbles:true}))"
          )
          is False
      )
      assert len(requests) == before_ime
      send('set alice params.goal ' + shlex.quote(json.dumps('Quiet music')))
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'unsaved changes'
      )
      assert server.get_project()['document'] == document
      # GUI Undo shares the command's history; command Redo restores it.
      page.get_by_role('button', name='Undo', exact=True).click()
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'saved definition'
      )
      send('redo')
      send("call project.reset '{}'")
      browser_api.expect(output).to_contain_text('Unsaved client fields')
      assert not any(item['operation'] == 'project.reset' for item in requests)
      send('run --steps 1')
      browser_api.expect(output).to_contain_text(
          'Unsaved draft: save before run'
      )
      assert server.get_project()['run']['status'] == 'not_started'
      send('set simulation max_steps 0')
      send('validate')
      browser_api.expect(
          output.get_by_role('button', name='Show invalid field')
      ).to_be_visible()
      assert page.locator('#editor-error').count() == 0
      assert (
          'expected integer' not in page.locator('#editor-toolbar').inner_text()
      )
      output.get_by_role('button', name='Show invalid field').click()
      browser_api.expect(page.locator('#editor-max-steps')).to_be_focused()
      page.locator('#editor-max-steps').fill('12')
      page.locator('[data-tab="log"]').click()
      send('save')
      assert server.get_project()['document']['max_steps'] == 12
      assert (
          server.get_project()['document']['instances'][0]['params']['goal']
          == 'Quiet music'
      )
      send('step')
      browser_api.expect(output).to_contain_text('Cannot step while ready')
      send('log overview --source current')
      browser_api.expect(output).to_contain_text('No current structured log')
      server.set_project_log(concordia_log_test._create_sample_log())
      send('log entities --source current')
      browser_api.expect(output).to_contain_text('Log source: current')
      with page.expect_download() as download:
        send('log bundle --source current')
      assert download.value.suggested_filename == 'log.html'
      send('run --steps 3')
      project_run_steps_test.joined(server)
      browser_api.expect(output).to_contain_text(
          'Run accepted; waiting for runner output.'
      )
      count = sum(item['operation'] == 'project.run' for item in requests)
      # New state snapshots / reconnect never resubmit a command.
      offline[0] = True
      page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
      browser_api.expect(output).to_contain_text('Disconnected; drafts kept')
      offline[0] = False
      page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
      browser_api.expect(page.locator('#editor-status')).not_to_contain_text(
          'Disconnected'
      )
      assert (
          sum(item['operation'] == 'project.run' for item in requests) == count
      )
      assert server._project_editor is not None
      server._project_editor.record_step({
          'step': 1,
          'acting_entity': 'Alice',
          'action': 'Unique narrative fixture',
          'entity_actions': {},
          'entity_logs': {},
      })
      page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
      browser_api.expect(output).to_contain_text('Unique narrative fixture')
      browser_api.expect(page.locator('#editor-step-summary')).to_have_text(
          'Step 1 · Alice'
      )
      page.screenshot(path=str(tmp_path / 'console-mobile.png'))
      # Production sink must not pull a reader away from older entries.
      page.evaluate(
          "for(let i=0;i<80;i++)logConsole('scroll fixture"
          " '+i,'info');document.getElementById('console-output').scrollTop=0;"
      )
      page.evaluate("logConsole('new information','info')")
      assert output.evaluate('(node)=>node.scrollTop') == 0
      assert not errors
    browser.close()


def test_full_authored_action_inventory_roundtrips_shared_registry():
  registry = project_test_support.scene_registry()
  document = registry.default_document('scenes-v1')
  journal = {
      'document': document,
      'selectedId': 'alice',
      'metadata': {
          'catalog': registry.catalog(document),
          'component_catalog': registry.component_catalog(document),
      },
  }

  def execute(line):
    nonlocal journal
    result = session_draft.apply(journal, session_commands.parse(line))
    journal = result['journal']
    return result['result']

  execute('add instance alice')
  actor = journal['selectedId']
  execute('set ' + actor + ' params.name ' + shlex.quote(json.dumps('Charlie')))
  execute('add component constant ' + actor)
  component = journal['selectedId']
  execute(
      'set '
      + component
      + ' params.state '
      + shlex.quote(json.dumps('Literal context'))
  )
  execute('move-component ' + component + ' bob')
  execute('duplicate ' + component)
  execute('remove')
  execute('add group')
  group = journal['selectedId']
  execute('set ' + group + ' participants ' + shlex.quote(json.dumps([actor])))
  execute('add scene-type')
  scene_type = journal['selectedId']
  execute(
      'set '
      + scene_type
      + ' group '
      + shlex.quote(json.dumps(group.split(':')[1]))
  )
  execute('add scene')
  scene = journal['selectedId']
  execute(
      'set '
      + scene
      + ' scene_type '
      + shlex.quote(json.dumps(scene_type.split(':')[1]))
  )
  execute('set ' + scene + ' participants ' + shlex.quote(json.dumps([actor])))
  assert execute('references ' + actor)
  execute('replace ' + actor + ' alice')
  execute('remove ' + actor)
  assert (
      registry.loads(registry.dumps(journal['document'])) == journal['document']
  )
  execute('remove ' + scene)
  execute('remove ' + scene_type)
  execute('remove ' + group)
  assert (
      registry.loads(registry.dumps(journal['document'])) == journal['document']
  )


def test_cli_log_exports_and_local_load_are_explicit(tmp_path):
  server = project_run_steps_test.editor()
  log = concordia_log_test._create_sample_log()
  source = tmp_path / 'log.json'
  source.write_text(log.to_json())
  args = argparse.Namespace(
      url='http://fixture',
      timeout=1,
      line='log bundle --source imported',
      draft=None,
      file=source,
      output=tmp_path / 'bundle.html',
  )
  with mock.patch.object(
      concordia_session.urllib.request, 'urlopen', side_effect=fake_http(server)
  ):
    result = concordia_session.friendly(args)
    assert result['output'] == str(args.output)
    assert 'Alice' in args.output.read_text()
    original = server.get_project()['document']
    imported = json.loads(json.dumps(original))
    imported['premise'] = 'An imported local draft'
    source.write_text(json.dumps(imported))
    args.draft = tmp_path / 'journal.json'
    args.line = 'load'
    concordia_session.friendly(args)
    assert server.get_project()['document'] == original
    assert json.loads(args.draft.read_text())['document'] == imported
    with pytest.raises(ValueError, match='Unsaved'):
      concordia_session.friendly(args)


@pytest.mark.parametrize('name', ['discover', 'state', 'call'])
def test_machine_modes_still_emit_json(name, capsys):
  server = project_run_steps_test.editor()
  capsys.readouterr()
  with (
      mock.patch.object(
          concordia_session.urllib.request,
          'urlopen',
          side_effect=fake_http(server),
      ),
      mock.patch.object(
          concordia_session.sys,
          'stdin',
          io.StringIO(
              json.dumps(
                  {'operation': 'session.plan', 'arguments': {'line': 'help'}}
              )
          ),
      ),
  ):
    assert concordia_session.main(['--url', 'http://fixture', name]) == 0
  assert 'result' in json.loads(capsys.readouterr().out)


def test_machine_watch_and_cli_session_binding(tmp_path, capsys):
  with mock.patch.object(
      concordia_session.urllib.request,
      'urlopen',
      return_value=io.BytesIO(b'data: {"revision":1}\n\n'),
  ):
    assert concordia_session.main(['--url', 'http://fixture', 'watch']) == 0
  assert json.loads(capsys.readouterr().out) == {'revision': 1}
  server = project_run_steps_test.editor()
  args = argparse.Namespace(
      url='http://fixture',
      timeout=1,
      line='select alice',
      draft=tmp_path / 'draft.json',
      file=None,
      output=None,
  )
  with mock.patch.object(
      concordia_session.urllib.request, 'urlopen', side_effect=fake_http(server)
  ):
    concordia_session.friendly(args)
    server.operation_service.references['session_id'] = 'a-new-session'
    args.line = 'save'
    with pytest.raises(ValueError, match='Server session changed'):
      concordia_session.friendly(args)
    args.line = 'reload --discard'
    concordia_session.friendly(args)
    assert json.loads(args.draft.read_text())['session_id'] == 'a-new-session'


def test_shared_discover_call_and_watch_use_existing_transport(capsys):
  server = project_run_steps_test.editor()
  args = argparse.Namespace(
      url='http://fixture',
      timeout=1,
      line='discover',
      draft=None,
      file=None,
      output=None,
  )
  requests = []
  transport = fake_http(server)

  def request(req, **kwargs):
    requests.append(req)
    return transport(req, **kwargs)

  with mock.patch.object(
      concordia_session.urllib.request, 'urlopen', side_effect=request
  ):
    assert (
        concordia_session.friendly(args)
        == server.operation_service.discover('developer')['result']
    )
    args.line = "call project.reset '{}'"
    concordia_session.friendly(args)
    body = json.loads(requests[-1].data)
    assert body['operation'] == 'project.reset'
    assert body['retry_key'] and body['references']['session_id']
    assert isinstance(body['revision'], int)
    args.line = "call runtime.step '{}'"
    with pytest.raises(operation_service.OperationError, match='Cannot step'):
      concordia_session.friendly(args)
    args.line = "call nonexistent '{}'"
    with pytest.raises(operation_service.OperationError):
      concordia_session.friendly(args)
  capsys.readouterr()
  with mock.patch.object(
      concordia_session.urllib.request,
      'urlopen',
      return_value=io.BytesIO(b'data: {"revision":1}\n\n'),
  ) as transport:
    assert (
        concordia_session.main(
            ['--url', 'http://fixture', 'command', '--line', 'watch']
        )
        == 0
    )
    assert transport.call_count == 1
    assert transport.call_args.args[0].full_url.endswith('/api/events')
  assert json.loads(capsys.readouterr().out) == {'revision': 1}


def test_raw_call_keeps_service_revision_retry_and_audience_contracts():
  server = project_run_steps_test.editor()
  service = server.operation_service
  envelope = service.snapshot('developer')
  plan = session_commands.parse("call project.reset '{}'")
  request = {key: plan[key] for key in ('operation', 'arguments')}
  request.update(
      revision=envelope['revision'],
      references=envelope['references'],
      retry_key='fixture-reset',
  )
  first = service.dispatch('developer', request)
  assert service.dispatch('developer', request) == first
  with pytest.raises(operation_service.OperationError):
    service.dispatch('developer', dict(request, retry_key='new-stale-request'))
  with pytest.raises(operation_service.OperationError):
    service.dispatch('forbidden', request)
  assert service.discover('forbidden')['result']['operations'] == []


@pytest.mark.parametrize('arguments', [['--help'], ['--nonexistent']])
def test_log_parser_review_errors_never_exit_or_print(arguments, capsys):
  with pytest.raises(ValueError):
    concordia_log.analyze(
        concordia_log_test._create_sample_log(), 'overview', arguments
    )
  assert capsys.readouterr() == ('', '')

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

"""In-process editor contracts: no engine execution, model calls or listeners."""

import copy
import json
import threading
import time
from unittest import mock
import uuid

from concordia.agents import entity_agent_with_logging
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import project_test_support as template
from concordia.utils import simulation_server
import pytest


@pytest.fixture(params=('fixed', 'structural'))
def editor(request):
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
    structural = request.param == 'structural'
    registry = (
        template.builder_registry() if structural else template.registry()
    )
    key = 'builder-v1' if structural else template.TEMPLATE_KEY
    document = registry.default_document(key)
    if structural:
      player = copy.deepcopy(document['instances'][0])
      player['id'] = 'charlie'
      player['params']['name'] = 'Charlie'
      gm = copy.deepcopy(document['instances'][2])
      gm['id'] = 'second-gm'
      gm['params']['name'] = 'Second GM'
      gm['params']['next_game_master_name'] = 'second-gm'
      document['instances'].extend([player, gm])
    server = simulation_server.SimulationServer(port=0)
    server.configure_project(
        registry,
        document,
        mock.Mock(),
        integrated=True,
        preview=lambda config: template.build(config).make_checkpoint_data(),
    )
    yield server, server._project_editor
    server.step_controller.stop()
    if server._project_thread:
      server._project_thread.join(3)
      assert not server._project_thread.is_alive()


def request(editor, operation, arguments=None):
  return dict(
      operation=operation,
      arguments=arguments or {},
      revision=editor.service.revision,
      references=copy.deepcopy(editor.service.references),
      retry_key=str(uuid.uuid4()),
  )


def dispatch(editor, operation, arguments=None):
  return editor.service.dispatch(
      'developer', request(editor, operation, arguments)
  )


def eventually(predicate):
  deadline = time.monotonic() + 3
  while not predicate():
    if time.monotonic() >= deadline:
      raise AssertionError('Boundary was not reached')
    time.sleep(0.005)


def test_preview_roundtrip_invalid_and_stale_are_atomic(editor):
  server, adapter = editor
  snapshot = adapter.service.snapshot('developer')
  entities = snapshot['result']['definition']['entities']
  assert (
      'SelfPerception'
      in entities['entity_1']['component_info']['context_components']
  )
  assert server.simulation is None
  doc = server.get_project()['document']
  doc['instances'][0]['params'][
      'custom_instructions'
  ] = '</script>\n🎵 & literal'
  doc['instances'][1]['params']['goal'] = 'Try new music'
  save = request(
      adapter, 'project.save', {'text': json.dumps(doc), 'revision': 0}
  )
  result = adapter.service.dispatch('developer', save)
  assert adapter.service.dispatch('developer', save) == result
  assert server.get_project()['revision'] == 1
  assert server.get_project()['document'] == doc
  before = copy.deepcopy(adapter.snapshot())
  for text in ('{', json.dumps(dict(doc, max_steps=True))):
    with pytest.raises(operation_service.OperationError):
      dispatch(adapter, 'project.save', {'text': text, 'revision': 1})
    assert adapter.snapshot() == before
  with pytest.raises(operation_service.OperationError, match='another tab'):
    dispatch(adapter, 'project.save', {'text': json.dumps(doc), 'revision': 0})
  rebuilt = template.build(
      adapter.registry.to_config(server.get_project()['document'])
  )
  player = rebuilt.get_entities()[0]
  assert isinstance(player, entity_agent_with_logging.EntityAgentWithLogging)
  assert (
      player.get_component('Instructions').get_state()['state']
      == '</script>\n🎵 & literal'
  )


def test_pause_ack_edit_step_retry_and_reset_wait_for_runner(editor):
  server, adapter = editor
  bound = threading.Event()
  approach = threading.Event()
  exit_runner = threading.Event()
  permissions = []

  def runner(config):
    sim = template.build(config)
    server.set_simulation(sim)
    server.broadcast_entity_info(sim.make_checkpoint_data())
    bound.set()
    approach.wait(3)
    while server.step_controller.wait_for_step_permission():
      permissions.append(True)
    exit_runner.wait(3)

  server._project_runner = runner
  initial = server.get_project()['document']
  start = request(adapter, 'project.run', {'revision': 0})
  started = adapter.service.dispatch('developer', start)
  assert bound.wait(3)
  assert adapter.service.dispatch('developer', start) == started
  dispatch(adapter, 'runtime.pause')
  assert adapter.state() == 'pausing'
  edit = {
      'instance_id': 'alice',
      'component': 'Instructions',
      'value': 'Runtime only\n🎵',
  }
  with pytest.raises(
      operation_service.OperationError, match='acknowledged pause'
  ):
    dispatch(adapter, 'runtime.edit', edit)
  approach.set()
  eventually(lambda: adapter.state() == 'paused')
  dispatch(adapter, 'runtime.edit', edit)
  assert (
      server.simulation.get_entities()[0]
      .get_component('Instructions')
      .get_state()['state']
      == edit['value']
  )
  assert server.get_project()['document'] == initial
  if initial['schema_version'] == 2:
    dispatch(
        adapter,
        'runtime.edit',
        {**edit, 'instance_id': 'charlie', 'value': 'Third player only'},
    )
    players = {
        player.name: player for player in server.simulation.get_entities()
    }
    assert (
        players['Charlie'].get_component('Instructions').get_state()['state']
        == 'Third player only'
    )
    assert (
        players['Alice'].get_component('Instructions').get_state()['state']
        == edit['value']
    )
    assert server.get_project()['document'] == initial
  with mock.patch.object(
      server.simulation,
      'make_checkpoint_data',
      side_effect=RuntimeError('snapshot failed'),
  ):
    with pytest.raises(
        operation_service.OperationError, match='snapshot failed'
    ):
      dispatch(adapter, 'runtime.edit', {**edit, 'value': 'Must roll back'})
  assert (
      server.simulation.get_entities()[0]
      .get_component('Instructions')
      .get_state()['state']
      == edit['value']
  )
  for bad in (
      {**edit, 'component': '__memory__'},
      {**edit, 'value': 42},
      {**edit, 'instance_id': 'unknown'},
  ):
    with pytest.raises(operation_service.OperationError):
      dispatch(adapter, 'runtime.edit', bad)
  with pytest.raises(operation_service.OperationError, match='active'):
    dispatch(
        adapter, 'project.save', {'text': json.dumps(initial), 'revision': 0}
    )
  step = request(adapter, 'runtime.step')
  stepped = adapter.service.dispatch('developer', step)
  eventually(lambda: len(permissions) == 1 and adapter.state() == 'paused')
  assert adapter.service.dispatch('developer', step) == stepped
  assert len(permissions) == 1
  dispatch(adapter, 'project.reset')
  assert adapter.state() == 'stopping'
  with pytest.raises(operation_service.OperationError, match='active'):
    dispatch(adapter, 'project.run', {'revision': 0})
  exit_runner.set()
  server._project_thread.join(3)
  assert adapter.state() == 'ready'
  assert server.get_project()['document'] == initial
  with pytest.raises(operation_service.OperationError):
    dispatch(adapter, 'runtime.edit', edit)


def test_old_run_scope_and_failure_preserve_saved_definition(editor):
  server, adapter = editor
  old = request(adapter, 'runtime.pause')
  server._project_runner = mock.Mock(side_effect=RuntimeError('Build failed'))
  doc = server.get_project()['document']
  dispatch(adapter, 'project.run', {'revision': 0})
  server._project_thread.join(3)
  assert adapter.state() == 'failed'
  assert 'Build failed' in adapter.snapshot()['run']['message']
  assert server.get_project()['document'] == doc
  with pytest.raises(operation_service.OperationError, match='Attach'):
    adapter.service.dispatch('developer', old)
  dispatch(adapter, 'project.reset')
  assert adapter.state() == 'ready'


class _RequestSocket:
  """Exercise the real HTTP handler without binding a listener."""

  def __init__(self, request_bytes, fail_send=False):
    import io  # Local: only used by the in-memory transport fixture.

    self.input = io.BytesIO(request_bytes)
    self.output = bytearray()
    self.fail_send = fail_send

  def makefile(self, *_args):
    return self.input

  def sendall(self, data):
    if self.fail_send:
      raise BrokenPipeError('Disconnected before headers')
    self.output.extend(data)

  def settimeout(self, _timeout):
    pass


def http_request(server, path, body=None, origin=None, fail_send=False):
  payload = json.dumps(body).encode() if body is not None else b''
  method = 'POST' if body is not None else 'GET'
  headers = f'{method} {path} HTTP/1.0\r\nHost: localhost\r\n'
  if origin:
    headers += f'Origin: {origin}\r\n'
  if body is not None:
    headers += (
        f'Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n'
    )
  socket = _RequestSocket(headers.encode() + b'\r\n' + payload, fail_send)
  server._create_handler()(socket, ('127.0.0.1', 1234), object())
  return bytes(socket.output)


def test_capability_http_routes_origin_and_validation(editor):
  server, adapter = editor
  before = server.get_project()
  for path in ('/project', '/runtime', '/cmd/play', '/status'):
    assert b'404' in http_request(server, path).split(b'\r\n')[0]
  document = copy.deepcopy(before['document'])
  document['premise'] = 'A new initial premise'
  save = request(
      adapter, 'project.save', {'text': json.dumps(document), 'revision': 0}
  )
  rejected = http_request(
      server, '/api/dispatch', save, origin='https://untrusted.example'
  )
  assert b'403' in rejected.split(b'\r\n')[0]
  assert server.get_project() == before
  accepted = http_request(
      server, '/api/dispatch', save, origin='http://localhost'
  )
  assert b'200' in accepted.split(b'\r\n')[0]
  assert server.get_project()['document'] == document
  assert b'Access-Control-Allow-Origin: *' not in accepted
  assert (
      http_request(
          server, '/api/dispatch', save, origin='http://localhost'
      ).split(b'\r\n\r\n')[1]
      == accepted.split(b'\r\n\r\n')[1]
  )


def test_initial_sse_disconnect_removes_subscription(editor):
  server, adapter = editor
  http_request(server, '/api/events', fail_send=True)
  assert not adapter.service._clients

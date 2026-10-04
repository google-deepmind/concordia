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

"""Requested run length contracts; mocked runners and no network listeners."""

import json
import threading
from unittest import mock
from urllib.parse import urlsplit

from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import project_operations_test as operations_test
from concordia.utils import project_test_support as template
from concordia.utils import simulation_server
import pytest


@pytest.fixture(autouse=True)
def no_execution():
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('No simulation'),
      ),
      mock.patch.object(
          simulation_server.SimulationServer,
          'start',
          side_effect=AssertionError('No listener'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_text',
          side_effect=AssertionError('No model'),
      ),
      mock.patch.object(
          no_language_model.NoLanguageModel,
          'sample_choice',
          side_effect=AssertionError('No model'),
      ),
  ):
    yield


def editor(maximum=40, runner=None):
  registry = template.registry()
  document = registry.default_document(template.TEMPLATE_KEY)
  document['max_steps'] = maximum
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry, document, run_with_steps=runner or mock.Mock(), integrated=True
  )
  return server


def joined(server):
  assert server._project_thread is not None
  server._project_thread.join(3)
  assert not server._project_thread.is_alive()


@pytest.mark.parametrize(
    'maximum,requested,expected',
    [(40, None, 10), (4, None, 4), (40, 1, 1), (40, 40, 40)],
)
def test_run_length_preserves_authored_config(maximum, requested, expected):
  runner = mock.Mock()
  server = editor(maximum, runner)
  before = server.get_project()['document']
  server.run_project(0, requested)
  joined(server)
  config, count = runner.call_args.args
  assert count == expected
  assert config.default_max_steps == maximum
  result = server.get_project()
  assert result['document'] == before
  assert result['run']['requested_steps'] == expected
  assert result['run']['maximum_steps'] == maximum
  assert result['run_limits'] == {
      'maximum_steps': maximum,
      'default_requested_steps': min(10, maximum),
  }


@pytest.mark.parametrize('requested', [True, False, 0, -1, 41, 1.5, '10'])
def test_invalid_length_is_atomic(requested):
  runner = mock.Mock()
  server = editor(runner=runner)
  before = server.get_project()
  with pytest.raises(ValueError, match='Requested steps'):
    server.run_project(0, requested)
  assert server.get_project() == before
  assert server._project_thread is None
  runner.assert_not_called()


def test_operation_bounds_follow_saved_config_and_reject_stale_request():
  runner = mock.Mock()
  server = editor(runner=runner)
  adapter = server._project_editor
  assert adapter is not None
  document = server.get_project()['document']
  document['max_steps'] = 3
  server.replace_project(json.dumps(document), 0)
  assert server.get_project()['run_limits']['default_requested_steps'] == 3
  for args in (
      {'revision': 0, 'requested_steps': 2},
      {'revision': 1, 'requested_steps': 4},
      {'revision': 1, 'requested_steps': True},
      {'revision': 1},
  ):
    with pytest.raises(operation_service.OperationError):
      operations_test.dispatch(adapter, 'project.run', args)
  runner.assert_not_called()
  operations_test.dispatch(
      adapter, 'project.run', {'revision': 1, 'requested_steps': 2}
  )
  joined(server)
  assert runner.call_args.args[1] == 2
  assert server.get_project()['document']['max_steps'] == 3
  operations_test.dispatch(adapter, 'project.reset')
  operations_test.dispatch(
      adapter, 'project.run', {'revision': 1, 'requested_steps': 3}
  )
  joined(server)
  assert runner.call_args.args[1] == 3


def test_legacy_runner_does_not_advertise_or_accept_requested_length():
  server = simulation_server.SimulationServer()
  registry = template.registry()
  runner = mock.Mock()
  server.configure_project(
      registry, registry.default_document(template.TEMPLATE_KEY), runner
  )
  assert server.get_project()['run_limits'] is None
  with pytest.raises(ValueError, match='does not support'):
    server.run_project(0, 1)
  runner.assert_not_called()


@pytest.mark.parametrize('maximum', [40, 4])
def test_numeric_control_in_chromium_with_intercepted_requests(maximum):
  browser_api = pytest.importorskip('playwright.sync_api')
  release = threading.Event()
  received = []

  def runner(config, requested):
    received.append((config.default_max_steps, requested))
    release.wait(10)

  server = editor(maximum, runner)
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
      route.fulfill(
          json=service.dispatch('developer', route.request.post_data_json)
      )
    else:
      route.fulfill(status=404, body='Fixture: no network')

  try:
    with browser_api.sync_playwright() as playwright:
      browser = playwright.chromium.launch()
      with browser.new_context(
          viewport={'width': 412, 'height': 915}
      ) as context:
        context.add_init_script('window.EventSource=class {close(){}};')
        context.route('**/*', route_request)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://localhost/')
        field = page.get_by_label('Steps to run', exact=True)
        run = page.get_by_role('button', name='Run', exact=True)
        browser_api.expect(field).to_have_value(str(min(10, maximum)))
        browser_api.expect(field).to_have_attribute('max', str(maximum))
        for invalid in ('', '0', '-1', '1.5', str(maximum + 1)):
          field.fill(invalid)
          browser_api.expect(run).to_be_disabled()
        requested = min(7, maximum)
        field.fill(str(requested))
        browser_api.expect(run).to_be_enabled()
        run.click()
        browser_api.expect(field).to_be_disabled()
        assert received == [(maximum, requested)]
        page.reload()
        browser_api.expect(field).to_have_value(str(requested))
        browser_api.expect(field).to_be_disabled()
        assert server.get_project()['document']['max_steps'] == maximum
        release.set()
        joined(server)
        browser_api.expect(field).to_be_enabled()
        page.reload()
        browser_api.expect(field).to_have_value(str(min(10, maximum)))
        assert not errors
      browser.close()
  finally:
    release.set()
    if server._project_thread is not None:
      joined(server)

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

"""Optional real Chromium lifecycle checks with explicitly synthetic callbacks.

Install Playwright and Chromium, then run this module with pytest -n 0.
Uses the standard server, controller, renderer and bound minimal entities.
No Simulation.play, engine run loop or model calls are allowed.
"""

import json
from unittest import mock

from concordia.environment import step_controller
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal
from concordia.prefabs.simulation import generic
from concordia.typing import prefab
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import numpy as np
import pytest

browser_api = pytest.importorskip('playwright.sync_api')


@pytest.fixture(scope='module', name='browser')
def browser_fixture():
  with browser_api.sync_playwright() as playwright:
    chromium = playwright.chromium.launch()
    yield chromium
    chromium.close()


@pytest.fixture(name='server')
def server_fixture():
  config = prefab.Config(
      prefabs={'minimal': minimal.Entity()},
      instances=[
          prefab.InstanceConfig(
              prefab='minimal',
              role=prefab.Role.ENTITY,
              params={'name': 'Alice'},
          )
      ],
  )
  with (
      mock.patch.object(generic.Simulation, 'play', side_effect=AssertionError),
      mock.patch.object(
          sequential.Sequential, 'run_loop', side_effect=AssertionError
      ),
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
    simulation = generic.Simulation(
        config=config,
        model=no_language_model.NoLanguageModel(),
        embedder=lambda _: np.ones(3),
        engine=sequential.Sequential(),
    )
    checkpoint = simulation.make_checkpoint_data()
    app = simulation_server.SimulationServer(
        port=0,
        html_content=visual_interface.visualize_config_to_html(
            config,
            title='Lifecycle verification — synthetic callbacks',
            checkpoint_data=checkpoint,
        ),
    )
    app.set_simulation(simulation)
    app.broadcast_entity_info(checkpoint)
    app.start()
    try:
      yield app
    finally:
      app.stop()


def _assert_controls(page, state):
  browser_api.expect(page.locator('#run-status')).to_have_text(
      state, timeout=10000
  )
  for selector, enabled in [
      ('#btn-play', state == 'Paused'),
      ('#btn-pause', state == 'Running'),
      ('#btn-step', state == 'Paused'),
  ]:
    expect = browser_api.expect(page.locator(selector))
    if enabled:
      expect.to_be_enabled()
    else:
      expect.to_be_disabled()
  browser_api.expect(page.locator('#step-counter')).to_have_count(1)


def _mock_step(server, step):
  server.broadcast_step(
      step_controller.StepData(
          step=step,
          acting_entity='Alice',
          action='Synthetic callback',
          entity_actions={'Alice': 'Synthetic callback'},
          entity_logs={},
      )
  )


def test_multitab_completion_reload_and_reconnect(browser, server, tmp_path):
  context = browser.new_context(viewport={'width': 1440, 'height': 1000})
  first = context.new_page()
  second = context.new_page()
  errors = []
  for page in [first, second]:
    page.on('pageerror', lambda error: errors.append(str(error)))
  url = f'http://127.0.0.1:{server.bound_port}/'
  try:
    first.goto(url)
    second.goto(url)
    for page in [first, second]:
      _assert_controls(page, 'Paused')
    first.locator('#btn-play').click()
    for page in [first, second]:
      _assert_controls(page, 'Running')
    # A new document while playing must use authoritative initial status.
    second.reload()
    _assert_controls(second, 'Running')
    running_snapshot = second.request.get(url + 'status').json()
    second.locator('#btn-pause').click()
    for page in [first, second]:
      _assert_controls(page, 'Paused')
    with first.expect_response('**/cmd/step') as response:
      first.locator('#btn-step').click()
    assert response.value.json()['status'] == 'stepping'
    assert server.step_controller.wait_for_step_permission()
    for page in [first, second]:
      _assert_controls(page, 'Paused')
      browser_api.expect(page.locator('#step-counter')).to_have_text('0')
    browser_api.expect(first.locator('#console-output')).to_contain_text(
        'Single step requested'
    )
    assert 'Step executed' not in first.locator('#console-output').inner_text()
    _mock_step(server, 1)
    for page in [first, second]:
      browser_api.expect(page.locator('#step-counter')).to_have_text('1')
    first.locator('#btn-play').click()
    _assert_controls(second, 'Running')
    _mock_step(server, 2)
    server.broadcast_completion()
    for page in [first, second]:
      _assert_controls(page, 'Completed')
      browser_api.expect(page.locator('#step-counter')).to_have_text('2')
    first.screenshot(path=str(tmp_path / 'completed.png'))
    # A delayed HTTP snapshot must not resurrect an older running state.
    first.evaluate('status => applyControlStatus(status)', running_snapshot)
    _assert_controls(first, 'Completed')
    second.reload()
    _assert_controls(second, 'Completed')
    browser_api.expect(second.locator('#step-counter')).to_have_text('2')
    # Exercise a real failed EventSource connection and automatic reconnect.
    context.set_offline(True)
    first.evaluate('eventSource.close(); connectSSE();')
    _assert_controls(first, 'Disconnected')
    context.set_offline(False)
    _assert_controls(first, 'Completed')
    browser_api.expect(first.locator('#step-counter')).to_have_text('2')
    # Subsequent/replayed step events must not reference a deleted counter.
    _mock_step(server, 2)
    _assert_controls(first, 'Completed')
    browser_api.expect(first.locator('#step-counter')).to_have_text('2')
    first.screenshot(path=str(tmp_path / 'reconnected.png'))
    assert not errors
    (tmp_path / 'journey.json').write_text(
        json.dumps(
            {
                'tabs': 2,
                'play_pause_synchronized': True,
                'reload_and_reconnect_completed': True,
                'final_step': 2,
                'counter_nodes': first.locator('#step-counter').count(),
                'page_errors': errors,
                'synthetic_callbacks': 3,
                'simulation_steps': 0,
                'model_calls': 0,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
  finally:
    context.close()


def test_empty_controls_and_rejected_command(browser, server, tmp_path):
  server.set_simulation(None)
  server.set_html_content(
      visual_interface.visualize_config_to_html(
          prefab.Config(prefabs={}, instances=[]), title='Empty configuration'
      )
  )
  page = browser.new_page()
  errors = []
  page.on('pageerror', lambda error: errors.append(str(error)))
  try:
    page.goto(f'http://127.0.0.1:{server.bound_port}/')
    _assert_controls(page, 'No simulation')
    assert page.locator('.entity-card').count() == 0
    # Bypass disabled button to exercise the real rejection/error UI.
    page.evaluate('simStep()')
    browser_api.expect(page.locator('#console-output')).to_contain_text(
        'Cannot step: simulation is empty.'
    )
    assert 'Step executed' not in page.locator('#console-output').inner_text()
    browser_api.expect(page.locator('#step-counter')).to_have_text('0')
    assert not server.current_step_data
    assert server.step_controller.is_paused
    page.reload()
    _assert_controls(page, 'No simulation')
    assert not errors
    page.screenshot(path=str(tmp_path / 'empty.png'))
  finally:
    page.close()


def test_stopped_controls_survive_reload(browser, server):
  page = browser.new_page()
  try:
    url = f'http://127.0.0.1:{server.bound_port}/'
    page.goto(url)
    _assert_controls(page, 'Paused')
    response = page.request.post(url + 'stop').json()
    assert response['status'] == 'stopped'
    _assert_controls(page, 'Stopped')
    page.reload()
    _assert_controls(page, 'Stopped')
  finally:
    page.close()

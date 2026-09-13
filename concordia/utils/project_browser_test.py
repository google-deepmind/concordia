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

"""Optional real Chromium initial-project journeys; no simulation/model calls.

The Run callback is a build-only fixture, explicitly not execution evidence.
A separate documented example exercises standard Simulation.play.
"""

from contextlib import contextmanager
import json
import threading
from unittest import mock

from concordia.environment import step_controller
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import pytest

from examples.project_editor import run
from examples.project_editor import template

browser_api = pytest.importorskip('playwright.sync_api')


@pytest.fixture(scope='module', name='browser')
def chromium_browser():
  with browser_api.sync_playwright() as playwright:
    chromium = playwright.chromium.launch()
    yield chromium
    chromium.close()


@contextmanager
def editor(runner):
  registry = template.registry()
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      registry, registry.default_document(template.TEMPLATE_KEY), runner
  )
  server.start()
  http_server = server._server  # pylint: disable=protected-access
  assert http_server is not None
  try:
    yield server, f'http://127.0.0.1:{server.bound_port}/'
  finally:
    server.stop()
    http_server.server_close()


def upload(page, text):
  page.locator('#project-file').set_input_files({
      'name': 'project.json',
      'mimeType': 'application/json',
      'buffer': text.encode('utf-8'),
  })


@pytest.mark.parametrize(
    'literal',
    [
        '"quoted" & <tags> café 🎵\n\nlast line\n',
        '',
        '</textarea><script>window.projectProbe = true;</script>',
    ],
)
def test_save_reopen_actor_and_gm(browser, tmp_path, literal):
  built = []
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('No simulations in this fixture'),
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
      browser.new_context(accept_downloads=True) as context,
  ):
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    with editor(lambda config: built.append(run.build(config))) as (
        server,
        url,
    ):
      page.goto(url)
      page.locator('#project-alice-custom_instructions').fill(literal)
      page.locator('#project-premise').fill(literal)
      page.locator('#project-max-steps').fill('2')
      page.locator('[data-project-id="conversation"]').click()
      page.locator('#project-conversation-name').fill(
          'GM "音楽" & <conversation>'
      )
      page.locator('#project-conversation-acting_order').fill('random')
      page.locator('#project-conversation-can_terminate_simulation').check()
      assert page.get_by_role('button', name='Run saved project').is_disabled()
      with page.expect_download() as download:
        page.get_by_role('button', name='Save project', exact=True).click()
      path = tmp_path / 'saved.json'
      download.value.save_as(path)
      saved = json.loads(path.read_text())
      assert saved == server.get_project()['document']
      assert saved['instances'][0]['params']['custom_instructions'] == literal
      assert saved['instances'][2]['params']['can_terminate_simulation'] is True
      assert saved['max_steps'] == 2
      assert not built
      page.reload()
      page.locator('[data-project-id="alice"]').click()
      browser_api.expect(
          page.locator('#project-alice-custom_instructions')
      ).to_have_value(literal)
      assert page.evaluate('window.projectProbe') is None

    # A distinct server starts from the template, not the prior Python draft.
    with editor(lambda config: built.append(run.build(config))) as (fresh, url):
      page.goto(url)
      browser_api.expect(page.locator('#project-alice-name')).to_have_value(
          'Alice'
      )
      upload(page, path.read_text())
      browser_api.expect(
          page.locator('#project-alice-custom_instructions')
      ).to_have_value(literal)
      page.locator('[data-project-id="conversation"]').click()
      browser_api.expect(
          page.locator('#project-conversation-name')
      ).to_have_value('GM "音楽" & <conversation>')
      browser_api.expect(
          page.locator('#project-conversation-acting_order')
      ).to_have_value('random')
      browser_api.expect(
          page.locator('#project-conversation-can_terminate_simulation')
      ).to_be_checked()
      assert fresh.get_project()['document'] == saved
      before = fresh.get_project()
      for bad in [
          '{',
          json.dumps(dict(saved, template='unknown')),
          json.dumps(dict(saved, max_steps=True)),
      ]:
        upload(page, bad)
        browser_api.expect(page.locator('#project-error')).not_to_be_empty()
        assert fresh.get_project() == before
      invalid_ref = json.loads(json.dumps(saved))
      invalid_ref['instances'][2]['params']['next_game_master_name'] = 'missing'
      upload(page, json.dumps(invalid_ref))
      browser_api.expect(page.locator('#project-error')).to_contain_text(
          'expected target'
      )
      assert fresh.get_project() == before
      page.get_by_role('button', name='Run saved project').click()
      browser_api.expect(page.locator('#project-status')).to_contain_text(
          'Run: completed'
      )
      assert len(built) == 1
      assert (
          built[0]
          .get_entities()[0]
          .get_component('Instructions')
          .get_state()['state']
          == literal
      )
      assert built[0].get_game_masters()[0].name == 'GM "音楽" & <conversation>'
      assert fresh.get_project()['document'] == saved
      page.screenshot(path=str(tmp_path / 'reopened.png'), full_page=True)
      page.locator('[data-project-id="alice"]').click()
      page.locator('#project-alice-goal').fill(
          'A new initial draft, not runtime state'
      )
      with page.expect_download():
        page.get_by_role('button', name='Save project', exact=True).click()
      browser_api.expect(page.locator('#project-status')).to_contain_text(
          'earlier saved revision'
      )
      assert len(built) == 1
      assert (
          'Goal' not in built[0].get_entities()[0].get_all_context_components()
      )
      assert not errors


def test_two_tabs_reject_stale_and_active_draft(browser):
  entered = threading.Event()
  release = threading.Event()

  def fixture(config):
    del config
    entered.set()
    release.wait(10)

  with browser.new_context() as context, editor(fixture) as (server, url):
    a = context.new_page()
    b = context.new_page()
    a.goto(url)
    b.goto(url)
    a.locator('#project-alice-goal').fill('Saved in first tab')
    with a.expect_download():
      a.get_by_role('button', name='Save project', exact=True).click()
    b.locator('#project-alice-goal').fill('Stale second tab')
    b.get_by_role('button', name='Save project', exact=True).click()
    browser_api.expect(b.locator('#project-error')).to_contain_text(
        'another tab'
    )
    before = server.get_project()
    b.reload()
    browser_api.expect(b.locator('#project-alice-goal')).to_have_value(
        'Saved in first tab'
    )
    a.get_by_role('button', name='Run saved project').click()
    assert entered.wait(2)
    try:
      browser_api.expect(
          b.get_by_role('button', name='Save project', exact=True)
      ).to_be_disabled()
      browser_api.expect(b.locator('#project-alice-goal')).to_be_disabled()
      server.step_controller.pause()
      response = b.request.post(
          url + 'project',
          data={
              'text': json.dumps(before['document']),
              'revision': before['revision'],
          },
      )
      assert response.status == 400
      assert 'active' in response.json()['error']
      assert server.get_project()['document'] == before['document']
    finally:
      release.set()
    browser_api.expect(a.locator('#project-status')).to_contain_text(
        'Run: completed'
    )


def test_completed_project_can_start_a_fresh_controllable_run(
    browser, tmp_path
):
  built = []
  second_bound = threading.Event()
  finish = threading.Event()

  def fixture(config):
    simulation = run.build(config)
    built.append(simulation)
    server.set_simulation(simulation)
    checkpoint = simulation.make_checkpoint_data()
    server.set_runtime_html_content(
        visual_interface.visualize_config_to_html(
            config, checkpoint_data=checkpoint
        )
    )
    server.broadcast_entity_info(checkpoint)
    if len(built) == 1:
      server.broadcast_step(
          step_controller.StepData(4, 'Alice', 'Synthetic callback', {}, {})
      )
    else:
      second_bound.set()
      finish.wait(10)
    server.broadcast_completion()

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
      editor(fixture) as (server, url),
      browser.new_context(viewport={'width': 1100, 'height': 760}) as context,
  ):
    initial = context.new_page()
    runtime = context.new_page()
    errors = []
    for page in (initial, runtime):
      page.on('pageerror', lambda error: errors.append(str(error)))
    initial.goto(url)
    initial.get_by_role('button', name='Run saved project').click()
    browser_api.expect(initial.locator('#project-status')).to_contain_text(
        'Run: completed'
    )
    runtime.goto(url + 'runtime')
    browser_api.expect(runtime.locator('#run-status')).to_have_text('Completed')
    browser_api.expect(runtime.locator('#step-counter')).to_have_text('4')
    initial.get_by_role('button', name='Run saved project').click()
    try:
      assert second_bound.wait(2)
      runtime.reload()
      browser_api.expect(runtime.locator('#run-status')).to_have_text('Running')
      browser_api.expect(runtime.locator('#step-counter')).to_have_text('0')
      runtime.locator('#btn-pause').click()
      browser_api.expect(runtime.locator('#run-status')).to_have_text('Paused')
      browser_api.expect(runtime.locator('#btn-step')).to_be_enabled()
      runtime.screenshot(path=str(tmp_path / 'fresh-runtime.png'))
      assert built[0] is not built[1]
      assert not errors
    finally:
      finish.set()

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
      page.locator('[data-project-id="bob"]').click()
      page.locator('#project-bob-goal').fill(
          'Share music & listen carefully 🎵'
      )
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
      assert saved['instances'][1]['prefab'] == 'basic'
      assert saved['instances'][1]['params']['goal'] == (
          'Share music & listen carefully 🎵'
      )
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
      bob = built[0].get_entities()[1]
      assert {'SituationPerception', 'SelfPerception', 'PersonBySituation'} <= (
          bob.get_all_context_components().keys()
      )
      assert bob.get_component('Goal').get_state()['state'] == (
          'Share music & listen carefully 🎵'
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

    runtime.locator('[data-entity-id="entity_0"]').click()
    browser_api.expect(runtime.locator('#inspector-subtitle')).to_have_text(
        'minimal'
    )
    assert runtime.locator('#toggle_comp_SelfPerception').count() == 0
    runtime.locator('[data-entity-id="entity_1"]').click()
    browser_api.expect(runtime.locator('#inspector-title')).to_have_text('Bob')
    browser_api.expect(runtime.locator('#inspector-subtitle')).to_have_text(
        'basic'
    )
    for name in ('SelfPerception', 'SituationPerception', 'PersonBySituation'):
      browser_api.expect(
          runtime.locator('#toggle_comp_' + name)
      ).to_be_visible()
    runtime.screenshot(path=str(tmp_path / 'basic-roommate-inspector.png'))

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


@pytest.fixture
def integrated_editor():
  """Real HTTP/browser surface, with all simulation execution forbidden."""
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
    registry = template.registry()
    server = simulation_server.SimulationServer(port=0)
    server.configure_project(
        registry,
        registry.default_document(template.TEMPLATE_KEY),
        lambda _: None,
        integrated=True,
        preview=lambda config: run.build(config).make_checkpoint_data(),
    )
    server.start()
    try:
      yield server, f'http://127.0.0.1:{server.bound_port}/'
    finally:
      server.step_controller.stop()
      server.stop()


@pytest.mark.parametrize('width', [360, 412])
def test_integrated_portrait_definition_roundtrip(
    browser, integrated_editor, tmp_path, width
):
  server, url = integrated_editor
  with browser.new_context(
      viewport={'width': width, 'height': 800}, has_touch=True
  ) as context:
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    page.locator('[data-instance-id="alice"]').click()
    literal = (
        'A quiet playlist\n\n"quotes" & 🎵'
        ' </textarea><script>window.probe=1</script>'
    )
    page.locator('#editor-alice-custom_instructions').fill(literal)
    assert page.get_by_role('button', name='Run', exact=True).is_disabled()
    page.get_by_role('button', name='Save draft', exact=True).click()
    browser_api.expect(page.locator('#editor-status')).to_contain_text(
        'saved definition'
    )
    assert (
        server.get_project()['document']['instances'][0]['params'][
            'custom_instructions'
        ]
        == literal
    )
    page.get_by_role('button', name='Hierarchy', exact=True).click()
    page.locator('[data-instance-id="bob"]').click()
    page.locator('#editor-bob-goal').fill('Find music both roommates enjoy.')
    page.get_by_role('button', name='Save draft', exact=True).click()
    browser_api.expect(page.locator('#editor-status')).to_contain_text(
        'saved definition'
    )
    assert page.locator('#toggle_comp_SelfPerception').count() == 1
    with page.expect_download() as download:
      page.get_by_role('button', name='Export JSON').click()
    exported = tmp_path / 'project.json'
    download.value.save_as(exported)
    assert json.loads(exported.read_text()) == server.get_project()['document']
    before = server.get_project()
    page.locator('#editor-file').set_input_files(
        {'name': 'bad.json', 'mimeType': 'application/json', 'buffer': b'{'}
    )
    browser_api.expect(page.locator('#editor-error')).not_to_be_empty()
    assert server.get_project() == before
    page.reload()
    page.locator('#editor-file').set_input_files(str(exported))
    browser_api.expect(page.locator('#editor-status')).to_contain_text(
        'saved definition'
    )
    page.locator('[data-instance-id="alice"]').click()
    browser_api.expect(
        page.locator('#editor-alice-custom_instructions')
    ).to_have_value(literal)
    page.set_viewport_size({'width': width, 'height': 480})
    page.locator('#editor-alice-custom_instructions').focus()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert page.evaluate('window.probe') is None
    page.screenshot(path=str(tmp_path / f'editor-{width}.png'), full_page=True)
    assert not errors


def test_integrated_multitab_draft_and_reconnect(browser, integrated_editor):
  server, url = integrated_editor
  with browser.new_context(viewport={'width': 412, 'height': 915}) as context:
    a, b = context.new_page(), context.new_page()
    a.goto(url)
    b.goto(url)
    for page in (a, b):
      page.locator('[data-instance-id="alice"]').click()
    a.locator('#editor-alice-goal').fill('First tab')
    b.locator('#editor-alice-goal').fill('Keep this unsaved field')
    a.get_by_role('button', name='Save draft').click()
    browser_api.expect(b.locator('#editor-error')).to_contain_text(
        'another tab'
    )
    browser_api.expect(b.locator('#editor-alice-goal')).to_have_value(
        'Keep this unsaved field'
    )
    b.get_by_role('button', name='Save draft').click()
    browser_api.expect(b.locator('#editor-error')).to_contain_text(
        'another tab'
    )
    assert (
        server.get_project()['document']['instances'][0]['params']['goal']
        == 'First tab'
    )
    context.set_offline(True)
    browser_api.expect(b.locator('#editor-status')).to_contain_text(
        'Disconnected'
    )
    assert b.get_by_role('button', name='Save draft').is_disabled()
    context.set_offline(False)
    browser_api.expect(
        b.get_by_role('button', name='Save draft')
    ).to_be_enabled()
    browser_api.expect(b.locator('#editor-alice-goal')).to_have_value(
        'Keep this unsaved field'
    )


def test_integrated_controls_use_acknowledged_boundary(
    browser, integrated_editor
):
  """Build-only runner plus permission waits; deliberately no Sequential run."""
  server, url = integrated_editor
  approach = threading.Event()

  def runner(config):
    simulation = run.build(config)
    server.set_simulation(simulation)
    server.broadcast_entity_info(simulation.make_checkpoint_data())
    approach.wait(10)
    count = 0
    while count < 3 and server.step_controller.wait_for_step_permission():
      count += 1
      server.broadcast_step(
          step_controller.StepData(
              count, 'Alice', f'Fixture step {count}', {}, {}
          )
      )
    server.broadcast_completion()

  server._project_runner = runner
  with browser.new_context(
      viewport={'width': 360, 'height': 800}, has_touch=True
  ) as context:
    page = context.new_page()
    page.goto(url)
    page.get_by_role('button', name='Run', exact=True).click()
    try:
      browser_api.expect(
          page.get_by_role('button', name='Pause', exact=True)
      ).to_be_enabled()
      page.get_by_role('button', name='Pause', exact=True).click()
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'pausing'
      )
      assert page.get_by_role('button', name='Step', exact=True).is_disabled()
      approach.set()
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'paused'
      )
      page.get_by_role('button', name='Hierarchy', exact=True).click()
      page.locator('[data-instance-id="alice"]').click()
      page.locator('#toggle_comp_Instructions').click()
      page.locator('#dyn_Instructions_state').fill('Runtime-only text')
      page.locator('.dynamic-save-btn[data-component="Instructions"]').click()
      browser_api.expect(page.locator('#dyn_Instructions_state')).to_have_value(
          'Runtime-only text'
      )
      assert (
          server.get_project()['document']['instances'][0]['params'][
              'custom_instructions'
          ]
          != 'Runtime-only text'
      )
      page.get_by_role('button', name='Step', exact=True).click()
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'Step 1'
      )
      browser_api.expect(
          page.get_by_role('button', name='Resume', exact=True)
      ).to_be_enabled()
      page.get_by_role('button', name='Resume', exact=True).click()
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'completed'
      )
      page.get_by_role('button', name='Reset', exact=True).click()
      browser_api.expect(page.locator('#editor-status')).to_contain_text(
          'ready'
      )
    finally:
      approach.set()
      server.step_controller.stop()
      server._project_thread.join(3)

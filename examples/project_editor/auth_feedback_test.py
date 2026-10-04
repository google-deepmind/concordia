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

"""Provider failures through real operations/UI with a fake runner and SDK."""

import json
from unittest import mock
from urllib.parse import urlsplit

from concordia.contrib.language_models.together import together_ai_model
from concordia.contrib.language_models.together import together_ai_model_test
from concordia.document import interactive_document
from concordia.environment import step_controller
from concordia.prefabs.simulation import generic
from concordia.utils import project_run_steps_test
import pytest

from examples.project_editor import run


@pytest.mark.parametrize('previous_step', [0, 2])
def test_auth_failure_reaches_log_without_advancing_or_fabricating_action(
    previous_step, capsys, caplog, tmp_path
):
  browser_api = pytest.importorskip('playwright.sync_api')
  client = mock.Mock()
  client.chat.completions.create.side_effect = together_ai_model_test.sdk_error(
      401
  )
  original_build = run.build

  def fake_build(config, *, model=None, engine=None):
    preview = original_build(config)
    if model is None:
      return preview
    simulation = mock.Mock()
    simulation.make_checkpoint_data.return_value = (
        preview.make_checkpoint_data()
    )

    def fake_play(*, step_callback, **_):
      if previous_step:
        step_callback(
            step_controller.StepData(
                step=previous_step,
                acting_entity='Alice',
                action='Recorded fixture action',
                entity_actions={'Alice': 'Recorded fixture action'},
                entity_logs={},
            )
        )
      # Real document/model contract, but no engine or Simulation.play execution.
      document = interactive_document.InteractiveDocument(model)
      action = document.open_question(
          'DUMMY_SECRET prompt', answer_prefix='Bob: ', max_tokens=16
      )
      step_callback(
          step_controller.StepData(
              step=previous_step + 1,
              acting_entity='Bob',
              action=action,
              entity_actions={'Bob': action},
              entity_logs={},
          )
      )

    simulation.play.side_effect = fake_play
    return simulation

  with (
      mock.patch.dict(
          run.os.environ,
          {
              'TOGETHER_API_KEY': 'DUMMY_SECRET',
              'TOGETHER_AI_API_KEY': 'UNUSED_DUMMY',
          },
          clear=True,
      ),
      mock.patch.object(
          together_ai_model, '_create_together_client', return_value=client
      ),
      mock.patch.object(
          together_ai_model.time,
          'sleep',
          side_effect=AssertionError('No auth retry'),
      ),
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('No simulation'),
      ),
      mock.patch.object(
          run.simulation_server.SimulationServer,
          'start',
          side_effect=AssertionError('No server'),
      ),
      mock.patch.object(run, 'build', side_effect=fake_build),
      mock.patch.object(run, 'save_result') as save,
      browser_api.sync_playwright() as playwright,
  ):
    server = run.create_editor(
        port=0,
        step_delay=0,
        model_selection=run.ModelSelection(
            'together_ai', 'deepseek-ai/DeepSeek-V4.1-Flash'
        ),
    )
    service = server.operation_service
    assert service is not None

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
        route.fulfill(status=404, body='No external network')

    browser = playwright.chromium.launch()
    context = browser.new_context(viewport={'width': 412, 'height': 915})
    context.add_init_script('window.EventSource=class {close(){}};')
    context.route('**/*', route_request)
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://localhost/')
    button = page.get_by_role('button', name='Run', exact=True)
    browser_api.expect(button).to_be_enabled()
    page.locator('#editor-requested-steps').fill('3')
    button.click()
    output = page.get_by_role('log', name='Simulation log')
    browser_api.expect(output).to_contain_text(
        'Together HTTP 401: authentication failed'
    )
    browser_api.expect(output).to_contain_text(
        'Credential source: TOGETHER_API_KEY.'
    )
    browser_api.expect(output).to_contain_text(
        'Run accepted; waiting for runner output.'
    )
    browser_api.expect(page.locator('.layout')).to_have_attribute(
        'data-tab', 'log'
    )
    browser_api.expect(page.locator('#editor-status')).to_contain_text('failed')
    assert 'authentication failed' not in page.locator('.header').inner_text()
    assert (
        'authentication failed'
        not in page.locator('.center-panel').inner_text()
    )
    assert 'DUMMY_SECRET' not in page.content()
    assert 'project.run completed' not in output.inner_text()
    project_run_steps_test.joined(server)
    snapshot = service.snapshot('developer')['result']
    assert snapshot['run']['status'] == 'failed'
    assert snapshot['current_step'] == previous_step
    assert all(step['acting_entity'] != 'Bob' for step in snapshot['steps'])
    assert 'DUMMY_SECRET' not in json.dumps(snapshot)
    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    browser_api.expect(
        output.locator('.console-line.error').filter(
            has_text='authentication failed'
        )
    ).to_have_count(1)
    assert (
        output.evaluate(
            'node=>node.scrollHeight-node.scrollTop-node.clientHeight'
        )
        < 5
    )
    page.screenshot(path=str(tmp_path / 'auth-failure.png'))
    assert not errors
    browser.close()
    save.assert_not_called()
  client.chat.completions.create.assert_called_once()
  assert 'DUMMY_SECRET' not in caplog.text + ''.join(capsys.readouterr())

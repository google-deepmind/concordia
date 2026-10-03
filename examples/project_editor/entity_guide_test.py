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

"""The GUI guide edits real example prefabs through intercepted HTTP only."""

import json
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

from concordia.command_line_interface import concordia_session
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import session_commands_test
import pytest

from examples.project_editor import run


@pytest.fixture(autouse=True)
def no_execution():
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('No simulation'),
      ),
      mock.patch.object(
          run.simulation_server.SimulationServer,
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
      mock.patch.object(
          run.ModelSelection,
          'create_model',
          side_effect=AssertionError('No provider'),
      ),
  ):
    yield


def test_graphical_guide_matches_copied_cli_design(tmp_path):
  browser_api = pytest.importorskip('playwright.sync_api')
  server = run.create_editor(port=0, output=tmp_path / 'unused')
  service = server.operation_service
  assert service is not None

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

  with browser_api.sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    with browser.new_context(
        viewport={'width': 1400, 'height': 950}
    ) as context:
      context.route('**/*', intercept)
      context.add_init_script('window.EventSource=class {close(){}};')
      page = context.new_page()
      page.goto('http://localhost/')
      expect = browser_api.expect
      expect(page.locator('#editor-status')).to_contain_text('saved definition')

      def save():
        page.get_by_role('button', name='Save draft', exact=True).click()
        expect(page.locator('#editor-status')).to_contain_text(
            'saved definition'
        )

      page.locator('[data-instance-id="alice"]').click()
      assert page.locator('#comp_Goal').count() == 0
      page.get_by_label(
          'Goal · initial text (empty removes the optional component)',
          exact=True,
      ).fill('Find music everyone enjoys')
      save()
      expect(page.locator('#comp_Goal')).to_be_attached()
      page.locator('#editor-prototype').select_option('alice')
      page.get_by_role('button', name='Add instance', exact=True).click()
      name_input = page.get_by_label('Entity name', exact=True)
      field_id = name_input.get_attribute('id')
      assert field_id is not None
      charlie_id = field_id[len('editor-') : -len('-name')]
      name_input.fill('Charlie')
      page.get_by_label('Instructions · initial text', exact=True).fill(
          'Charlie listens carefully.'
      )
      page.locator('#component-type').select_option('constant')
      page.get_by_role('button', name='Add component', exact=True).click()
      page.get_by_label('Context text', exact=True).fill(
          'Listen before replying.'
      )
      page.get_by_label('Context label', exact=True).fill('Reminder')
      save()
      graphical = server.get_project()['document']
      assert 'groups' not in graphical
      assert (
          page.get_by_role('button', name='Add group', exact=True).count() == 0
      )
      # Versioned export/reopen preserves the component design.
      with page.expect_download() as download:
        page.get_by_role('button', name='Export JSON', exact=True).click()
      exported = tmp_path / 'design.json'
      download.value.save_as(exported)
      assert json.loads(exported.read_text()) == graphical
      with page.expect_file_chooser() as chooser:
        page.get_by_role('button', name='Open JSON', exact=True).click()
      chooser.value.set_files(exported)
      expect(page.get_by_role('log', name='Simulation log')).to_contain_text(
          'Loaded local draft'
      )
    browser.close()

  readme = (
      Path(__file__).parents[2] / 'concordia/command_line_interface/README.md'
  ).read_text()
  start = readme.index('```text\nset alice params.goal') + len('```text\n')
  lines = readme[start : readme.index('```', start)].strip().splitlines()
  other = run.create_editor(port=0, output=tmp_path / 'also-unused')
  with mock.patch(
      'builtins.input', side_effect=[*lines, 'exit']
  ), mock.patch.object(
      concordia_session.urllib.request,
      'urlopen',
      side_effect=session_commands_test.fake_http(other),
  ):
    assert (
        concordia_session.main([
            '--url',
            'http://fixture',
            'interactive',
            '--draft',
            str(tmp_path / 'cli.json'),
        ])
        == 0
    )
  typed = other.get_project()['document']
  graphical['instances'][-1]['id'] = 'charlie'
  graphical['components'][0]['instance'] = 'charlie'
  graphical['components'][0]['id'] = typed['components'][0]['id']
  assert graphical == typed
  # Copy the guide's generic dynamic field block as well.
  start = readme.index('```text\nstate-field alice Instructions') + len(
      '```text\n'
  )
  lines = readme[start : readme.index('```', start)].strip().splitlines()
  with mock.patch(
      'builtins.input', side_effect=[*lines, 'exit']
  ), mock.patch.object(
      concordia_session.urllib.request,
      'urlopen',
      side_effect=session_commands_test.fake_http(other),
  ):
    assert (
        concordia_session.main([
            '--url',
            'http://fixture',
            'interactive',
            '--draft',
            str(tmp_path / 'cli.json'),
        ])
        == 0
    )
  assert (
      other.get_project()['document']['dynamic_states']['alice'][
          'Instructions'
      ]['state']
      == 'Alice asks before choosing a song.'
  )

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

"""Real DOM component CRUD with in-process transport; no listener or simulation."""

import json
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service
from concordia.utils import project_test_support as fixtures
from concordia.utils import simulation_server
import pytest

browser_api = pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize('width', [360, 1280])
def test_component_crud_dom(width, tmp_path):
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('no simulation'),
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
      browser_api.sync_playwright() as playwright,
  ):
    registry = fixtures.scene_registry()
    server = simulation_server.SimulationServer(port=0)
    server.configure_project(
        registry,
        registry.default_document('scenes-v1'),
        mock.Mock(side_effect=AssertionError('no run')),
        integrated=True,
        preview=lambda c: fixtures.build(c).make_checkpoint_data(),
    )
    browser = playwright.chromium.launch()
    try:
      context = browser.new_context(
          viewport={'width': width, 'height': 900}, accept_downloads=True
      )
      page = context.new_page()
      errors = []
      page.on('pageerror', lambda error: errors.append(str(error)))
      # Production HTML/JS and operation service, with HTTP/SSE replaced.
      page.add_init_script('window.EventSource=class {close(){}};')

      def route(request):
        path = urlsplit(request.request.url).path
        if path == '/api/state':
          request.fulfill(json=server.operation_service.snapshot('developer'))
        elif path == '/api/dispatch':
          try:
            result = server.operation_service.dispatch(
                'developer', request.request.post_data_json
            )
          except operation_service.OperationError as error:
            request.fulfill(
                status=409,
                json={'error': {'code': error.code, 'message': str(error)}},
            )
          else:
            request.fulfill(json=result)
        elif path == '/':
          request.fulfill(content_type='text/html', body=server.html_content)
        else:
          request.abort()

      page.route('**/*', route)
      page.goto('http://localhost/')

      page.locator('[data-world-id="scenes:opening"]').click()
      page.locator('#world-rounds').fill('0')
      page.get_by_role('button', name='Save draft', exact=True).click()
      browser_api.expect(page.locator('#editor-error')).to_contain_text(
          'expected integer from 1 to 1000'
      )
      page.get_by_role('button', name='Hierarchy', exact=True).click()
      page.locator('[data-instance-id="bob"]').click()
      page.get_by_role('button', name='Show invalid field', exact=True).click()
      browser_api.expect(page.locator('#world-rounds')).to_be_focused()
      browser_api.expect(page.locator('#world-rounds')).to_have_value('0')
      assert server.get_project()['document']['scenes'][0]['num_rounds'] == 2
      page.locator('#world-rounds').fill('2')
      browser_api.expect(page.locator('#editor-error')).to_be_empty()

      def select_owner(owner):
        page.get_by_role('button', name='Hierarchy', exact=True).click()
        page.locator('[data-instance-id="' + owner + '"]').click()

      def save():
        page.get_by_role('button', name='Save draft', exact=True).click()
        browser_api.expect(page.locator('#editor-status')).to_contain_text(
            'saved definition'
        )

      select_owner('bob')
      page.locator('#component-type').select_option('constant')
      page.get_by_role('button', name='Add component', exact=True).click()
      literal = 'Literal </textarea><script>window.probe=1</script> 🎵\n\n'
      page.locator('#world-name').fill('Identity')
      page.locator('#component-param-state').fill(literal)
      page.locator('#component-param-pre_act_label').fill('Context')
      save()
      first_id = server.get_project()['document']['components'][0]['id']
      select_owner('bob')
      page.locator('#component-type').select_option('recent-observations')
      page.get_by_role('button', name='Add component', exact=True).click()
      page.locator('#component-param-history_length').fill('3')
      page.get_by_role('button', name='Move earlier', exact=True).click()
      save()
      records = server.get_project()['document']['components']
      assert [r['type'] for r in records] == ['recent-observations', 'constant']
      assert records[1]['params']['state'] == literal
      assert page.evaluate('window.probe') is None
      page.get_by_role('button', name='Hierarchy', exact=True).click()
      page.locator('[data-component-id="' + first_id + '"]').click()
      page.get_by_role('button', name='Duplicate', exact=True).click()
      page.get_by_role('button', name='Remove', exact=True).click()
      page.get_by_role('button', name='Undo', exact=True).click()
      browser_api.expect(page.locator('#world-name')).to_have_value(
          'Identity copy'
      )
      page.get_by_role('button', name='Redo', exact=True).click()
      save()
      page.reload()
      page.get_by_role('button', name='Hierarchy', exact=True).click()
      page.locator('[data-component-id="' + first_id + '"]').click()
      browser_api.expect(page.locator('#component-param-state')).to_have_value(
          literal
      )
      page.get_by_role('button', name='Owner: Bob', exact=True).click()
      browser_api.expect(page.locator('#inspector-title')).to_have_text('Bob')
      select_owner('alice')
      page.locator('[data-reference-source="groups:ensemble"]').click()
      browser_api.expect(page.locator('#world-name')).to_have_value('Roommates')
      select_owner('alice')
      page.locator('#reference-target').select_option('bob')
      page.get_by_role('button', name='Replace references', exact=True).click()
      browser_api.expect(
          page.locator('[aria-label="Used by"]')
      ).to_contain_text('No incoming authored references.')
      page.get_by_role('button', name='Undo', exact=True).click()
      browser_api.expect(
          page.locator('[data-reference-source="groups:ensemble"]')
      ).to_be_visible()
      page.get_by_role('button', name='Redo', exact=True).click()
      save()
      replaced = server.get_project()['document']
      assert replaced['groups'][0]['participants'] == ['bob']
      assert replaced['scenes'][0]['participants'] == ['bob']
      assert replaced['components'][1]['params']['state'] == literal
      page.reload()
      select_owner('alice')
      browser_api.expect(
          page.locator('[aria-label="Used by"]')
      ).to_contain_text('No incoming authored references.')
      with page.expect_download() as download:
        page.get_by_role('button', name='Export JSON', exact=True).click()
      exported = json.loads(Path(download.value.path()).read_text())
      assert exported == server.get_project()['document']
      page.screenshot(
          path=str(tmp_path / ('component-crud-' + str(width) + '.png')),
          full_page=True,
      )
      assert not errors
      assert server.simulation is None
      assert not server.is_serving
    finally:
      browser.close()

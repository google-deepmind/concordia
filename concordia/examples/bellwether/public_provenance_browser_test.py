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

"""Mobile saved setup context using existing service, CLI and offline views."""

import json
import subprocess
import sys

from concordia.examples.bellwether.recipe_browser_test import host_fixture  # pylint: disable=unused-import
import pytest

pw = pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_downloaded_setup_is_same_as_cli_and_readable_offline(
    host, tmp_path, width, height
):
  value, url = host('mutual-aid')
  value.server.start()
  value.world.emit('private_message', 'PRIVATE_PROVENANCE', ['Nell'])
  value.case.manifest['private_detail'] = 'PRIVATE_PROVENANCE'
  before = value.operations.snapshot('developer')
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      errors = []
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.goto(url)
      pw.expect(page.locator('#export-json')).to_be_enabled()
      page.locator('#action').fill('Unsent private draft')
      artifacts = {}
      for extension in ('json', 'html', 'svg'):
        with page.expect_download() as download:
          page.locator('#export-' + extension).click()
        path = tmp_path / ('account.' + extension)
        download.value.save_as(path)
        artifacts[extension] = path
        assert 'PRIVATE_PROVENANCE' not in path.read_text()
        assert 'Unsent private draft' not in path.read_text()
      data = json.loads(artifacts['json'].read_text())
      assert data['declared_setup']['fuel_consumed_before_play'] == 2
      assert data['declared_setup']['available_fuel_at_start'] == 6
      result = subprocess.run(
          [
              sys.executable,
              '-m',
              'concordia.command_line_interface.concordia_session',
              '--url',
              f'http://127.0.0.1:{value.server.bound_port}',
              'call',
          ],
          input=json.dumps({
              'operation': 'game.public_account',
              'arguments': {'format': 'json'},
          }),
          capture_output=True,
          text=True,
          check=True,
      )
      assert json.loads(result.stdout)['result']['content'] == (
          artifacts['json'].read_text()
      )
      assert value.operations.snapshot('developer') == before
      assert page.locator('#action').input_value() == 'Unsent private draft'
      page.reload()
      pw.expect(page.locator('#export-json')).to_be_enabled()
      assert page.locator('#action').input_value() == 'Unsent private draft'
      context = browser.new_context(
          viewport={'width': width, 'height': height}, offline=True
      )
      offline = context.new_page()
      offline.on('pageerror', lambda e: errors.append(str(e)))
      requests = []
      offline.on('request', lambda r: requests.append(r.url))
      offline.goto(artifacts['html'].as_uri())
      pw.expect(offline.locator('body')).to_contain_text('Recipe: mutual-aid')
      pw.expect(offline.locator('body')).to_contain_text(
          'Declared before play: 2 fuel used; 6 fuel initially available.'
      )
      pw.expect(offline.locator('body')).to_contain_text(
          'not a complete run configuration'
      )
      assert offline.evaluate('document.documentElement.scrollWidth') <= width
      offline.screenshot(path=str(tmp_path / 'setup-html.png'), full_page=True)
      offline.goto(artifacts['svg'].as_uri())
      pw.expect(offline.locator('#figure-description')).to_contain_text(
          'Recipe: mutual-aid'
      )
      pw.expect(offline.locator('#figure-description')).to_contain_text(
          'Declared before play: 2 fuel used; 6 fuel initially available.'
      )
      assert offline.evaluate('document.documentElement.scrollWidth') <= width
      offline.evaluate(
          '()=>new Promise(resolve=>requestAnimationFrame('
          '()=>requestAnimationFrame(resolve)))'
      )
      offline.locator('svg').screenshot(
          path=str(tmp_path / 'setup-svg.png'), timeout=5000
      )
      assert requests == [artifacts['html'].as_uri(), artifacts['svg'].as_uri()]
      assert value.operations.snapshot('developer') == before
      assert not errors
    finally:
      browser.close()

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

"""Unavailable dawn replies are visibly not quotations; component fixtures."""

import json

from concordia.examples.bellwether import focus_browser_test
from concordia.examples.bellwether import response_failure_test
import pytest

pwlib = pytest.importorskip('playwright.sync_api')
host_fixture = focus_browser_test.host_fixture


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_missing_dawn_reply_and_public_export_are_honest(host, width, height):
  game, url = host()
  response_failure_test.finish_with_missing_dawn_response(game.world)
  game.phase = 'completed'
  game.inbox.finish('Completed component fixture; no model or engine run.')
  before = game.world.get_state()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      errors = []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.goto(url)
      pwlib.expect(page.locator('#epilogue')).to_be_visible()
      missing = page.locator('#responses [data-response-status=unavailable]')
      pwlib.expect(missing).to_have_count(1)
      pwlib.expect(missing).to_have_text(
          'Response unavailable. No usable reply was recorded.'
      )
      assert (
          page.locator('#responses [data-response-status=available]').count()
          == 3
      )
      assert 'I cannot give a clear commitment' not in page.content()
      assert 'PRIVATE_DAWN_BAD_OUTPUT' not in page.content()
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
      page.reload()
      pwlib.expect(
          page.locator('#responses [data-response-status=unavailable]')
      ).to_have_count(1)
      # Existing scoped read-only export; no new export implementation.
      account = game.operations.dispatch(
          'player',
          {
              'operation': 'game.public_account',
              'arguments': {'format': 'json'},
          },
      )['result']['content']
      records = json.loads(account)['events']
      failures = [
          row for row in records if row['kind'] == 'response_unavailable'
      ]
      assert len(failures) == 1
      assert 'Nell' in failures[0]['text']
      assert 'PRIVATE_DAWN_BAD_OUTPUT' not in account
      assert 'I cannot give a clear commitment' not in account
      assert game.world.get_state() == before
      assert not errors
    finally:
      browser.close()

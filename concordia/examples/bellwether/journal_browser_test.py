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

"""Search received entries through native controls and real scoped SSE."""

from concordia.examples.bellwether import focus_browser_test as helpers
from concordia.examples.bellwether import submission_browser_test as inputs
import pytest

pwlib = pytest.importorskip('playwright.sync_api')
host_fixture = helpers.host_fixture
no_resident_calls = inputs.no_resident_model_calls


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_search_filters_live_entries_without_sending_or_losing_draft(
    host, tmp_path, width, height
):
  game, url = host()
  literal = 'Café <script>window.journalProbe=true</script> & shelter'
  game.world.emit('speech', literal)
  game.world.emit('private_message', 'Café PRIVATE_NOTE', ['Coordinator'])
  game.world.emit('private_message', 'UNRECEIVED_NOTE', ['Nell'])
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      helpers.track_snapshots(page)
      requests, errors = [], []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.goto(url)
      pwlib.expect(page.locator('#journal')).to_contain_text(literal)
      # Subsequent search interactions must not make even a read request.
      page.on('request', lambda request: requests.append(request.url))
      before = game.world.get_state()
      page.fill('#action', 'My unsent draft 🌊')
      search = page.locator('#journal-search')
      search.fill('CAFE\u0301')
      visible = page.locator('#journal .story:visible')
      pwlib.expect(visible).to_have_count(2)
      assert 'UNRECEIVED_NOTE' not in page.content()
      page.select_option('#journal-audience', 'public')
      pwlib.expect(visible).to_have_count(1)
      assert visible.inner_text().endswith(literal)
      assert page.evaluate('window.journalProbe') is None
      page.select_option('#journal-watch', '1')
      pwlib.expect(visible).to_have_count(0)
      pwlib.expect(page.locator('#journal-count')).to_contain_text('No matches')
      page.select_option('#journal-watch', '0')
      search.focus()
      page.evaluate(
          "window.originalEntry=document.querySelector('#journal"
          " .story:not([hidden])')"
      )
      helpers.publish(game, page)
      pwlib.expect(search).to_be_focused()
      assert search.input_value() == 'CAFE\u0301'
      assert game.world.get_state() == before
      game.world.emit('speech', 'A new café account, without a simulated turn.')
      helpers.publish(game, page)
      pwlib.expect(visible).to_have_count(2)
      assert page.evaluate('window.originalEntry.isConnected')
      assert page.locator('#action').input_value() == 'My unsent draft 🌊'
      pwlib.expect(search).to_be_focused()
      page.select_option('#journal-audience', 'private')
      pwlib.expect(visible).to_have_count(1)
      assert 'PRIVATE_NOTE' in visible.inner_text()
      page.screenshot(path=str(tmp_path / 'filtered-journal.png'))
      page.get_by_role('button', name='Clear filters').click()
      pwlib.expect(search).to_have_value('')
      pwlib.expect(search).to_be_focused()
      assert page.locator('#journal .story[hidden]').count() == 0
      assert not requests
      assert not errors
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    finally:
      browser.close()


def test_role_revocation_clears_search_and_private_entries(host):
  game, url = host(shared=True)
  game.world.emit('private_message', 'PRIVATE_SEARCH_SENTINEL', ['Coordinator'])
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      helpers.track_snapshots(page)
      page.goto(url)
      principal = inputs.approve(game, page, 'Coordinator')
      page.fill('#journal-search', 'PRIVATE_SEARCH_SENTINEL')
      page.select_option('#journal-audience', 'private')
      pwlib.expect(page.locator('#journal .story:visible')).to_have_count(1)
      game.sessions.revoke(principal)
      helpers.publish(game, page)
      pwlib.expect(page.locator('#join')).to_be_visible()
      assert 'PRIVATE_SEARCH_SENTINEL' not in page.content()
      assert page.locator('#journal-search').input_value() == ''
      assert page.locator('#journal-audience').input_value() == 'all'
      assert page.locator('#journal-count').inner_text() == ''
      inputs.approve(game, page, 'spectator')
      pwlib.expect(page.locator('#journal')).to_contain_text(
          'Rain blows sideways'
      )
      page.fill('#journal-search', 'PRIVATE_SEARCH_SENTINEL')
      pwlib.expect(page.locator('#journal-count')).to_contain_text('No matches')
      assert page.locator('#journal .private').count() == 0
    finally:
      browser.close()

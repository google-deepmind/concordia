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

"""Requested display names must never change role ownership or records."""

from concordia.examples.bellwether.focus_browser_test import host_fixture  # pylint: disable=unused-import
from concordia.examples.bellwether.focus_browser_test import publish
from concordia.examples.bellwether.focus_browser_test import track_snapshots
from concordia.examples.bellwether.visual_note_browser_test import pending
import pytest

pw = pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_person_watch_and_literal_records_remain_distinct(
    host, tmp_path, width, height
):
  value, url = host()
  results, thread = pending(value.inbox)
  literal = 'Coordinator said “Dusk” in the recorded message.'
  value.world.emit('speech', literal)
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': width, 'height': height})
      track_snapshots(page)
      errors, posts = [], []
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.on(
          'request',
          lambda r: posts.append(r.url) if r.method == 'POST' else None,
      )
      page.goto(url)
      pw.expect(page.locator('#act')).to_be_enabled()
      before = value.operations.snapshot('developer')
      pw.expect(page.locator('#role-badge')).to_contain_text(
          'Coordinator Alice'
      )
      pw.expect(page.locator('#watch')).to_have_text(
          'Watch 1 · Early evening · 4 choices remain'
      )
      assert value.world.view()['watch'] == 'Dusk'
      assert value.world.view()['next_actor'] == 'Coordinator'
      pw.expect(page.locator('#journal')).to_contain_text(literal)
      assert page.locator('#context').text_content() == (
          value.inbox.snapshot()['pending']['context']
      )
      page.fill('#action', 'Keep my Coordinator draft at Dusk')
      assert any(
          key.endswith(':Coordinator')
          for key in page.evaluate('Object.keys(sessionStorage)')
      )
      publish(value, page)
      page.reload()
      pw.expect(page.locator('#act')).to_be_enabled()
      assert (
          page.locator('#action').input_value()
          == 'Keep my Coordinator draft at Dusk'
      )
      pw.expect(page.locator('#role-badge')).to_contain_text(
          'Coordinator Alice'
      )
      assert value.world.view()['watch'] == 'Dusk'
      assert not results and not posts and not errors
      # The test publish increments only service revision, not world ownership.
      assert (
          value.operations.snapshot('developer')['result']['night']
          == before['result']['night']
      )
      page.screenshot(
          path=str(tmp_path / 'clear-player-watch.png'), full_page=True
      )
      assert page.evaluate('document.documentElement.scrollWidth') <= width
      for watch in ('High Tide', 'Before Dawn'):
        value.world.data['watch'] = {'High Tide': 1, 'Before Dawn': 2}[watch]
        publish(value, page)
        pw.expect(page.locator('#watch')).to_have_text(
            watch + ' · 4 choices remain'
        )
    finally:
      value.inbox.finish('Test fixture closed')
      thread.join(2)
      browser.close()


def test_join_display_label_preserves_authenticated_roles(host):
  value, url = host(shared=True)
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    try:
      for role, label in [
          ('Coordinator', 'Coordinator Alice'),
          ('Nell', 'Nell'),
          ('spectator', 'Spectator'),
      ]:
        context = browser.new_context(viewport={'width': 360, 'height': 800})
        page = context.new_page()
        track_snapshots(page)
        page.goto(url)
        pw.expect(
            page.locator('#join-role option[value=Coordinator]')
        ).to_have_text('Coordinator Alice')
        page.fill('#join-name', 'Role fixture ' + role)
        page.select_option('#join-role', role)
        page.click('#join-submit')
        pw.expect(page.locator('#join-status')).to_contain_text(
            'Waiting for host approval'
        )
        request = next(
            x
            for x in value.sessions.pending()
            if x['label'] == 'Role fixture ' + role
        )
        assert request['requested_role'] == role
        value.sessions.approve(request['id'])
        publish(value, page)
        pw.expect(page.locator('#join')).to_be_hidden()
        pw.expect(page.locator('#role-badge')).to_contain_text(label)
        assert (
            value.operations.snapshot(request['id'])['result']['role'] == role
        )
        if role != 'Coordinator':
          assert (
              'Coordinator Alice'
              not in page.locator('#role-badge').text_content()
          )
        page.reload()
        pw.expect(page.locator('#role-badge')).to_contain_text(label)
        assert (
            value.operations.snapshot(request['id'])['result']['role'] == role
        )
      assert value.world.view()['watch'] == 'Dusk'
      assert value.world.view()['next_actor'] == 'Coordinator'
    finally:
      browser.close()

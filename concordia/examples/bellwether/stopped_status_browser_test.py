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

"""Stopped lifecycle states through standard role-scoped browser fixtures."""

import json
from pathlib import Path

from concordia.examples.bellwether import focus_browser_test as browser_helpers
from concordia.examples.bellwether import submission_browser_test as inputs
import pytest

pwlib = pytest.importorskip('playwright.sync_api')
host_fixture = browser_helpers.host_fixture
no_resident_calls = inputs.no_resident_model_calls


@pytest.mark.parametrize(
    'role,width,height',
    [
        ('solo', 360, 800),
        ('Coordinator', 800, 360),
        ('Nell', 360, 800),
        ('spectator', 360, 800),
    ],
)
def test_failure_is_interrupted_for_each_role_and_remains_readable(
    host, role, width, height, tmp_path
):
  game, url = host(shared=role != 'solo')
  game.phase = 'running'
  game.world.emit('private_message', 'PRIVATE_ONLY_NELL', ['Nell'])
  reader = None
  if role != 'spectator':
    if role == 'Nell':
      game.world.resolve('Coordinator', 'message Nell: Can we discuss?')
    _, reader = inputs.pending(
        game.nell_inbox if role == 'Nell' else game.inbox,
        'Nell' if role == 'Nell' else 'Coordinator',
        'synthetic-pending-input',
    )
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': width, 'height': height})
      browser_helpers.track_snapshots(page)
      errors, mutations = [], []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.goto(url)
      if role != 'solo':
        inputs.approve(game, page, role)
      if role != 'spectator':
        page.fill('#action', 'Keep this unsent draft \nwith exact words 🌊')
      # Invoke the existing failure finalizer; no engine/provider is called.
      game.phase = 'failed'
      game._failure = 'FixtureRuntimeFailure'  # pylint: disable=protected-access
      game._finish()  # pylint: disable=protected-access
      browser_helpers.publish(game, page)
      if reader:
        reader.join(timeout=2)
        assert not reader.is_alive()
      before = game.world.get_state()
      page.on(
          'request',
          lambda request: mutations.append(request.post_data_json)
          if request.method == 'POST'
          else None,
      )
      pwlib.expect(page.locator('#connection')).to_contain_text('Night stopped')
      pwlib.expect(page.locator('#run-status')).to_contain_text(
          'The run stopped unexpectedly.'
      )
      pwlib.expect(page.locator('#run-status')).to_contain_text(
          'Reloading does not restart the run.'
      )
      pwlib.expect(page.locator('#begin')).to_be_hidden()
      pwlib.expect(page.locator('#act')).to_be_disabled()
      pwlib.expect(page.locator('#epilogue')).to_be_hidden()
      assert page.locator('#role-wait').inner_text() == ''
      assert 'FixtureRuntimeFailure' not in page.content()
      assert ('PRIVATE_ONLY_NELL' in page.content()) == (role == 'Nell')
      for button in page.locator('#map button').all():
        button.click()
      assert not mutations
      with page.expect_download() as downloaded:
        page.click('#export-json')
      document = json.loads(
          Path(downloaded.value.path()).read_text(encoding='utf-8')
      )
      assert document['status'] == 'interrupted'
      assert 'PRIVATE_ONLY_NELL' not in json.dumps(document)
      page.reload()
      pwlib.expect(page.locator('#connection')).to_contain_text('Night stopped')
      if role != 'spectator':
        assert page.locator('#action').input_value() == (
            'Keep this unsent draft \nwith exact words 🌊'
        )
      page.route('**/api/events', lambda route: route.abort('failed'))
      page.reload()
      pwlib.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      page.unroute('**/api/events')
      pwlib.expect(page.locator('#connection')).to_contain_text(
          'Night stopped', timeout=15000
      )
      page.locator('#connection').scroll_into_view_if_needed()
      page.screenshot(path=str(tmp_path / 'stopped-player.png'))
      (tmp_path / 'browser-evidence.json').write_text(
          json.dumps(
              {
                  'role': role,
                  'viewport': [width, height],
                  'connection': page.locator('#connection').inner_text(),
                  'export_status': document['status'],
                  'mutations': mutations,
                  'page_errors': errors,
              },
              indent=2,
          ),
          encoding='utf-8',
      )
      assert not errors
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
      assert game.world.get_state() == before
      assert [row['operation'] for row in mutations] == ['game.public_account']
    finally:
      browser.close()
      if reader:
        game.inbox.finish('Test closed')
        if hasattr(game, 'nell_inbox'):
          game.nell_inbox.finish('Test closed')
        reader.join(timeout=2)


def test_success_and_closed_input_are_not_confused(host):
  game, url = host()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      browser_helpers.track_snapshots(page)
      page.goto(url)
      pwlib.expect(page.locator('#begin')).to_be_visible()
      game.inbox.finish('Transport fixture closed, not completed')
      browser_helpers.publish(game, page)
      pwlib.expect(page.locator('#connection')).to_contain_text('Input closed')
      pwlib.expect(page.locator('#begin')).to_be_hidden()
      assert 'The account is complete' not in page.locator('body').inner_text()
      game.phase = 'completed'
      browser_helpers.publish(game, page)
      pwlib.expect(page.locator('#connection')).to_contain_text(
          'The account is complete'
      )
      pwlib.expect(page.locator('#run-status')).to_be_hidden()
    finally:
      browser.close()

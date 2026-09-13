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

"""Three real browser contexts with synthetic prompts; zero simulation steps."""

import json
import threading

from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether.multiplayer_test import envelope
from concordia.examples.bellwether.multiplayer_test import hosted_fixture  # pylint: disable=unused-import
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
from concordia.utils import visual_interface
import pytest

pw = pytest.importorskip('playwright.sync_api')


def join(page, url, name, role):
  page.goto(url)
  pw.expect(page.locator('#join')).to_be_visible()
  page.fill('#join-name', name)
  page.select_option('#join-role', role)
  page.click('#join-submit')
  pw.expect(page.locator('#join-status')).to_contain_text(
      'Waiting for host approval'
  )


def approve_in_editor(host, name, revision):
  pw.expect(host.locator('#op-status')).to_contain_text(f'revision {revision}')
  if not host.locator('#op-snapshot').evaluate('e=>e.open'):
    host.locator('#op-snapshot summary').click()
  host.select_option('#op-preview-field', 'join_requests')
  host.click('#op-preview-refresh')
  rows = json.loads(host.locator('#op-state').inner_text())
  row = next(r for r in rows if r['label'] == name)
  host.select_option('#op-name', 'session.approve')
  host.locator('[data-key=request_id]').fill(row['id'])
  host.click('#op-submit')
  pw.expect(host.locator('#op-error')).to_have_text('')
  return row['id']


def test_two_players_host_approval_private_turn_reconnect_and_android(
    hosted, tmp_path
):
  game, _, url = hosted
  game.server.set_html_content(
      visual_interface.visualize_operations_to_html(game.config)
  )
  game.server.start()
  results = []
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    cctx = browser.new_context(
        viewport={'width': 360, 'height': 800}, is_mobile=True, has_touch=True
    )
    nctx = browser.new_context(
        viewport={'width': 412, 'height': 820}, is_mobile=True, has_touch=True
    )
    sctx = browser.new_context(
        viewport={'width': 800, 'height': 360}, has_touch=True
    )
    host = browser.new_page()
    coordinator, nell, spectator = (
        cctx.new_page(),
        nctx.new_page(),
        sctx.new_page(),
    )
    pages = [coordinator, nell, spectator, host]
    errors = []
    for page in pages:
      page.on('pageerror', lambda e: errors.append(str(e)))
    host.goto(f'http://127.0.0.1:{game.server.bound_port}')
    for page, name, role in [
        (coordinator, 'A', 'Coordinator'),
        (nell, 'B', 'Nell'),
        (spectator, 'C', 'spectator'),
    ]:
      join(page, url, name, role)
      assert 'PRIVATE_' not in page.content()
      approve_in_editor(
          host, name, game.operations.snapshot('developer')['revision']
      )
      pw.expect(page.locator('#role-badge')).to_contain_text(
          'Spectator' if role == 'spectator' else role
      )
    for page in [coordinator, nell, spectator]:
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    with game.operations.lock:
      game.world.emit('speech', 'PUBLIC_FOR_ALL')
      game.world.emit('speech', 'ONLY_COORDINATOR', ['Coordinator'])
      game.world.emit('speech', 'ONLY_NELL', ['Nell'])
      game.world.emit('speech', 'ONLY_MARA', ['Mara'])
      game.operations.publish({'kind': 'synthetic.deliver'})
    pw.expect(coordinator.locator('#journal')).to_contain_text(
        'ONLY_COORDINATOR'
    )
    pw.expect(nell.locator('#journal')).to_contain_text('ONLY_NELL')
    pw.expect(spectator.locator('#journal')).to_contain_text('PUBLIC_FOR_ALL')
    assert 'ONLY_NELL' not in coordinator.content()
    assert 'ONLY_COORDINATOR' not in nell.content()
    assert all(
        x not in spectator.content()
        for x in ['ONLY_NELL', 'ONLY_MARA', 'ONLY_COORDINATOR', 'PRIVATE_']
    )
    pw.expect(spectator.locator('#composer')).to_be_hidden()
    game.phase = 'running'
    game.world.data['agenda'] = [{
        'name': 'Nell',
        'purpose': 'request: release reserve ' + 'unbroken' * 20,
        'audience': rules.NAMES,
        'watch': 0,
    }]
    request = human_input.HumanInputRequest(
        request_id='synthetic-nell',
        entity_name='Nell',
        action_spec=entity_lib.free_action_spec(call_to_action='Your decision'),
        contexts={},
        context='OWN_CONTEXT_NELL',
    )
    thread = threading.Thread(
        target=lambda: results.append(game.nell_inbox(request))
    )
    thread.start()
    try:
      pw.expect(nell.locator('#act')).to_be_enabled()
      assert nell.evaluate('document.documentElement.scrollWidth') <= 412
      pw.expect(coordinator.locator('#act')).to_be_disabled()
      assert 'OWN_CONTEXT_NELL' not in coordinator.content()
      nell.select_option('#decision', 'decline')
      words = 'No — "reserve" is for members. 🌊 <img src=x onerror=alert(1)>'
      nell.fill('#action', words)
      nell.reload()
      pw.expect(nell.locator('#act')).to_be_enabled()
      assert nell.locator('#action').input_value() == words
      nell.select_option('#decision', 'decline')
      nell.route('**/api/events', lambda route: route.abort('failed'))
      nell.reload()
      pw.expect(nell.locator('#connection')).to_contain_text('Reconnecting')
      nell.unroute('**/api/events')
      pw.expect(nell.locator('#act')).to_be_enabled(timeout=15000)
      assert nell.locator('#action').input_value() == words
      nell.select_option('#decision', 'decline')
      nell.locator('#act').scroll_into_view_if_needed()
      assert nell.locator('#act').evaluate(
          'el => {const r=el.getBoundingClientRect();return el.contains('
          'document.elementFromPoint(r.x+r.width/2,r.y+r.height/2))}'
      )
      nell.set_viewport_size({'width': 800, 'height': 360})
      nell.click('#jump-action')
      assert nell.evaluate('document.documentElement.scrollWidth <= innerWidth')
      with nell.expect_request('**/api/dispatch') as sent:
        nell.click('#act')
      thread.join(2)
      assert results == [
          json.dumps(
              {'decision': 'decline', 'speech': words},
              separators=(',', ':'),
              ensure_ascii=False,
          )
      ]
      body = sent.value.post_data_json
      retried = nell.evaluate(
          """async body => {
            const response = await fetch('api/dispatch', {
              method:'POST', headers:{'Content-Type':'application/json'},
              body:JSON.stringify(body)
            });
            return await response.json();
          }""",
          body,
      )
      assert retried['result']['accepted'] is True
      assert len(results) == 1
      assert nell.locator('img').count() == 0
      # Revocation invalidates an already subscribed browser, not just new tabs.
      row = next(r for r in game.sessions.pending() if r['role'] == 'Nell')
      game.operations.dispatch(
          'developer',
          envelope(game, 'session.revoke', {'request_id': row['id']}),
      )
      pw.expect(nell.locator('#join')).to_be_visible()
      assert 'OWN_CONTEXT_NELL' not in nell.content()
      assert 'ONLY_NELL' not in nell.content()
      coordinator.screenshot(
          path=str(tmp_path / 'android-coordinator.png'), full_page=True
      )
      spectator.screenshot(
          path=str(tmp_path / 'spectator-landscape.png'), full_page=True
      )
      assert not errors
    finally:
      game.nell_inbox.finish('done')
      thread.join(2)
      browser.close()

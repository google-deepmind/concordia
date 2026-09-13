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

"""Native Chromium offline hints; real scoped SSE, no simulated turns."""

import json

from concordia.examples.bellwether import focus_browser_test as browser_helpers
from concordia.examples.bellwether import submission_browser_test as inputs
import pytest

pwlib = pytest.importorskip('playwright.sync_api')
host_fixture = browser_helpers.host_fixture
no_resident_calls = inputs.no_resident_model_calls


def track_streams(page):
  """Keep native EventSource; record frames for explicit late-message checks."""
  page.add_init_script("""
    window.streams=[];window.snapshots=[];window.snapshotCount=0;
    const Original=window.EventSource;
    window.EventSource=class extends Original {
      constructor(...args){super(...args);window.streams.push(this);
        this.addEventListener('message',e=>{
          window.snapshots.push(e.data);window.snapshotCount++;});}
    };
  """)


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_native_offline_disables_send_then_refreshes_once_without_resending(
    host, tmp_path, width, height
):
  game, url = host()
  game.phase = 'running'
  _, reader = inputs.pending(game.inbox, 'Coordinator', 'still-pending')
  before = game.world.get_state()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      track_streams(page)
      posts, errors = [], []
      page.on(
          'request',
          lambda r: posts.append(r.url) if r.method == 'POST' else None,
      )
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.fill('#action', 'My unsent private draft 🌊')
      page.context.set_offline(True)
      page.wait_for_function('navigator.onLine === false')
      pwlib.expect(page.locator('#act')).to_be_disabled()
      pwlib.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      assert page.evaluate('window.streams[0].readyState') == 2
      page.fill('#action', 'Edited while offline 🌊')
      page.locator('#map button').first.click()
      assert 'Rain blows sideways' in page.locator('#journal').inner_text()
      # Current server state differs when the browser resumes. No turn is run.
      game.inbox.finish('Input closed while offline')
      game.phase = 'failed'
      game.operations.publish({'kind': 'offline.fixture'})
      page.context.set_offline(False)
      pwlib.expect(page.locator('#connection')).to_contain_text('Night stopped')
      assert page.evaluate('window.streams.length') == 2
      assert page.locator('#action').input_value() == 'Edited while offline 🌊'
      pwlib.expect(page.locator('#act')).to_be_disabled()
      # Synthetic late callback on the closed old stream must not undo failure.
      page.evaluate("""()=>window.streams[0].dispatchEvent(
        new MessageEvent('message',{data:window.snapshots[0]}))""")
      pwlib.expect(page.locator('#connection')).to_contain_text('Night stopped')
      page.screenshot(path=str(tmp_path / 'reconnected.png'))
      (tmp_path / 'browser-evidence.json').write_text(
          json.dumps(
              {
                  'network': (
                      'Chromium context offline emulation, not physical Android'
                  ),
                  'streams': page.evaluate('window.streams.length'),
                  'posts': posts,
                  'errors': errors,
                  'world_unchanged': game.world.get_state() == before,
              },
              indent=2,
          ),
          encoding='utf-8',
      )
      assert not posts
      assert not errors
      assert game.world.get_state() == before
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


def test_online_hint_does_not_enable_sending_before_fresh_snapshot(host):
  game, url = host()
  game.phase = 'running'
  _, reader = inputs.pending(game.inbox, 'Coordinator', 'pending')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page()
      track_streams(page)
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.fill('#action', 'Do not send without fresh authority')
      page.context.set_offline(True)
      pwlib.expect(page.locator('#act')).to_be_disabled()
      # Keep the native stream, but hold its first post-online message handler.
      page.evaluate("""()=>{
        const Original=window.EventSource;
        window.EventSource=class extends Original {
          set onmessage(handler){super.onmessage=e=>window.releaseSnapshot=()=>handler(e);}
        };
      }""")
      page.context.set_offline(False)
      page.wait_for_function('typeof window.releaseSnapshot === "function"')
      pwlib.expect(page.locator('#act')).to_be_disabled()
      pwlib.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      page.evaluate('window.releaseSnapshot()')
      pwlib.expect(page.locator('#act')).to_be_enabled()
      assert (
          page.locator('#action').input_value()
          == 'Do not send without fresh authority'
      )
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


def test_revoked_role_is_reauthorized_after_native_reconnect(host):
  game, url = host(shared=True)
  game.world.emit(
      'private_message', 'PRIVATE_OFFLINE_SENTINEL', ['Coordinator']
  )
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page()
      track_streams(page)
      page.goto(url)
      principal = inputs.approve(game, page, 'Coordinator')
      pwlib.expect(page.locator('#journal')).to_contain_text(
          'PRIVATE_OFFLINE_SENTINEL'
      )
      page.fill('#action', 'Private scoped draft')
      page.context.set_offline(True)
      pwlib.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      game.sessions.revoke(principal)
      page.context.set_offline(False)
      pwlib.expect(page.locator('#join')).to_be_visible()
      assert 'PRIVATE_OFFLINE_SENTINEL' not in page.content()
      assert page.locator('#action').input_value() == ''
      # An old authorized message cannot restore the revoked role.
      page.evaluate("""() => {
        const old = window.snapshots.find(text => text.includes('PRIVATE_OFFLINE_SENTINEL'));
        window.streams[0].dispatchEvent(new MessageEvent('message', {data: old}));
      }""")
      assert 'PRIVATE_OFFLINE_SENTINEL' not in page.content()
      pwlib.expect(page.locator('#join')).to_be_visible()
      pwlib.expect(page.locator('#join-submit')).to_be_enabled()
      page.context.set_offline(True)
      pwlib.expect(page.locator('#join-submit')).to_be_disabled()
    finally:
      browser.close()


def test_native_reconnect_does_not_resend_an_uncertain_received_action(host):
  game, url = host()
  game.phase = 'running'
  results, reader = inputs.pending(game.inbox, 'Coordinator', 'uncertain')
  before = game.world.get_state()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page()
      track_streams(page)
      inputs.delay_reply(page, fail=True)
      posts = []
      page.on(
          'request',
          lambda r: posts.append(r.url) if r.method == 'POST' else None,
      )
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      words = 'message Nell: Please keep this private.'
      page.fill('#action', words)
      page.click('#act')
      page.wait_for_function('window.replyArrived === true')
      reader.join(timeout=2)
      assert results == [words]
      page.context.set_offline(True)
      pwlib.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      inputs.release_reply(page)
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Confirmation is unavailable'
      )
      page.context.set_offline(False)
      pwlib.expect(page.locator('#connection')).to_contain_text('Connected')
      assert page.locator('#action').input_value() == words
      assert page.evaluate('window.streams.length') == 2
      assert len(posts) == 1
      assert results == [words]
      assert game.world.get_state() == before
      pwlib.expect(page.locator('#act')).to_be_disabled()
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)

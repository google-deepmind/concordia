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

"""Receipt-scoped waiting feedback with native Chromium virtual time."""

import json
import re

from concordia.examples.bellwether import focus_browser_test as browser_helpers
from concordia.examples.bellwether import submission_browser_test as inputs
import pytest

pwlib = pytest.importorskip('playwright.sync_api')
host_fixture = browser_helpers.host_fixture
no_resident_calls = inputs.no_resident_model_calls


def test_clock_starts_at_confirmation_not_send_and_stops_at_new_prompt(
    host, tmp_path
):
  game, url = host()
  game.phase = 'running'
  results, reader = inputs.pending(game.inbox, 'Coordinator', 'old-input')
  readers = [reader]
  before = game.world.get_state()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      page.clock.install()
      inputs.delay_reply(page)
      posts, errors = [], []
      page.on(
          'request',
          lambda request: posts.append(request.post_data_json)
          if request.method == 'POST'
          else None,
      )
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.fill('#action', 'wait')
      page.click('#act')
      page.wait_for_function('window.replyArrived === true')
      page.clock.fast_forward(60000)
      pwlib.expect(page.locator('#waiting-status')).to_be_hidden()
      inputs.release_reply(page)
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Action received'
      )
      page.clock.fast_forward(20000)
      pwlib.expect(page.locator('#waiting-status')).to_be_visible()
      text = page.locator('#waiting-status').inner_text()
      match = re.search(r'Confirmed (\d+) seconds', text)
      assert match is not None
      seconds = int(match.group(1))
      assert 15 <= seconds <= 25  # Not the minute spent awaiting confirmation.
      assert 'not a progress estimate' in text
      assert page.locator('#waiting-status').get_attribute('aria-live') == 'off'
      page.locator('#waiting-status').scroll_into_view_if_needed()
      page.screenshot(path=str(tmp_path / 'confirmed-wait.png'))
      (tmp_path / 'browser-evidence.json').write_text(
          json.dumps(
              {
                  'native_virtual_clock': True,
                  'held_reply_ms': 60000,
                  'advanced_after_confirmation_ms': 20000,
                  'display': text,
                  'page_errors': errors,
                  'posts_at_capture': posts,
              },
              indent=2,
          ),
          encoding='utf-8',
      )
      page.fill('#action', 'An unsent next-turn draft 🌊')
      reader.join(timeout=2)
      assert results == ['wait']
      _, next_reader = inputs.pending(game.inbox, 'Coordinator', 'new-input')
      readers.append(next_reader)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.clock.fast_forward(60000)
      pwlib.expect(page.locator('#waiting-status')).to_be_hidden()
      assert (
          page.locator('#action').input_value()
          == 'An unsent next-turn draft 🌊'
      )
      assert [x['operation'] for x in posts] == ['human.respond']
      assert game.world.get_state() == before
      assert not errors
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    finally:
      browser.close()
      game.inbox.finish('Test done')
      for thread in readers:
        thread.join(timeout=2)


def test_uncertain_reply_never_claims_confirmed_wait(host):
  game, url = host()
  game.phase = 'running'
  results, reader = inputs.pending(game.inbox, 'Coordinator', 'input')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 800, 'height': 360})
      page.clock.install()
      inputs.delay_reply(page, fail=True)
      page.goto(url)
      page.fill('#action', 'wait')
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.click('#act')
      inputs.release_reply(page)
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Confirmation is unavailable'
      )
      page.clock.fast_forward(120000)
      pwlib.expect(page.locator('#waiting-status')).to_be_hidden()
      assert page.locator('#action').input_value() == 'wait'
      reader.join(timeout=2)
      assert results == [
          'wait'
      ]  # Server accepted, but the page cannot infer it.
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


@pytest.mark.parametrize(
    'ending', ['failed', 'completed', 'revoked', 'stream_error', 'reload']
)
def test_wait_age_does_not_outlive_authority_or_known_run_state(host, ending):
  game, url = host(shared=ending == 'revoked')
  game.phase = 'running'
  results, reader = inputs.pending(game.inbox, 'Coordinator', 'input')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      page.clock.install()
      browser_helpers.track_snapshots(page)
      # Preserve the real stream, exposing it only to inject an error signal.
      page.add_init_script("""
        const Original = window.EventSource;
        window.EventSource = class extends Original {
          constructor(...args){super(...args);window.testStream=this;}
        };
      """)
      page.goto(url)
      principal = None
      if ending == 'revoked':
        principal = inputs.approve(game, page, 'Coordinator')
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.fill('#action', 'wait')
      page.click('#act')
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Action received'
      )
      page.clock.fast_forward(20000)
      pwlib.expect(page.locator('#waiting-status')).to_be_visible()
      page.fill('#action', 'Keep my local draft')
      reader.join(timeout=2)
      assert results == ['wait']
      if ending in ('failed', 'completed'):
        game.phase = ending
        game.inbox.finish('Lifecycle fixture closed')
        browser_helpers.publish(game, page)
      elif ending == 'revoked':
        assert principal is not None
        game.sessions.revoke(principal)
        browser_helpers.publish(game, page)
        pwlib.expect(page.locator('#join')).to_be_visible()
      elif ending == 'stream_error':
        # Synthetic error signal, not a claim of a physical network outage.
        page.evaluate("window.testStream.dispatchEvent(new Event('error'))")
        pwlib.expect(page.locator('#connection')).to_contain_text(
            'Reconnecting'
        )
      else:
        page.reload()
        pwlib.expect(page.locator('#connection')).to_contain_text('Connected')
        assert page.locator('#action').input_value() == 'Keep my local draft'
      page.clock.fast_forward(120000)
      pwlib.expect(page.locator('#waiting-status')).to_be_hidden()
      assert page.locator('#waiting-status').text_content() == ''
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)

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

"""Late delivery and scoped drafts; no simulation or resident/provider calls."""

import threading
from unittest import mock

from concordia.examples.astral_canticle import human_io
from concordia.examples.bellwether import focus_browser_test as browser_helpers
from concordia.examples.bellwether import game_prefab
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
import pytest

pwlib = pytest.importorskip('playwright.sync_api')
host_fixture = browser_helpers.host_fixture


@pytest.fixture(autouse=True)
def no_resident_model_calls():
  with (
      mock.patch.object(
          game_prefab.FixtureModel,
          'sample_text',
          side_effect=AssertionError('No resident model calls'),
      ),
      mock.patch.object(
          game_prefab.FixtureModel,
          'sample_choice',
          side_effect=AssertionError('No resident model calls'),
      ),
  ):
    yield


def pending(inbox, role, request_id):
  """Open one synthetic transport request, not a simulated entity turn."""
  results = []

  def read():
    try:
      results.append(
          inbox(
              human_input.HumanInputRequest(
                  request_id=request_id,
                  entity_name=role,
                  action_spec=entity_lib.free_action_spec(
                      call_to_action='Your turn'
                  ),
                  contexts={},
                  context=(
                      'Only this controlled role sees this fixture context.'
                  ),
              )
          )
      )
    except human_io.InputClosed:
      pass

  thread = threading.Thread(target=read)
  thread.start()
  return results, thread


def delay_reply(page, fail=False):
  page.add_init_script(
      'window.failHeldReply = ' + ('true;' if fail else 'false;')
  )
  page.add_init_script("""
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      const body = args[1]?.body;
      if (body && JSON.parse(body).operation === 'human.respond') {
        window.replyArrived = true;
        const originalJson = response.json.bind(response);
        response.json = async () => {const value = await originalJson();window.replyDecoded = true;return value;};
        await new Promise(resolve => window.releaseReply = resolve);
        if(window.failHeldReply){window.replyDecoded=true;throw new Error("PRIVATE_OLD_REPLY_ERROR");}
      }
      return response;
    };
  """)


def release_reply(page):
  page.wait_for_function('window.replyArrived === true')
  page.evaluate('window.releaseReply()')
  page.wait_for_function('window.replyDecoded === true')


def approve(game, page, role):
  page.fill('#join-name', 'Browser ' + role)
  page.select_option('#join-role', role)
  page.click('#join-submit')
  pwlib.expect(page.locator('#join-status')).to_contain_text('Waiting for host')
  row = next(
      row
      for row in game.sessions.pending()
      if row['label'] == 'Browser ' + role
  )
  game.sessions.approve(row['id'])
  browser_helpers.publish(game, page)
  pwlib.expect(page.locator('#join')).to_be_hidden()
  return row['id']


def test_late_reply_preserves_same_text_reentered_as_a_new_draft(host):
  game, url = host()
  results, reader = pending(game.inbox, 'Coordinator', 'old-input')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      delay_reply(page)
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      words = 'message Nell: Keep this for my next turn.'
      page.fill('#action', words)
      page.click('#act')
      page.wait_for_function('window.replyArrived === true')
      reader.join(timeout=2)
      assert results == [words]
      page.fill('#action', '')
      page.fill('#action', words)
      release_reply(page)
      # response.json instrumentation has drained the reply handler.
      assert page.locator('#action').input_value() == words
      page.reload()
      pwlib.expect(page.locator('#connection')).to_contain_text('Connected')
      assert page.locator('#action').input_value() == words
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


@pytest.mark.parametrize('fail', [False, True])
def test_old_role_reply_cannot_clear_new_roles_identical_draft(host, fail):
  game, url = host(shared=True)
  results, reader = pending(game.inbox, 'Coordinator', 'coordinator-input')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 800, 'height': 360})
      browser_helpers.track_snapshots(page)
      delay_reply(page, fail=fail)
      page.goto(url)
      principal = approve(game, page, 'Coordinator')
      pwlib.expect(page.locator('#act')).to_be_enabled()
      words = 'message Nell: The same words in a separate draft.'
      page.fill('#action', words)
      page.click('#act')
      page.wait_for_function('window.replyArrived === true')
      reader.join(timeout=2)
      assert results == [words]
      game.sessions.revoke(principal)
      browser_helpers.publish(game, page)
      pwlib.expect(page.locator('#join')).to_be_visible()
      approve(game, page, 'Nell')
      page.fill('#action', words)
      release_reply(page)
      # Late old-role completion must not own this role's draft/status.
      assert page.locator('#submission-status').inner_text() == ''
      assert page.locator('#action').input_value() == words
      assert page.locator('#error').inner_text() == ''
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_delayed_ack_explains_delivery_without_claiming_outcome(
    host, tmp_path, width, height
):
  game, url = host()
  results, reader = pending(game.inbox, 'Coordinator', 'slow-input')
  before = game.world.get_state()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      delay_reply(page)
      errors, posts = [], []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.on(
          'request',
          lambda request: posts.append(request.url)
          if request.method == 'POST'
          else None,
      )
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.fill('#action', 'message Nell: Can we talk?')
      page.click('#act')
      page.wait_for_function('window.replyArrived === true')
      pwlib.expect(page.locator('#act')).to_have_text('Sending…')
      pwlib.expect(page.locator('#act')).to_have_attribute('aria-busy', 'true')
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'confirmation is pending'
      )
      pwlib.expect(page.locator('#act')).to_be_disabled()
      page.locator('#submission-status').scroll_into_view_if_needed()
      page.screenshot(path=str(tmp_path / 'sending.png'))
      release_reply(page)
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Delivery is confirmed, not its outcome'
      )
      pwlib.expect(page.locator('#act')).to_have_attribute('aria-busy', 'false')
      assert page.locator('#action').input_value() == ''
      reader.join(timeout=2)
      assert results == ['message Nell: Can we talk?']
      assert len(posts) == 1
      assert game.world.get_state() == before
      assert page.evaluate('document.documentElement.scrollWidth') <= width
      page.reload()
      pwlib.expect(page.locator('#connection')).to_contain_text('Connected')
      assert page.locator('#action').input_value() == ''
      assert not errors
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


@pytest.mark.parametrize('loss', ['before_send', 'after_acceptance'])
def test_network_uncertainty_keeps_draft_and_does_not_auto_resubmit(host, loss):
  game, url = host()
  results, reader = pending(game.inbox, 'Coordinator', 'network-input')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      page.add_init_script(
          'window.loseAfterAcceptance = '
          + ('true;' if loss == 'after_acceptance' else 'false;')
      )
      page.add_init_script("""
        const originalFetch=window.fetch;
        window.envelopes=[];
        window.fetch=async(...args)=>{
          if(args[1]?.body && JSON.parse(args[1].body).operation==='human.respond'){
            window.envelopes.push(JSON.parse(args[1].body));
            if(window.envelopes.length===1){
              if(window.loseAfterAcceptance)await originalFetch(...args);
              throw new TypeError('Controlled connection loss');
            }
          }
          return originalFetch(...args);
        };
      """)
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      words = 'message Nell: A careful proposal.'
      page.fill('#action', words)
      page.click('#act')
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Confirmation is unavailable'
      )
      assert page.locator('#action').input_value() == words
      assert len(page.evaluate('window.envelopes')) == 1
      if loss == 'before_send':
        assert not results
        pwlib.expect(page.locator('#act')).to_be_enabled()
        page.click('#act')
        pwlib.expect(page.locator('#submission-status')).to_contain_text(
            'Delivery is confirmed'
        )
        first, second = page.evaluate('window.envelopes')
        assert first == second
      else:
        pwlib.expect(page.locator('#act')).to_be_disabled()
        page.reload()
        pwlib.expect(page.locator('#connection')).to_contain_text('Connected')
        assert page.locator('#action').input_value() == words
        assert page.evaluate('window.envelopes') == []
      reader.join(timeout=2)
      assert results == [words]
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)


def test_rejection_leaves_editable_draft_and_received_status_expires_next_turn(
    host,
):
  game, url = host()
  results, reader = pending(game.inbox, 'Coordinator', 'validation-input')
  second_reader = None
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page()
      page.goto(url)
      pwlib.expect(page.locator('#act')).to_be_enabled()
      page.fill('#action', 'ambiguous allocation')
      page.click('#act')
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'not accepted'
      )
      pwlib.expect(page.locator('#act')).to_be_enabled()
      assert page.locator('#action').input_value() == 'ambiguous allocation'
      assert not results
      page.fill('#action', 'wait')
      page.click('#act')
      pwlib.expect(page.locator('#submission-status')).to_contain_text(
          'Delivery is confirmed'
      )
      reader.join(timeout=2)
      assert results == ['wait']
      _, second_reader = pending(game.inbox, 'Coordinator', 'next-input')
      pwlib.expect(page.locator('#act')).to_be_enabled()
      assert page.locator('#submission-status').inner_text() == ''
    finally:
      browser.close()
      game.inbox.finish('Test done')
      reader.join(timeout=2)
      if second_reader is not None:
        second_reader.join(timeout=2)

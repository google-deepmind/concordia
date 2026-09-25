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

"""Actual browser assets with controlled public state, not a simulation.

These optional regressions cover observed reading and returning-player failures.
Install Playwright and its Chromium browser to run them.
"""

import pathlib
import re

import pytest

playwright = pytest.importorskip('playwright.sync_api')
expect = playwright.expect


@pytest.fixture
def public_page():
  static = pathlib.Path(__file__).with_name('static')
  shared = static.parent.parent / 'astral_canticle/static/browser-storage.js'
  state = {
      'revision': 1,
      'finished': False,
      'status': 'Your move',
      'pending': {'id': 'first-request'},
      'game': {
          'mode': 'fixture',
          'session_id': 'first-game',
          'turn': 1,
          'events': [],
          'ending': None,
          'votes': {},
      },
  }
  submissions = []
  with playwright.sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={'width': 390, 'height': 844})

    def route(request):
      path = request.request.url.removeprefix('http://journey.test/')
      if path == 'api/state':
        request.fulfill(json=state)
      elif path == 'api/action':
        submissions.append(request.request.post_data_json)
        request.abort()
      else:
        asset = (
            shared
            if path == 'shared/browser-storage.js'
            else static / (path.removeprefix('static/') or 'index.html')
        )
        content_type = {
            '.js': 'text/javascript',
            '.css': 'text/css',
            '.html': 'text/html',
        }[asset.suffix]
        request.fulfill(path=str(asset), content_type=content_type)

    page.route('http://journey.test/**', route)
    page.goto('http://journey.test/')
    expect(page.locator('#send')).to_be_enabled()
    expect(page.locator('#reply-kind')).to_contain_text('scripted replies')
    try:
      yield page, state, submissions
    finally:
      browser.close()


def test_copying_final_offer_survives_completed_state_polling(public_page):
  page, state, _ = public_page
  state.update(revision=10, pending=None, finished=True, status='Complete')
  state['game'].update(
      mode='qwen3:8b',
      turn=3,
      ending='Agreement reached',
      votes={'Maya': 'ACCEPT', 'Leon': 'ACCEPT'},
      events=[
          {'step': 7, 'actor': 'You', 'text': 'You: Quiet song, then silence.'},
          {'step': 8, 'actor': 'Maya', 'text': 'Maya: ACCEPT'},
          {'step': 9, 'actor': 'Leon', 'text': 'Leon: ACCEPT'},
      ],
  )
  expect(page.locator('#result')).to_be_visible()
  expect(page.locator('#reply-kind')).to_contain_text('AI characters')
  page.locator('#final-offer').evaluate(
      '(node) => {const range=document.createRange();'
      'range.selectNodeContents(node);const selection=window.getSelection();'
      'selection.removeAllRanges();selection.addRange(range);}'
  )
  page.wait_for_timeout(1400)  # Two ordinary polls must preserve the selection.
  assert page.evaluate('window.getSelection().toString()') == (
      'Quiet song, then silence.'
  )


@pytest.mark.parametrize('uncertain_send', [False, True])
def test_return_after_host_restart_offers_draft_without_resending(
    public_page, uncertain_send
):
  page, state, submissions = public_page
  draft = 'My carefully written offer before the host restarted.'
  page.locator('#reply').fill(draft)
  if uncertain_send:
    page.locator('#send').click()
    expect(page.locator('#send')).to_have_text('Check last send')
  # Unload the old UI before the host changes: an in-memory transition cannot
  # rescue this draft. Session storage belongs to this same browser tab.
  page.goto('about:blank')
  state['game']['session_id'] = 'new-game'
  state['pending']['id'] = 'new-request'
  page.goto('http://journey.test/')
  expect(page.locator('#send')).to_have_text('Say it')
  expect(page.locator('#reply')).to_have_value('')
  expect(page.locator('#error')).to_contain_text('fresh game')
  expect(page.locator('#restore-draft')).to_have_text(
      'Use draft from previous game'
  )
  page.locator('#restore-draft').click()
  expect(page.locator('#reply')).to_have_value(draft)
  expect(page.locator('#restore-draft')).to_be_hidden()
  assert len(submissions) == int(uncertain_send)


def test_wait_elapsed_does_not_restart_when_the_player_reloads(public_page):
  page, state, _ = public_page
  state.update(revision=2, pending=None, status='Waiting')
  state['game']['events'] = [
      {'step': 1, 'actor': 'You', 'text': 'You: Could we have one last song?'}
  ]
  expect(page.locator('#phase')).to_contain_text('Maya’s reply')
  expect(page.locator('#phase')).to_contain_text(
      re.compile(r'\((?:[2-9]|[1-9]\d+)s\)')
  )
  before_match = re.search(r'\((\d+)s\)', page.locator('#phase').inner_text())
  assert before_match is not None
  before = int(before_match[1])
  page.reload()
  expect(page.locator('#phase')).to_contain_text('Maya’s reply')
  after_match = re.search(r'\((\d+)s\)', page.locator('#phase').inner_text())
  assert after_match is not None
  after = int(after_match[1])
  assert after >= before


@pytest.mark.parametrize(
    ('finished', 'ending', 'label', 'target'),
    [
        (False, None, 'Conversation underway', '#conversation'),
        (True, 'No shared agreement', 'Conversation complete', '#result'),
        (True, None, 'Conversation stopped', '#conversation'),
    ],
)
def test_late_arrival_can_find_existing_conversation_without_reset(
    public_page, finished, ending, label, target
):
  page, state, submissions = public_page
  state.update(revision=8, finished=finished, pending=None)
  state['game'].update(
      turn=3,
      ending=ending,
      events=[
          {
              'step': i,
              'actor': 'Maya',
              'text': 'Maya: ' + 'A recorded reply. ' * 20,
          }
          for i in range(1, 8)
      ],
  )
  page.reload()
  arrival = page.locator('.start-link')
  expect(arrival).to_contain_text(label)
  arrival.click()
  expect(page.locator(target)).to_be_in_viewport()
  assert page.locator('#conversation article').count() == 7
  assert not submissions

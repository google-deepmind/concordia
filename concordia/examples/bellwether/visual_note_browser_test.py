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

"""Actual mobile file review, late-read guards and private role boundaries."""

import io
import json
import threading
from unittest import mock

from concordia.examples.astral_canticle import human_io
from concordia.examples.bellwether import visual_note
from concordia.examples.bellwether.focus_browser_test import host_fixture  # pylint: disable=unused-import
from concordia.examples.bellwether.focus_browser_test import publish
from concordia.examples.bellwether.focus_browser_test import track_snapshots
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
from PIL import Image
import pytest

pw = pytest.importorskip('playwright.sync_api')


def note_bytes(text):
  image = io.BytesIO()
  Image.new('RGB', (8, 8), 'red').save(image, format='PNG')
  with mock.patch.object(visual_note.ollama, 'Client') as client:
    client.return_value.show.return_value = {'capabilities': ['vision']}
    client.return_value.chat.return_value = {
        'done': True,
        'done_reason': 'stop',
        'message': {'content': text},
    }
    provider = visual_note.OllamaVisualDraft('fixture-vision')
    note = provider.describe(
        visual_note.prepare_image(image.getvalue()), 'Describe the diagram.'
    )
  return json.dumps(note, ensure_ascii=False).encode('utf-8')


def pending(inbox, role='Coordinator'):
  results = []
  request = human_input.HumanInputRequest(
      request_id='note-fixture-' + role,
      entity_name=role,
      action_spec=entity_lib.free_action_spec(call_to_action='Your words?'),
      contexts={},
      context='Only this controlled role sees the fixture prompt.',
  )

  def wait():
    try:
      results.append(inbox(request))
    except human_io.InputClosed:
      pass

  thread = threading.Thread(target=wait)
  thread.start()
  return results, thread


def choose(page, data, name='note.json'):
  page.locator('#note-file').set_input_files(
      {'name': name, 'mimeType': 'application/json', 'buffer': data}
  )


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_mobile_local_review_errors_preserve_draft_and_send_only_explicitly(
    host, tmp_path, width, height
):
  game, url = host()
  results, thread = pending(game.inbox)
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height},
          has_touch=True,
          is_mobile=True,
      )
      track_snapshots(page)
      errors, posts = [], []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.on(
          'request', lambda r: posts.append(r) if r.method == 'POST' else None
      )
      page.goto(url)
      pw.expect(page.locator('#act')).to_be_enabled()
      page.locator('#visual-note > summary').click()
      page.fill('#action', 'message Nell: My draft.')
      before = game.world.get_state()
      words = 'A red sign says "é & 🌊".\n<img src=x onerror=unsafe()>'
      choose(page, note_bytes(words))
      pw.expect(page.locator('#note-text')).to_have_value(words)
      pw.expect(page.locator('#note-summary')).to_contain_text('fixture-vision')
      assert page.locator('#action').input_value() == 'message Nell: My draft.'
      assert not posts
      assert page.locator('img').count() == 0
      page.fill('#note-text', 'My reviewed description.\nStill unverified.')
      page.locator('#note-text').evaluate('e=>e.setSelectionRange(3,10)')
      publish(game, page)
      pw.expect(page.locator('#note-text')).to_be_focused()
      assert page.locator('#note-text').evaluate(
          'e=>[e.selectionStart,e.selectionEnd]'
      ) == [3, 10]
      # Invalid imports preserve both the edited review and action draft.
      for invalid in (b'[]', b'{', b'\xff', b'x' * (128 * 1024 + 1)):
        choose(page, invalid)
        pw.expect(page.locator('#note-status')).to_contain_text('kept')
        assert (
            page.locator('#note-text').input_value().startswith('My reviewed')
        )
        assert (
            page.locator('#action').input_value() == 'message Nell: My draft.'
        )
      page.screenshot(
          path=str(tmp_path / f'review-{width}.png'), full_page=True
      )
      assert page.evaluate('document.documentElement.scrollWidth') <= width
      assert not posts
      assert game.world.get_state() == before
      page.click('#use-note')
      expected = (
          'message Nell: My draft. My reviewed description.\nStill unverified.'
      )
      assert page.locator('#action').input_value() == expected
      assert page.locator('#note-text').input_value() == ''
      assert not posts
      assert not results
      # Real failed SSE reconnect keeps only the explicitly appended draft.
      page.route('**/api/events', lambda route: route.abort('failed'))
      page.reload()
      pw.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      assert page.locator('#note-text').input_value() == ''
      page.unroute('**/api/events')
      pw.expect(page.locator('#act')).to_be_enabled(timeout=15000)
      assert page.locator('#action').input_value() == expected
      page.click('#act')
      thread.join(2)
      assert results == [expected]
      assert len(posts) == 1
      assert game.world.get_state() == before
      assert not errors
    finally:
      game.inbox.finish('Fixture closed')
      thread.join(2)
      browser.close()


def test_late_file_cancel_and_replacement_never_overwrite_review(host):
  game, url = host()
  _, thread = pending(game.inbox)
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      page.add_init_script("""
        const original=File.prototype.arrayBuffer;
        File.prototype.arrayBuffer=function(){
          if(this.name==='delayed.json')return new Promise(resolve=>{
            window.releaseFile=async()=>{resolve(await original.call(this))};
          });
          return original.call(this);
        };
      """)
      page.goto(url)
      pw.expect(page.locator('#act')).to_be_enabled()
      page.locator('#visual-note > summary').click()
      page.fill('#action', 'Preserved draft')
      choose(page, note_bytes('Late private caption'), 'delayed.json')
      pw.expect(page.locator('#note-status')).to_contain_text('Reading')
      page.click('#clear-note')
      page.evaluate('releaseFile()')
      pw.expect(page.locator('#note-text')).to_have_value('')
      assert page.locator('#action').input_value() == 'Preserved draft'
      choose(page, note_bytes('Older caption'), 'delayed.json')
      pw.expect(page.locator('#note-status')).to_contain_text('Reading')
      choose(page, note_bytes('New caption'))
      pw.expect(page.locator('#note-text')).to_have_value('New caption')
      page.evaluate('releaseFile()')
      pw.expect(page.locator('#note-text')).to_have_value('New caption')
    finally:
      game.inbox.finish('Fixture closed')
      thread.join(2)
      browser.close()


def test_private_nell_review_never_reaches_other_roles_and_revocation_clears(
    host,
):
  game, url = host(shared=True)
  _, thread = pending(game.nell_inbox, 'Nell')
  with pw.sync_playwright() as runner:
    browser = runner.chromium.launch()
    try:
      contexts = [browser.new_context() for _ in range(3)]
      pages = [context.new_page() for context in contexts]
      identities = []
      for page, role in zip(pages, ['Nell', 'Coordinator', 'spectator']):
        track_snapshots(page)
        page.goto(url)
        page.fill('#join-name', 'Fixture ' + role)
        page.select_option('#join-role', role)
        page.click('#join-submit')
        pw.expect(page.locator('#join-status')).to_contain_text(
            'Waiting for host'
        )
        row = next(
            r
            for r in game.sessions.pending()
            if r['label'] == 'Fixture ' + role
        )
        game.sessions.approve(row['id'])
        identities.append(row['id'])
        publish(game, page)
        pw.expect(page.locator('#join')).to_be_hidden()
      nell, coordinator, spectator = pages
      pw.expect(nell.locator('#visual-note')).to_be_visible()
      pw.expect(coordinator.locator('#visual-note')).to_be_hidden()
      pw.expect(spectator.locator('#visual-note')).to_be_hidden()
      nell.locator('#visual-note > summary').click()
      choose(nell, note_bytes('PRIVATE_NOTE_ONLY_IN_NELL_BROWSER'))
      pw.expect(nell.locator('#note-text')).to_have_value(
          'PRIVATE_NOTE_ONLY_IN_NELL_BROWSER'
      )
      assert 'PRIVATE_NOTE_ONLY_IN_NELL_BROWSER' not in json.dumps(
          game.operations.snapshot('developer')
      )
      assert coordinator.locator('#note-text').input_value() == ''
      assert spectator.locator('#note-text').input_value() == ''
      nell.evaluate("""()=>{
        const original=File.prototype.arrayBuffer;
        File.prototype.arrayBuffer=function(){return new Promise(resolve=>{
          window.releaseRevokedFile=async()=>resolve(await original.call(this));
        })};
      }""")
      choose(nell, note_bytes('LATE_PRIVATE_REVOKED_NOTE'))
      pw.expect(nell.locator('#note-status')).to_contain_text('Reading')
      game.sessions.revoke(identities[0])
      publish(game, nell)
      pw.expect(nell.locator('#join')).to_be_visible()
      assert nell.locator('#note-text').input_value() == ''
      assert nell.locator('#note-provenance').inner_text() == ''
      assert nell.locator('#note-summary').inner_text() == ''
      nell.evaluate('releaseRevokedFile()')
      pw.expect(nell.locator('#note-text')).to_have_value('')
      assert nell.locator('#note-file').input_value() == ''
    finally:
      game.nell_inbox.finish('Fixture closed')
      thread.join(2)
      browser.close()

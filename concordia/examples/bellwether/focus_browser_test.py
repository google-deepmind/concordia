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

"""Native focus, selection and role clearing through real scoped SSE updates."""

import pathlib
from unittest import mock

from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import multiplayer
from concordia.prefabs.simulation import generic
from concordia.utils import simulation_server
import pytest

pwlib = pytest.importorskip('playwright.sync_api')


@pytest.fixture(name='host')
def host_fixture(tmp_path):
  owned = []

  def create(shared=False):
    game = (
        multiplayer.SharedGame(tmp_path, secure=False)
        if shared
        else game_service.Game(tmp_path)
    )
    server = simulation_server.SimulationServer(
        port=0,
        operation_service=game.operations,
        audience='player',
        browser_sessions=game.sessions
        if isinstance(game, multiplayer.SharedGame)
        else None,
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
    )
    server.start()
    owned.append((game, server))
    return game, f'http://127.0.0.1:{server.bound_port}'

  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No simulation')
  ):
    try:
      yield create
    finally:
      for game, server in owned:
        game.close()
        server.stop()


def track_snapshots(page):
  page.add_init_script("""
    window.snapshotCount=0;
    const Original=window.EventSource;
    window.EventSource=class extends Original {
      constructor(...args){super(...args);this.addEventListener('message',()=>window.snapshotCount++);}
    };
  """)


def publish(game, page):
  count = page.evaluate('window.snapshotCount')
  game.operations.publish({'kind': 'test.unchanged_snapshot'})
  page.wait_for_function('count=>window.snapshotCount>count', arg=count)


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_controls_and_reading_selection_survive_real_updates(
    host, width, height
):
  game, url = host()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      track_snapshots(page)
      errors, mutations = [], []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.on(
          'request',
          lambda request: mutations.append(request.url)
          if request.method == 'POST'
          else None,
      )
      page.goto(url)
      pwlib.expect(page.locator('#cast button').first).to_be_visible()
      before = game.world.get_state()
      for locator in ['#cast button', '#choices button']:
        control = page.locator(locator).first
        control.focus()
        publish(game, page)
        pwlib.expect(control).to_be_focused()
        # Native Enter still edits the draft, not a sent action.
        page.keyboard.press('Enter')
        pwlib.expect(page.locator('#action')).to_be_focused()
        assert page.locator('#action').input_value()
      page.fill('#action', 'My careful draft')
      page.locator('#action').evaluate('e=>e.setSelectionRange(3,10)')
      publish(game, page)
      assert page.locator('#action').evaluate(
          'e=>[e.selectionStart,e.selectionEnd]'
      ) == [3, 10]
      assert page.locator('#action').input_value() == 'My careful draft'
      assert game.world.get_state() == before
      page.evaluate("""()=>{
        const span=document.querySelector('#journal p span');
        window.firstStory=span;const range=document.createRange();
        range.selectNodeContents(span);const selection=getSelection();
        selection.removeAllRanges();selection.addRange(range);
        window.selectedStory=selection.toString();
      }""")
      publish(game, page)
      assert page.evaluate('getSelection().toString()===window.selectedStory')
      game.world.emit('speech', 'New public fixture note; no turn was played.')
      publish(game, page)
      pwlib.expect(page.locator('#journal')).to_contain_text(
          'New public fixture note'
      )
      assert page.evaluate('getSelection().toString()===window.selectedStory')
      assert page.evaluate('window.firstStory.isConnected')
      assert not mutations
      assert not errors
      assert page.evaluate('document.documentElement.scrollWidth') <= width
      # Changed filtering and completion still replace unavailable controls.
      page.locator('#map [data-location="Generator yard"]').click()
      assert page.locator('#cast article').count() == 1
      game.inbox.finish('Test completion fixture')
      publish(game, page)
      assert page.locator('#cast button').count() == 0
      assert page.locator('#choices button').count() == 0
    finally:
      browser.close()


def test_role_revocation_clears_retained_journal_and_controls(host):
  game, url = host(shared=True)
  game.world.emit('private_message', 'PRIVATE_FOCUS_SENTINEL', ['Coordinator'])
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      track_snapshots(page)
      page.goto(url)

      def approve(role):
        page.fill('#join-name', 'Test ' + role)
        page.select_option('#join-role', role)
        page.click('#join-submit')
        pwlib.expect(page.locator('#join-status')).to_contain_text(
            'Waiting for host'
        )
        row = next(
            row
            for row in game.sessions.pending()
            if row['label'] == 'Test ' + role
        )
        game.sessions.approve(row['id'])
        publish(game, page)
        pwlib.expect(page.locator('#join')).to_be_hidden()
        return row['id']

      principal = approve('Coordinator')
      pwlib.expect(page.locator('#journal')).to_contain_text(
          'PRIVATE_FOCUS_SENTINEL'
      )
      page.fill('#action', 'Private draft')
      game.sessions.revoke(principal)
      publish(game, page)
      pwlib.expect(page.locator('#join')).to_be_visible()
      assert 'PRIVATE_FOCUS_SENTINEL' not in page.content()
      assert page.locator('#action').input_value() == ''
      approve('spectator')
      assert 'PRIVATE_FOCUS_SENTINEL' not in page.content()
      assert not page.locator('#composer').is_visible()
      assert page.locator('#cast button').count() == 0
      assert page.locator('#choices button').count() == 0
      pwlib.expect(page.locator('#journal')).to_contain_text(
          'Rain blows sideways'
      )
    finally:
      browser.close()

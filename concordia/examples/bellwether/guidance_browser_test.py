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

"""First-play browser checks on real services; no simulation starts."""

import pathlib
from unittest import mock

from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import multiplayer
from concordia.utils import simulation_server
import pytest

pwlib = pytest.importorskip('playwright.sync_api')


@pytest.fixture(name='host')
def host_fixture(tmp_path):
  servers = []
  games = []
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation in guidance tests'),
  ):

    def make(shared=False):
      game = (
          multiplayer.SharedGame(tmp_path, secure=False)
          if shared
          else game_service.Game(tmp_path)
      )
      server = simulation_server.SimulationServer(
          port=0,
          operation_service=game.operations,
          audience='player',
          browser_sessions=(
              game.sessions
              if isinstance(game, multiplayer.SharedGame)
              else None
          ),
          html_content=pathlib.Path(__file__)
          .with_name('player.html')
          .read_text(encoding='utf-8'),
      )
      games.append(game)
      servers.append(server)
      server.start()
      return game, f'http://127.0.0.1:{server.bound_port}'

    try:
      yield make
    finally:
      for game in games:
        game.close()
      for server in servers:
        server.stop()


def test_interpretation_is_free_literal_and_uses_shared_parser(host, tmp_path):
  game, url = host()
  initial = game.operations.snapshot('developer')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 360, 'height': 800})
      errors = []
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.goto(url)
      pwlib.expect(page.locator('#role-badge')).to_contain_text(
          'Single player · You are Coordinator Alice · Four scripted residents'
      )
      assert not page.locator('#join').is_visible()
      page.locator('#first-play summary').click()
      page.screenshot(
          path=str(tmp_path / 'first-play-mobile.png'), full_page=True
      )
      page.locator('#action').fill('ask Nell for fuel and part')
      with page.expect_request('**/api/dispatch') as request:
        page.click('#preview')
      assert request.value.post_data_json == {
          'operation': 'game.preview',
          'arguments': {'text': 'ask Nell for fuel and part'},
      }
      pwlib.expect(page.locator('#preview-result')).to_contain_text(
          'She decides whether to accept.'
      )
      assert (
          'no choice was spent' in page.locator('#preview-result').inner_text()
      )
      page.locator('#action').fill('invent fuel magically')
      page.click('#preview')
      pwlib.expect(page.locator('#preview-result')).to_contain_text(
          'Nothing changed'
      )
      assert page.locator('#action').input_value() == 'invent fuel magically'
      words = 'message everyone: "visible" <img src=x onerror=alert(1)> 🌊'
      page.locator('#action').fill(words)
      page.click('#preview')
      pwlib.expect(page.locator('#preview-result')).to_contain_text(
          'Public discussion to everyone'
      )
      assert '<img' in page.locator('#preview-result').inner_text()
      assert page.locator('#preview-result img').count() == 0
      assert game.operations.snapshot('developer') == initial
      assert page.evaluate('innerWidth') == 360
      assert page.evaluate('document.documentElement.scrollWidth') <= 360
      assert not errors
    finally:
      browser.close()


def test_delayed_preview_cannot_describe_a_changed_draft_and_reconnect(host):
  game, url = host()
  initial = game.operations.snapshot('developer')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 412, 'height': 915})
      # Delay only delivery after a REAL read-only server query. The delayed
      # Response ignores browser abort to exercise the stale-result guard too.
      page.add_init_script("""
        const originalFetch=window.fetch.bind(window);
        window.fetch=async(...args)=>{
          const response=await originalFetch(...args);
          if(window.holdPreview&&args[1]?.body?.includes('game.preview')){
            const body=await response.text();
            return new Promise(resolve=>{
              window.releasePreview=()=>resolve(new Response(body,{
                status:response.status,
                headers:{'Content-Type':'application/json'}
              }));
            });
          }
          return response;
        };
      """)
      page.goto(url)
      pwlib.expect(page.locator('#connection')).to_contain_text('Connected')
      page.locator('#action').fill('ask Nell for fuel')
      page.evaluate('window.holdPreview=true')
      page.click('#preview')
      page.wait_for_function('typeof window.releasePreview==="function"')
      page.locator('#action').fill('wait')
      page.evaluate("""async()=>{
        window.releasePreview();
        await new Promise(resolve=>setTimeout(resolve,50));
      }""")
      pwlib.expect(page.locator('#preview-result')).to_be_empty()
      page.evaluate('window.holdPreview=false')
      page.click('#preview')
      pwlib.expect(page.locator('#preview-result')).to_contain_text(
          'Spend one choice waiting when you send'
      )
      page.route('**/api/events', lambda route: route.abort('failed'))
      page.reload()
      pwlib.expect(page.locator('#connection')).to_contain_text('Reconnecting')
      pwlib.expect(page.locator('#preview')).to_be_disabled()
      # Draft is restored only after a session/role snapshot is known.
      page.unroute('**/api/events')
      pwlib.expect(page.locator('#connection')).to_contain_text(
          'Connected', timeout=15000
      )
      pwlib.expect(page.locator('#preview')).to_be_enabled()
      assert page.locator('#action').input_value() == 'wait'
      assert game.operations.snapshot('developer') == initial
    finally:
      browser.close()


@pytest.mark.parametrize('role', ['Nell', 'spectator'])
def test_other_roles_have_specific_guidance_not_coordinator_preview(host, role):
  game, url = host(shared=True)
  assert isinstance(game, multiplayer.SharedGame)
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(viewport={'width': 412, 'height': 915})
      page.goto(url)
      page.locator('#join-name').fill('Test ' + role)
      page.select_option('#join-role', role)
      page.click('#join-submit')
      pwlib.expect(page.locator('#join-status')).to_contain_text(
          'Waiting for host'
      )
      row = next(
          r for r in game.sessions.pending() if r['label'] == 'Test ' + role
      )
      game.sessions.approve(row['id'])
      game.operations.publish({'kind': 'test.host_approved'})
      pwlib.expect(page.locator('#join')).to_be_hidden()
      pwlib.expect(page.locator('#preview')).to_be_hidden()
      page.locator('#first-play summary').click()
      pwlib.expect(page.locator('#role-guide')).to_contain_text(
          'You control Nell' if role == 'Nell' else 'Watch public events'
      )
      if role == 'Nell':
        assert 'Two human roles' in page.locator('#role-badge').inner_text()
      else:
        assert not page.locator('#composer').is_visible()
    finally:
      browser.close()

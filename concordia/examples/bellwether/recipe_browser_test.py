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

"""Recipe browser openings, privacy and reload, without a played night."""

import json
import pathlib
from unittest import mock

from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import multiplayer
from concordia.examples.bellwether import researcher
from concordia.prefabs.simulation import generic
from concordia.utils import simulation_server
import pytest

pwlib = pytest.importorskip('playwright.sync_api')


@pytest.fixture(name='host')
def host_fixture(tmp_path):
  owned = []

  def create(recipe, shared=False):
    builder = multiplayer.SharedGame if shared else game_service.Game
    game = builder(
        tmp_path / recipe,
        recipe=recipe,
        **({'secure': False} if shared else {}),
    )
    server = simulation_server.SimulationServer(
        port=0,
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
        operation_service=game.operations,
        audience='player',
        browser_sessions=(
            game.sessions if isinstance(game, multiplayer.SharedGame) else None
        ),
    )
    server.start()
    owned.append((game, server))
    return game, f'http://127.0.0.1:{server.bound_port}'

  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No execution')
  ):
    try:
      yield create
    finally:
      for game, server in owned:
        game.close()
        server.stop()


@pytest.mark.parametrize('name', researcher.RECIPES)
def test_recipe_opening_stock_context_and_reload(host, tmp_path, name):
  game, url = host(name)
  before = game.operations.snapshot('developer')
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': 360, 'height': 800}, has_touch=True
      )
      errors, mutations = [], []
      page.on('pageerror', lambda error: errors.append(str(error)))
      page.on(
          'request',
          lambda request: mutations.append(request.url)
          if request.method == 'POST'
          else None,
      )
      page.goto(url)
      pwlib.expect(page.locator('#opening')).to_have_text(game.case.opening)
      if name == 'bellwether':
        pwlib.expect(page.locator('#recipe-note')).to_be_empty()
      else:
        pwlib.expect(page.locator('#recipe-note')).to_contain_text(
            name.replace('-', ' ')
        )
      if name == 'mutual-aid':
        pwlib.expect(page.locator('#resources')).to_contain_text(
            'Generator: 4 fuel'
        )
        pwlib.expect(page.locator('#recipe-note')).to_contain_text(
            '2 fuel used before play'
        )
        assert game.world.inventory_state()['Used']['fuel'] == 2
      if name == 'resource-governance':
        pwlib.expect(page.locator('#institutions')).to_contain_text(
            'Proposed charter'
        )
      if name == 'institutional-dispute':
        assert 'my recollection may be' not in page.content()
      page.reload()
      pwlib.expect(page.locator('#opening')).to_have_text(game.case.opening)
      pwlib.expect(page.locator('#begin')).to_be_enabled()
      assert game.operations.snapshot('developer') == before
      assert not mutations
      assert 'PRIVATE_' not in page.content()
      assert page.evaluate('document.documentElement.scrollWidth') <= 360
      assert not errors
      page.screenshot(
          path=str(tmp_path / (name + '-opening.png')), full_page=True
      )
    finally:
      browser.close()


def test_private_dispute_recipe_delivered_only_to_approved_nell(host):
  game, url = host('institutional-dispute', shared=True)
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      pages = {}
      for role in ['Nell', 'spectator']:
        page = browser.new_page(viewport={'width': 360, 'height': 800})
        page.goto(url)
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
        game.operations.publish({'kind': 'test.approved'})
        pwlib.expect(page.locator('#join')).to_be_hidden()
        pwlib.expect(page.locator('#recipe-note')).to_contain_text(
            'institutional dispute'
        )
        pages[role] = page
      for _ in range(4):
        game.world.resolve('Coordinator', 'wait')  # Component fixture only.
      game.operations.publish({'kind': 'test.boundary_fixture'})
      pwlib.expect(pages['Nell'].locator('#journal')).to_contain_text(
          'A disputed note'
      )
      assert 'A disputed note' not in pages['spectator'].content()
      assert 'Private initial account' not in pages['spectator'].content()
      # Read-only export strips the note even for its actual recipient.
      account = game.operations.dispatch(
          'developer',
          {'operation': 'game.public_account', 'arguments': {'format': 'json'}},
      )['result']['content']
      assert 'A disputed note' not in account
      assert json.loads(account)['backend'] == 'fixture'
    finally:
      browser.close()

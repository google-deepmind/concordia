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

"""Map/history accessibility using component fixtures, without an engine."""

import pathlib
from unittest import mock

from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether import game_service
from concordia.utils import simulation_server
import pytest

pwlib = pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize(
    ('width', 'height', 'forced'),
    [(360, 800, 'none'), (800, 360, 'none'), (360, 800, 'active')],
)
def test_map_status_history_and_free_inspection(
    tmp_path, width, height, forced
):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('Component fixtures only; no simulation'),
  ):
    game = game_service.Game(tmp_path)
    server = simulation_server.SimulationServer(
        port=0,
        operation_service=game.operations,
        audience='player',
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
    )
    server.start()
    try:
      with pwlib.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
          page = browser.new_page(
              viewport={'width': width, 'height': height},
              has_touch=True,
              forced_colors=forced,
          )
          errors, mutations = [], []
          page.on('pageerror', lambda e: errors.append(str(e)))
          page.on(
              'request',
              lambda r: mutations.append(r.url) if r.method == 'POST' else None,
          )
          page.goto(f'http://127.0.0.1:{server.bound_port}')
          beacon = page.locator('#map [data-location="Harbor beacon"]')
          pwlib.expect(beacon).to_contain_text('Not resolved yet')
          assert not beacon.evaluate("e=>e.classList.contains('unserved')")
          page.locator('#service-history summary').click()
          assert page.locator('#service-head th').all_text_contents() == [
              'Watch',
              'Beacon',
              'Shelter',
              'Cold store',
          ]
          assert page.locator('#service-rows td').all_text_contents() == (
              ['Not resolved'] * 9
          )
          # Pure scenario-component resolution supplies real accounting data
          # to the existing view. No entity.act/Simulation.play/model is run.
          game.world.resolve(rules.PLAYER, 'allocate shelter and cold store')
          for _ in range(3):
            game.world.resolve(rules.PLAYER, 'wait')
          game.operations.publish({'kind': 'test.resolved_watch_fixture'})
          pwlib.expect(beacon).to_contain_text(
              'Watch 1 · Early evening · Unserved'
          )
          assert beacon.evaluate("e=>e.classList.contains('unserved')")
          assert not beacon.evaluate("e=>e.classList.contains('lit')")
          shelter = page.locator('#map [data-location="Storm shelter"]')
          pwlib.expect(shelter).to_contain_text(
              'Watch 1 · Early evening · Maintained'
          )
          assert (
              'Watch 1 · Early evening · Maintained'
              in shelter.get_attribute('aria-label')
          )
          pwlib.expect(
              page.locator('#map [data-location="Generator yard"]')
          ).to_contain_text('4 fuel available')
          assert page.locator('#service-rows td').all_text_contents() == [
              'Unserved',
              'Served',
              'Served',
              *(['Not resolved'] * 6),
          ]
          assert page.locator('#service-rows th[scope=row]').count() == 3
          assert page.locator('#service-head th[scope=col]').count() == 4
          before = game.operations.snapshot('developer')
          for button in page.locator('#map button').all():
            button.focus()
            page.keyboard.press('Enter')
            assert button.get_attribute('aria-pressed') == 'true'
          assert game.operations.snapshot('developer') == before
          assert not mutations
          assert 'PRIVATE_NELL' not in page.content()
          assert page.evaluate('innerWidth') == width
          assert page.evaluate('document.documentElement.scrollWidth') <= width
          page.screenshot(
              path=str(tmp_path / 'service-history.png'), full_page=True
          )
          page.add_style_tag(content=':root{font-size:32px}')
          assert page.evaluate('document.documentElement.scrollWidth') <= width
          assert not errors
        finally:
          browser.close()
    finally:
      game.close()
      server.stop()

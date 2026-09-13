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

"""Read actual boundary diagnostics in Chromium, with no engine execution."""

from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether.focus_browser_test import host_fixture  # pylint: disable=unused-import
from concordia.examples.bellwether.focus_browser_test import publish
from concordia.examples.bellwether.focus_browser_test import track_snapshots
import pytest

pwlib = pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_free_boundary_inspection_matches_record_and_survives_sse(
    host, tmp_path, width, height
):
  game, url = host()
  game.world.transfer('Generator', 'Used', 'fuel', 5)
  for words in ['allocate cold store and shelter', 'wait', 'wait', 'wait']:
    game.world.resolve(rules.PLAYER, words)
  records = game.world.view()['services']
  before = game.world.get_state()
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page(
          viewport={'width': width, 'height': height}, has_touch=True
      )
      track_snapshots(page)
      errors, posts = [], []
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.on(
          'request',
          lambda r: posts.append(r.url) if r.method == 'POST' else None,
      )
      page.goto(url)
      page.locator('#service-history summary').click()
      assert page.locator('#service-rows button').count() == 3
      for record in records:
        label = 'Watch 1 · Early evening · ' + record['facility'] + ' · '
        button = page.get_by_role(
            'button',
            name=label
            + ('Served' if record['served'] else 'Unserved')
            + ' · Boundary details',
            exact=True,
        )
        before_click = game.operations.snapshot('developer')
        button.focus()
        page.keyboard.press('Enter')
        expected = (
            label[:-3]
            + '\n'
            + record['resolution']['explanation']
            + '\n'
            + record['consequence']
        )
        pwlib.expect(page.locator('#service-explanation')).to_have_text(
            expected
        )
        assert game.operations.snapshot('developer') == before_click
        publish(game, page)
        pwlib.expect(button).to_be_focused()
        pwlib.expect(page.locator('#service-explanation')).to_have_text(
            expected
        )
        bounds = button.bounding_box()
        assert bounds is not None and bounds['height'] >= 44
      assert game.world.get_state() == before
      assert not posts and not errors
      assert page.evaluate('document.documentElement.scrollWidth') <= width
      page.screenshot(path=str(tmp_path / 'boundary-notes.png'), full_page=True)
      page.reload()
      page.locator('#service-history summary').click()
      page.locator('#service-rows button').first.click()
      pwlib.expect(page.locator('#service-explanation')).to_contain_text(
          'not requested'
      )
      page.add_style_tag(content=':root{font-size:32px}')
      assert page.evaluate('document.documentElement.scrollWidth') <= width
      assert game.world.get_state() == before
      assert not posts and not errors
    finally:
      browser.close()


def test_legacy_unknown_and_literal_text_remain_inert(host):
  game, url = host()
  for _ in range(4):
    game.world.resolve(rules.PLAYER, 'wait')
  game.world.data['services'][0].pop('resolution')
  words = '<img src=x onerror="window.injected=true"> & literal details'
  game.world.data['services'][1]['resolution']['explanation'] = words
  with pwlib.sync_playwright() as pw:
    browser = pw.chromium.launch()
    try:
      page = browser.new_page()
      page.goto(url)
      page.locator('#service-history summary').click()
      page.locator('#service-rows button').nth(0).click()
      pwlib.expect(page.locator('#service-explanation')).to_contain_text(
          'Boundary details were not recorded.'
      )
      page.locator('#service-rows button').nth(1).click()
      pwlib.expect(page.locator('#service-explanation')).to_contain_text(words)
      assert page.locator('#service-explanation img').count() == 0
      assert not page.evaluate('Boolean(window.injected)')
    finally:
      browser.close()

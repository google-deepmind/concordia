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

"""Public SVG provenance, privacy and real offline browser rendering."""

import json
import pathlib
import subprocess
import sys
from unittest import mock
from xml.etree import ElementTree as etree

from concordia.examples.bellwether import game_service
from concordia.prefabs.simulation import generic
from concordia.utils import operation_service as ops
from concordia.utils import simulation_server
import pytest

SVG = '{http://www.w3.org/2000/svg}'
PRIVATE = 'PRIVATE_FIGURE_SENTINEL'
PUBLIC = 'Public prose </script><img src=x onerror=alert(1)>'


@pytest.fixture(name='game')
def game_fixture(tmp_path):
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No simulation')
  ):
    game = game_service.Game(tmp_path)
    game.world.emit('private_message', PRIVATE, ['Nell'])
    game.world.emit('speech', PUBLIC)
    try:
      yield game
    finally:
      game.close()


def export(game, audience='player'):
  return game.operations.dispatch(
      audience,
      {'operation': 'game.public_account', 'arguments': {'format': 'svg'}},
  )['result']


def assert_inert(svg):
  root = etree.fromstring(svg)
  assert root.tag == SVG + 'svg'
  assert root.attrib['role'] == 'img'
  assert root.attrib['aria-labelledby'] == 'figure-title figure-description'
  ids = {n.attrib['id'] for n in root.iter() if 'id' in n.attrib}
  for node in root.iter():
    assert node.tag.rsplit('}', 1)[-1] not in [
        'script',
        'foreignObject',
        'a',
        'form',
    ]
    for key, value in node.attrib.items():
      assert not key.lower().startswith('on')
      if key.rsplit('}', 1)[-1] == 'href':
        assert value.startswith('#') or value.startswith(
            'data:image/png;base64,'
        )
        if value.startswith('#'):
          assert value[1:] in ids
  assert PRIVATE not in svg
  assert PUBLIC not in svg
  return root


def test_projection_repeatability_provenance_and_no_effects(game):
  before = game.operations.snapshot('developer')
  artifact = export(game)
  assert artifact['media_type'] == 'image/svg+xml; charset=utf-8'
  root = assert_inert(artifact['content'])
  description = root.find(SVG + 'desc').text
  assert description.count('not resolved') == 9
  assert 'fixture backend' in description
  assert 'bellwether-public-account/v1' in description
  assert 'Generator: 6 fuel' in description
  assert 'Nell: 2 fuel, 1 spare parts' in description
  assert 'not predictions' in description
  assert artifact == export(game, 'developer') == export(game, 'role:spectator')
  assert game.operations.snapshot('developer') == before
  with pytest.raises(ops.OperationError):
    export(game, 'visitor')


def test_resolved_and_pending_cells_remain_distinct(game):
  game.world.resolve('Coordinator', 'allocate shelter and cold store')
  for _ in range(3):
    game.world.resolve('Coordinator', 'wait')
  root = assert_inert(export(game)['content'])
  text = root.find(SVG + 'desc').text
  assert 'Dusk: beacon unserved, shelter served, cold store served.' in text
  assert text.count('not resolved') == 6
  assert 'Generator: 4 fuel' in text and 'Used: 2 fuel' in text
  game.fixture = False
  game.phase = 'failed'
  text = assert_inert(export(game)['content']).find(SVG + 'desc').text
  assert 'live backend; interrupted' in text


def test_shared_cli_receives_the_same_projected_image(game):
  server = simulation_server.SimulationServer(
      port=0, operation_service=game.operations
  )
  server.start()
  try:
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'concordia.command_line_interface.concordia_session',
            '--url',
            f'http://127.0.0.1:{server.bound_port}',
            'call',
        ],
        input=json.dumps(
            {'operation': 'game.public_account', 'arguments': {'format': 'svg'}}
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)['result'] == export(game)
  finally:
    server.stop()


@pytest.mark.parametrize('width,height', [(360, 800), (800, 360)])
def test_player_download_and_offline_figure(game, tmp_path, width, height):
  pwlib = pytest.importorskip('playwright.sync_api')
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
            viewport={'width': width, 'height': height}, has_touch=True
        )
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.bound_port}')
        pwlib.expect(page.locator('#export-svg')).to_be_enabled()
        page.fill('#action', 'Draft remains unsent')
        before = game.operations.snapshot('developer')
        with page.expect_download() as saved:
          page.click('#export-svg')
        artifact = tmp_path / saved.value.suggested_filename
        saved.value.save_as(artifact)
        assert artifact.read_text() == export(game)['content']
        assert page.locator('#action').input_value() == 'Draft remains unsent'
        assert game.operations.snapshot('developer') == before
        assert not errors
        offline = browser.new_page(viewport={'width': width, 'height': height})
        network = []
        offline.on(
            'request',
            lambda request: network.append(request.url)
            if request.url.startswith(('http:', 'https:'))
            else None,
        )
        offline.route('http://**', lambda route: route.abort())
        offline.route('https://**', lambda route: route.abort())
        offline.goto(artifact.as_uri())
        assert offline.locator('svg[role=img]').count() == 1
        assert 'fixture backend' in offline.locator('desc').first.text_content()
        assert offline.evaluate('document.documentElement.scrollWidth') <= width
        assert not network
        # Wait for SVG use/glyph compositing, not only document load.
        offline.evaluate(
            '()=>new Promise(resolve=>requestAnimationFrame('
            '()=>requestAnimationFrame(resolve)))'
        )
        offline.locator('svg').screenshot(
            path=str(tmp_path / 'public-service-figure.png'), timeout=5000
        )
      finally:
        browser.close()
  finally:
    server.stop()

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

"""Public export/CLI/browser evidence; seeded records, never simulation runs."""

import json
import pathlib
import subprocess
import sys
from unittest import mock

from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import multiplayer
from concordia.utils import operation_service as ops
from concordia.utils import simulation_server
import pytest

PRIVATE = 'PRIVATE_EXPORT_SENTINEL'
LITERAL = 'Public "words"\n</script><img src=x onerror=alert(1)> & café 🌊'


@pytest.fixture(name='game')
def game_fixture(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation in export checks'),
  ):
    game = game_service.Game(tmp_path)
    game.world.emit('private_message', PRIVATE, ['Coordinator', 'Nell'])
    game.world.emit('speech', LITERAL, secret_extra=PRIVATE)
    try:
      yield game
    finally:
      game.close()


def call(game, audience='player', format_name='json'):
  return game.operations.dispatch(
      audience,
      {
          'operation': 'game.public_account',
          'arguments': {'format': format_name},
      },
  )['result']


def test_public_allowlist_no_identifiers_or_effects(game):
  before = game.operations.snapshot('developer')
  result = call(game)
  document = json.loads(result['content'])
  assert PRIVATE not in result['content']
  assert document['events'][-1]['text'] == LITERAL
  assert [x['number'] for x in document['events']] == list(
      range(1, len(document['events']) + 1)
  )
  assert set(document['events'][-1]) == {'number', 'watch', 'kind', 'text'}
  assert 'recipients' not in result['content']
  assert 'session_id' not in result['content']
  assert 'request_id' not in result['content']
  assert 'secret_extra' not in result['content']
  assert call(game, 'developer') == result
  assert call(game, 'role:spectator') == result
  assert game.operations.snapshot('developer') == before


@pytest.mark.parametrize(
    ('phase', 'expected'),
    [
        ('paused', 'in_progress'),
        ('running', 'in_progress'),
        ('failed', 'interrupted'),
        ('completed', 'completed'),
    ],
)
def test_status_does_not_claim_incomplete_as_complete(game, phase, expected):
  game.phase = phase
  data = json.loads(call(game)['content'])
  assert data['status'] == expected
  assert data['backend'] == 'fixture'
  game.fixture = False
  assert json.loads(call(game)['content'])['backend'] == 'live'


def test_empty_public_timeline_and_explicit_non_replay_limits(game):
  game.world.data['events'] = []
  content = call(game, format_name='html')['content']
  assert 'No public events recorded yet.' in content
  assert 'not a checkpoint' in content
  assert 'not anonymization' in content


def test_invalid_format_and_scope_fail_without_changes(game):
  before = game.operations.snapshot('developer')
  with pytest.raises(ops.OperationError, match='json, html or svg'):
    call(game, format_name='xml')
  with pytest.raises(ops.OperationError, match='unavailable'):
    call(game, audience='visitor')
  assert game.operations.snapshot('developer') == before


def test_multiplayer_public_bytes_equal_and_revocation_denies(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No run'),
  ):
    game = multiplayer.SharedGame(tmp_path, secure=False)
    try:
      game.world.emit('private_message', PRIVATE, ['Coordinator', 'Nell'])
      expected = call(game, 'developer')
      for role in ('Coordinator', 'Nell', 'spectator'):
        principal, _ = game.sessions.identify(None, create=True)
        game.sessions.request(principal, role, role)
        game.sessions.approve(principal)
        assert call(game, principal) == expected
        game.sessions.revoke(principal)
        with pytest.raises(ops.OperationError, match='unavailable'):
          call(game, principal)
    finally:
      game.close()


def test_html_escape_and_material_projection(game):
  game.world.data['services'] = [{
      'watch': 'Dusk',
      'facility': 'shelter',
      'served': False,
      'consequence': LITERAL,
      'private_debug': PRIVATE,
  }]
  game.world.data['epilogue'] = {
      'services_maintained': 0,
      'services_total': 9,
      'fuel_used': 0,
      'repair_completed': False,
      'dispute_settled': False,
      'commitments': [{'private_text': PRIVATE}],
  }
  artifact = call(game, format_name='html')
  assert '<script' not in artifact['content']
  assert '<img' not in artifact['content']
  assert '&lt;img' in artifact['content']
  assert PRIVATE not in artifact['content']
  account = json.loads(call(game)['content'])
  assert account['dawn']['dispute_settled'] is False
  assert account['accounting']['services'][0]['consequence'] == LITERAL


def test_chromium_download_cli_parity_and_offline_inertness(game, tmp_path):
  pwlib = pytest.importorskip('playwright.sync_api')
  game.server.start()
  player = simulation_server.SimulationServer(
      port=0,
      operation_service=game.operations,
      audience='player',
      html_content=pathlib.Path(__file__)
      .with_name('player.html')
      .read_text(encoding='utf-8'),
  )
  player.start()
  before = game.operations.snapshot('developer')
  try:
    with pwlib.sync_playwright() as pw:
      browser = pw.chromium.launch()
      try:
        page = browser.new_page(viewport={'width': 360, 'height': 800})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{player.bound_port}')
        pwlib.expect(page.locator('#export-html')).to_be_enabled()
        page.locator('#action').fill('private draft remains in this browser')
        with page.expect_download() as json_download:
          page.click('#export-json')
        json_path = tmp_path / 'public.json'
        json_download.value.save_as(json_path)
        with page.expect_download() as html_download:
          page.click('#export-html')
        html_path = tmp_path / 'public.html'
        html_download.value.save_as(html_path)
        assert page.locator('#action').input_value().startswith('private draft')
        page.route(
            '**/api/dispatch',
            lambda route: route.fulfill(
                status=409,
                content_type='application/json',
                body=json.dumps({'error': {'message': 'Export unavailable'}}),
            ),
        )
        page.click('#export-json')
        pwlib.expect(page.locator('#export-status')).to_contain_text(
            'Export unavailable. Your draft is kept.'
        )
        assert page.locator('#action').input_value().startswith('private draft')
        page.unroute('**/api/dispatch')
        assert game.operations.snapshot('developer') == before
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'concordia.command_line_interface.concordia_session',
                '--url',
                f'http://127.0.0.1:{game.server.bound_port}',
                'call',
            ],
            input=json.dumps({
                'operation': 'game.public_account',
                'arguments': {'format': 'json'},
            }),
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout)['result']['content'] == (
            json_path.read_text(encoding='utf-8')
        )
        assert 'private draft' not in html_path.read_text(encoding='utf-8')
        assert PRIVATE not in html_path.read_text(encoding='utf-8')
        assert PRIVATE not in json_path.read_text(encoding='utf-8')
        page.reload()
        pwlib.expect(page.locator('#export-html')).to_be_enabled()
        assert page.locator('#action').input_value().startswith('private draft')
        resource_requests = []
        offline = browser.new_page(viewport={'width': 360, 'height': 800})
        offline.on('pageerror', lambda e: errors.append(str(e)))
        offline.on('request', lambda r: resource_requests.append(r.url))
        offline.context.set_offline(True)
        offline.goto(html_path.as_uri())
        assert offline.locator('script,img,iframe,a,form').count() == 0
        assert LITERAL in offline.locator('body').inner_text()
        assert offline.evaluate('innerWidth') == 360
        assert offline.evaluate('document.documentElement.scrollWidth') <= 360
        offline.screenshot(
            path=str(tmp_path / 'public-account-mobile.png'), full_page=True
        )
        assert resource_requests == [html_path.as_uri()]
        assert not errors
      finally:
        browser.close()
  finally:
    player.stop()

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

"""Chromium transport checks on a synthetic input, not simulation execution."""

import json
import pathlib
import subprocess
import sys
import threading
from unittest import mock

from concordia.examples.bellwether import game_service
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
from concordia.utils import simulation_server
import pytest

playwright = pytest.importorskip('playwright.sync_api')


def test_full_game_inspection_clarification_and_retry(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('This test never launches a simulation'),
  ):
    game = game_service.Game(tmp_path)
    player = simulation_server.SimulationServer(
        port=0,
        operation_service=game.operations,
        audience='player',
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
    )
    results = []
    request = human_input.HumanInputRequest(
        request_id='synthetic-input',
        entity_name='Coordinator',
        action_spec=entity_lib.free_action_spec(
            call_to_action='What do you do?'
        ),
        contexts={},
        context='Only the coordinator’s available context.',
    )
    reader = threading.Thread(
        target=lambda: results.append(game.inbox(request))
    )
    player.start()
    reader.start()
    try:
      with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 390, 'height': 844})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        url = f'http://127.0.0.1:{player.bound_port}'
        page.goto(url)
        playwright.expect(page.locator('#act')).to_be_enabled()
        playwright.expect(page.locator('#watch')).to_have_text(
            'Watch 1 · Early evening · 4 choices remain'
        )
        initial = game.operations.snapshot('player')
        for button in page.locator('#map button').all():
          button.click()
        assert game.operations.snapshot('player') == initial
        page.click('#all-cast')
        assert page.locator('#cast .card').count() == 4
        page.locator('#cast .card').filter(has_text='Nell').get_by_text(
            'Private', exact=True
        ).click()
        assert page.locator('#action').input_value() == 'message Nell: '
        page.locator('#action').fill('ambiguous allocation')
        page.reload()
        playwright.expect(page.locator('#act')).to_be_enabled()
        assert page.locator('#action').input_value() == 'ambiguous allocation'
        page.click('#act')
        playwright.expect(page.locator('#error')).to_contain_text(
            'Nothing changed'
        )
        assert game.operations.snapshot('player') == initial
        assert not results
        words = 'message Nell: "A proposal" 🌊 & <img src=x onerror=alert(1)>'
        page.locator('#action').fill(words)
        with page.expect_request('**/api/dispatch') as sent:
          page.click('#act')
        reader.join(timeout=2)
        assert results == [words]
        body = sent.value.post_data_json
        first = game.operations.dispatch('player', body)
        # Noninteractive attached CLI retries the SAME effect/key.
        retried = subprocess.run(
            [
                sys.executable,
                '-m',
                'concordia.command_line_interface.concordia_session',
                '--url',
                url,
                'call',
            ],
            input=json.dumps(body),
            text=True,
            capture_output=True,
            check=True,
        )
        assert json.loads(retried.stdout) == first
        assert results == [words]
        assert page.locator('img').count() == 0
        assert 'PRIVATE_NELL' not in page.content()
        page.route('**/api/events', lambda route: route.abort('failed'))
        page.reload()
        playwright.expect(page.locator('#connection')).to_contain_text(
            'Reconnecting'
        )
        playwright.expect(page.locator('#begin')).to_be_disabled()
        page.unroute('**/api/events')
        playwright.expect(page.locator('#connection')).to_contain_text(
            'Connected', timeout=15000
        )
        assert page.evaluate(
            'document.documentElement.scrollWidth <= innerWidth'
        )
        assert not errors
        browser.close()
    finally:
      game.close()
      player.stop()
      reader.join(timeout=2)

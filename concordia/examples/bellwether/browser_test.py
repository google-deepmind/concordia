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

"""Actual Chromium/CLI parity on real components, guarded against simulation."""

import json
import pathlib
import subprocess
import sys
from unittest import mock
import urllib.request

from concordia.examples.bellwether import service
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import pytest

playwright = pytest.importorskip('playwright.sync_api')


def test_gui_cli_semantics_reconnect_and_player_privacy(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No run in this test'),
  ):
    gui = service.Bellwether(tmp_path / 'gui')
    cli = service.Bellwether(tmp_path / 'cli')
    gui.server.set_html_content(
        visual_interface.visualize_operations_to_html(gui.config)
    )
    public = simulation_server.SimulationServer(
        port=0,
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
        operation_service=gui.operations,
        audience='player',
    )
    gui.server.start()
    cli.server.start()
    public.start()
    try:
      with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        player = browser.new_page(viewport={'width': 390, 'height': 844})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        player.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{gui.server.bound_port}')
        player.goto(f'http://127.0.0.1:{public.bound_port}')
        playwright.expect(page.locator('#op-status')).to_contain_text(
            'Connected'
        )
        playwright.expect(player.locator('#connection')).to_contain_text(
            'Connected'
        )
        page.locator('#op-snapshot summary').click()
        page.select_option('#op-preview-field', 'target')
        page.select_option('#op-name', 'component.edit')
        value = (
            'PRIVATE_EDIT: "quoted"\n</script><img src=x onerror=alert(1)> &'
            ' mémoire 🌧️'
        )
        page.locator('[data-key=value]').fill(value)
        with page.expect_request('**/api/dispatch') as captured:
          page.click('#op-submit')
        playwright.expect(page.locator('#op-preview-status')).to_contain_text(
            'newer'
        )
        page.click('#op-preview-refresh')
        playwright.expect(page.locator('#op-state')).to_contain_text(
            'PRIVATE_EDIT'
        )
        body = captured.value.post_data_json
        body['references'] = cli.operations.references
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'concordia.command_line_interface.concordia_session',
                '--url',
                f'http://127.0.0.1:{cli.server.bound_port}',
                'call',
            ],
            input=json.dumps(body),
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert gui.developer_view() == cli.developer_view()
        assert page.locator('#operations-panel img').count() == 0
        assert 'PRIVATE_EDIT' not in player.content()
        before = gui.operations.snapshot('developer')
        # A second actor changes the service after a GUI draft starts.
        page.locator('[data-key=value]').fill('keep this stale draft')
        changed = {
            **body,
            'references': gui.operations.references,
            'revision': before['revision'],
            'retry_key': 'second',
            'arguments': {'value': 'new live text'},
        }
        gui.operations.dispatch('developer', changed)
        playwright.expect(page.locator('#op-preview-status')).to_contain_text(
            'newer'
        )
        page.click('#op-preview-refresh')
        playwright.expect(page.locator('#op-state')).to_contain_text(
            'new live text'
        )
        page.click('#op-submit')
        playwright.expect(page.locator('#op-error')).to_contain_text(
            'State changed'
        )
        assert (
            page.locator('[data-key=value]').input_value()
            == 'keep this stale draft'
        )
        page.reload()
        playwright.expect(page.locator('#op-status')).to_contain_text(
            'Connected'
        )
        page.locator('#op-snapshot summary').click()
        page.select_option('#op-preview-field', 'target')
        playwright.expect(page.locator('#op-state')).to_contain_text(
            'new live text'
        )
        # Genuine disconnect/reconnect of the operation stream, not mock SSE.
        player.route('**/api/events', lambda route: route.abort('failed'))
        player.reload()
        playwright.expect(player.locator('#connection')).to_contain_text(
            'Reconnecting'
        )
        player.unroute('**/api/events')
        playwright.expect(player.locator('#connection')).to_contain_text(
            'Connected', timeout=15000
        )
        assert player.evaluate(
            'document.documentElement.scrollWidth <= innerWidth'
        )
        for endpoint in ('/api/state', '/api/operations', '/'):
          with urllib.request.urlopen(
              f'http://127.0.0.1:{public.bound_port}' + endpoint
          ) as response:
            assert 'PRIVATE_' not in response.read().decode()
        # The one-turn service shares this page but has no public exporter.
        gui.phase = 'failed'
        gui._failure = 'FixtureRuntimeFailure'  # pylint: disable=protected-access
        gui._finish()  # pylint: disable=protected-access
        gui.operations.publish({'kind': 'test.failed_fixture'})
        playwright.expect(player.locator('#connection')).to_contain_text(
            'Night stopped'
        )
        playwright.expect(player.locator('#run-status')).to_be_visible()
        assert (
            'public account' not in player.locator('#run-status').inner_text()
        )
        playwright.expect(player.locator('#public-account')).to_be_hidden()
        assert 'FixtureRuntimeFailure' not in player.content()
        assert not errors
        browser.close()
    finally:
      gui.close()
      cli.close()
      public.stop()

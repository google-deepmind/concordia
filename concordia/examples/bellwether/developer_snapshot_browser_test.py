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

"""Bounded developer rendering on real server state; no simulation or model."""

import json
from unittest import mock

from concordia.examples.bellwether import service
from concordia.prefabs.simulation import generic
from concordia.utils import visual_interface
import pytest

pwlib = pytest.importorskip('playwright.sync_api')


def test_large_received_snapshot_is_lazy_bounded_and_downloadable(tmp_path):
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No simulation')
  ):
    host = service.Bellwether(tmp_path / 'host')
    original_view = host.developer_view
    payload = {
        'fixture_version': 0,
        'fixture_big_integer': 9223372036854775783,
        'fixture_large_history': (
            ['literal </script><img onerror=alert(1)> & 🌊' * 30] * 400
        ),
    }
    host.operations.set_view(
        'developer', lambda: {**payload, **original_view()}
    )
    host.server.set_html_content(
        visual_interface.visualize_operations_to_html(host.config)
    )
    host.server.start()
    try:
      with pwlib.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
          page = browser.new_page(viewport={'width': 1200, 'height': 850})
          errors = []
          page.on('pageerror', lambda e: errors.append(str(e)))
          page.add_init_script("""
            window.fullStringifies=0;const stringify=JSON.stringify;
            JSON.stringify=function(value,...args){
              if(value?.result?.fixture_large_history?.length>100)
                window.fullStringifies++;
              return stringify.call(JSON,value,...args);
            };
          """)
          page.goto(f'http://127.0.0.1:{host.server.bound_port}')
          pwlib.expect(page.locator('#op-status')).to_contain_text('Connected')
          # Baseline eagerly inserted a large text block on every SSE.
          assert len(page.locator('#op-state').inner_text()) == 0
          assert page.evaluate('window.fullStringifies') == 0
          page.select_option('#op-name', 'component.edit')
          page.locator('[data-key=value]').fill(
              'Keep this unsent developer draft'
          )
          page.locator('[data-key=value]').evaluate(
              'e=>e.setSelectionRange(5,9)'
          )
          initial = host.operations.snapshot('developer')
          page.locator('#op-snapshot summary').click()
          pwlib.expect(page.locator('#op-preview-status')).to_contain_text(
              'abbreviated'
          )
          preview = page.locator('#op-state').inner_text()
          assert len(preview) < 30000
          assert '"fixture_version": 0' in preview
          assert '[preview:' in preview
          assert 'number outside safe integer range' in preview
          assert page.locator('#operations-panel img').count() == 0
          # Refresh is explicit; inbound SSE must not rewrite text being read.
          payload['fixture_version'] = 1
          host.operations.publish({'kind': 'fixture.updated'})
          pwlib.expect(page.locator('#op-preview-status')).to_contain_text(
              'newer'
          )
          assert page.locator('#op-state').inner_text() == preview
          assert (
              page.locator('[data-key=value]').input_value()
              == 'Keep this unsent developer draft'
          )
          assert page.locator('[data-key=value]').evaluate(
              'e=>[e.selectionStart,e.selectionEnd]'
          ) == [5, 9]
          page.click('#op-preview-refresh')
          pwlib.expect(page.locator('#op-state')).to_contain_text(
              '"fixture_version": 1'
          )
          assert page.evaluate('window.fullStringifies') == 0
          page.select_option('#op-preview-field', 'fixture_large_history')
          pwlib.expect(page.locator('#op-state')).to_contain_text('[preview:')
          assert len(page.locator('#op-state').inner_text()) < 30000
          assert page.locator('#operations-panel img').count() == 0
          with page.expect_download() as download:
            page.click('#op-snapshot-download')
          path = download.value.path()
          downloaded = json.loads(path.read_text())
          current = json.loads(
              json.dumps(host.operations.snapshot('developer'))
          )

          assert downloaded == current
          assert (
              downloaded['result']['fixture_big_integer'] == 9223372036854775783
          )
          assert page.evaluate('window.fullStringifies') == 0
          assert (
              downloaded['result']['fixture_large_history']
              == payload['fixture_large_history']
          )
          assert downloaded['revision'] > initial['revision']
          # Inspection/download did not silently send or repair a stale edit.
          page.click('#op-submit')
          pwlib.expect(page.locator('#op-error')).to_contain_text(
              'State changed'
          )
          assert (
              page.locator('[data-key=value]').input_value()
              == 'Keep this unsent developer draft'
          )
          page.screenshot(
              path=str(tmp_path / 'bounded-developer.png'), timeout=5000
          )
          page.reload()
          pwlib.expect(page.locator('#op-status')).to_contain_text('Connected')
          assert page.locator('#op-state').inner_text() == ''
          page.route('**/api/events', lambda route: route.abort('failed'))
          page.reload()
          pwlib.expect(page.locator('#op-status')).to_contain_text(
              'reconnecting'
          )
          pwlib.expect(page.locator('#op-snapshot-download')).to_be_disabled()
          page.unroute('**/api/events')
          pwlib.expect(page.locator('#op-status')).to_contain_text(
              'Connected', timeout=15000
          )
          pwlib.expect(page.locator('#op-snapshot-download')).to_be_enabled()
          assert page.locator('#op-state').inner_text() == ''
          assert not errors
          (tmp_path / 'browser-evidence.json').write_text(
              json.dumps(
                  {
                      'automated': True,
                      'preview_characters': len(preview),
                      'download_bytes': path.stat().st_size,
                      'whole_snapshot_stringifications': page.evaluate(
                          'window.fullStringifies'
                      ),
                      'exact_download_and_large_integer': True,
                      'errors': errors,
                      'simulation_steps': 0,
                      'provider_calls': 0,
                      'physical_device': False,
                  },
                  indent=2,
              )
          )
        finally:
          browser.close()
    finally:
      host.close()

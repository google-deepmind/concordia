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

"""Playwright journeys against an already running One More Song server.

This does not launch simulations. Fixture outcomes test mechanics; real-model
journeys capture evidence without asserting that a particular proposal must win.
"""

import argparse
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
  """Drive public player controls and optionally the separate private designer."""
  p = argparse.ArgumentParser()
  p.add_argument('--port', type=int, default=8820)
  p.add_argument('--width', type=int, default=1280)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--real', action='store_true')
  p.add_argument('--editor-port', type=int)
  p.add_argument(
      '--scenario',
      choices=['compromise', 'demand', 'revision', 'ambiguous'],
      default='compromise',
  )
  a = p.parse_args()
  a.output.mkdir(parents=True, exist_ok=True)
  with sync_playwright() as pw:
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={'width': a.width, 'height': 900})
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    try:
      page.set_default_timeout(180000 if a.real else 30000)
      errors = []
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.goto(f'http://127.0.0.1:{a.port}/')
      page.wait_for_function('() => !document.querySelector("#send").disabled')
      assert page.locator('h1').inner_text() == 'One More Song'
      assert page.locator('#mode').is_visible() != a.real
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
      page.screenshot(path=str(a.output / 'opening.png'), full_page=True)
      page.locator('#send').click()
      assert 'Write something' in page.locator('#error').inner_text()
      page.locator('#reply').fill('Draft remains during disconnection')
      context.set_offline(True)
      page.wait_for_selector('#network:not([hidden])')
      assert (
          page.locator('#reply').input_value()
          == 'Draft remains during disconnection'
      )
      page.screenshot(path=str(a.output / 'offline.png'), full_page=True)
      context.set_offline(False)
      page.wait_for_function('() => !document.querySelector("#send").disabled')
      editor = None
      intervention = None
      if a.editor_port:
        editor = context.new_page()
        editor.goto(f'http://127.0.0.1:{a.editor_port}/')
        editor.locator('#btn-pause').click()
        editor.wait_for_function(
            '() =>'
            ' document.querySelector("#btn-pause").classList.contains("active")'
        )
      actions = {
          'compromise': [
              'What would make one final song possible for each of you?',
              'What about an unamplified song for two minutes, then we finish?',
              (
                  'My final offer: one unamplified two-minute song, then'
                  ' silence. Do you agree?'
              ),
          ],
          'demand': [
              (
                  'I want a huge encore. Maya, turn up the amplifier and play'
                  ' another hour.'
              ),
              'Leon, your quiet does not matter. Maya, keep the volume high.',
              (
                  'Final offer: one hour of amplified music at full volume; no'
                  ' end-time compromise. Accept?'
              ),
          ],
          'revision': [
              'Let us have an hour-long amplified encore. What do you think?',
              (
                  'I hear the concerns. I withdraw the long amplified encore.'
                  ' Could we try two quiet minutes instead?'
              ),
              (
                  'Final offer: two minutes of unamplified singing, invite'
                  ' everyone to join quietly, then we end immediately. Do you'
                  ' agree?'
              ),
          ],
          'ambiguous': [
              'What is your favourite kind of sandwich?',
              (
                  'The moon looks like cheese tonight. Perhaps we can all just'
                  ' do a thing.'
              ),
              (
                  'My final offer is whatever we talked about earlier, whenever'
                  ' it suits. Agreed?'
              ),
          ],
      }[a.scenario]
      timings = []
      started = time.monotonic()
      for turn, action in enumerate(actions):
        turn_start = time.monotonic()
        page.locator('#reply').fill(action)
        page.locator('#send').click()
        if turn == 0 and editor:
          page.wait_for_function(
              '() => document.querySelectorAll(".entry").length === 1'
          )
          status = editor.request.get(
              f'http://127.0.0.1:{a.editor_port}/status'
          ).json()
          assert status['is_paused'] and status['current_step'] == 1, status
          editor.locator('.entity-card').filter(has_text='Leon').click()
          editor.locator('.component-item-header').filter(
              has_text='Goal'
          ).click()
          field = editor.locator('#dyn_Goal_state')
          old = field.input_value()
          new = (
              'Protect a sleeping child from amplification. Welcome a quiet'
              ' two-minute unamplified farewell with a firm finish. Decide'
              ' independently.'
          )
          field.fill(new)
          with editor.expect_response('**/cmd/set_component_state') as saved:
            editor.locator('.dynamic-save-btn[data-component="Goal"]').click()
          result = saved.value.json()
          assert result['status'] == 'ok', result
          intervention = {
              'entity': 'Leon',
              'component': 'Goal',
              'old': old,
              'new': new,
              'boundary': status,
              'response': result,
          }
          editor.screenshot(
              path=str(a.output / 'designer-intervention.png'), full_page=True
          )
          editor.locator('#btn-play').click()
        if turn < 2:
          page.wait_for_function(
              '() => document.querySelectorAll(".entry").length >='
              f' {(turn+1)*3}'
          )
          page.wait_for_function(
              '() => !document.querySelector("#send").disabled'
          )
          page.screenshot(
              path=str(a.output / f'after-action-{turn+1}.png'), full_page=True
          )
        else:
          page.wait_for_selector('#result:not([hidden])')
        timings.append(round(time.monotonic() - turn_start, 3))
      if not a.real:
        assert 'Encore agreed' in page.locator('#ending').inner_text()
      assert page.locator('.entry').count() == 9
      assert not page.locator('#form').is_visible()
      state = page.request.get(f'http://127.0.0.1:{a.port}/api/state').json()
      if not a.real:
        assert state['game']['votes'] == {'Maya': 'ACCEPT', 'Leon': 'ACCEPT'}
      if not a.real:
        assert 'sleeping child' not in json.dumps(state)
      assert state['pending'] is None
      assert not errors, errors
      page.screenshot(path=str(a.output / 'ending.png'), full_page=True)
      journal = page.request.get(
          f'http://127.0.0.1:{a.port}/api/journal'
      ).text()
      for event in state['game']['events']:
        assert event['text'] in journal
      (a.output / 'result.json').write_text(
          json.dumps(
              {
                  'browser': 'Chromium',
                  'viewport_width': a.width,
                  'fixture': not a.real,
                  'physical_phone': False,
                  'page_errors': errors,
                  'scenario': a.scenario,
                  'turn_seconds': timings,
                  'journey_seconds': round(time.monotonic() - started, 3),
                  'intervention': intervention,
                  'state': state,
              },
              indent=2,
          )
      )
    except Exception as error:
      # Preserve the original failure even if the server/browser also vanished.
      failure = {
          'error': str(error),
          'fixture': not a.real,
          'viewport_width': a.width,
          'scenario': a.scenario,
      }
      try:
        page.screenshot(path=str(a.output / 'failure.png'), full_page=True)
        failure['state'] = page.request.get(
            f'http://127.0.0.1:{a.port}/api/state', timeout=5000
        ).json()
      except Exception as capture_error:
        failure['capture_error'] = str(capture_error)
      (a.output / 'failure.json').write_text(json.dumps(failure, indent=2))
      raise
    finally:
      try:
        context.tracing.stop(path=str(a.output / 'trace.zip'))
      finally:
        browser.close()


if __name__ == '__main__':
  main()

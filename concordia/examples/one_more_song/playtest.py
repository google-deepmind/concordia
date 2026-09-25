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
import importlib
import json
import time
from pathlib import Path


def main():
  """Drive public player controls and optionally the separate private designer."""
  p = argparse.ArgumentParser()
  p.add_argument('--port', type=int, default=8820)
  p.add_argument('--width', type=int, default=1280)
  p.add_argument(
      '--browser', choices=['chromium', 'webkit', 'firefox'], default='chromium'
  )
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--real', action='store_true')
  p.add_argument(
      '--timeout-seconds',
      type=float,
      help=(
          'Per browser wait; defaults to 180 for live models and 30 for'
          ' fixtures.'
      ),
  )
  p.add_argument('--editor-port', type=int)
  p.add_argument(
      '--network-faults',
      action='store_true',
      help='Exercise stalled polling and an accepted POST with a lost reply.',
  )
  p.add_argument(
      '--scenario',
      choices=['compromise', 'demand', 'revision', 'ambiguous'],
      default='compromise',
  )
  p.add_argument(
      '--actions-file',
      type=Path,
      help='JSON array of three player utterances; overrides --scenario.',
  )
  a = p.parse_args()
  timeout_seconds = (
      a.timeout_seconds
      if a.timeout_seconds is not None
      else (180 if a.real else 30)
  )
  if not 0 < timeout_seconds < float('inf'):
    p.error('--timeout-seconds must be a positive finite number.')
  custom_actions = None
  if a.actions_file:
    try:
      custom_actions = json.loads(a.actions_file.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
      p.error(f'Cannot read --actions-file: {error}')
    if not (
        isinstance(custom_actions, list)
        and len(custom_actions) == 3
        and all(
            isinstance(action, str) and action.strip() and len(action) <= 8000
            for action in custom_actions
        )
    ):
      p.error(
          '--actions-file must contain exactly three nonempty strings, each at'
          ' most 8000 characters.'
      )
  if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
    p.error(
        '--output must be a new or empty directory; keep earlier journey'
        ' evidence for comparison.'
    )
  a.output.mkdir(parents=True, exist_ok=True)
  try:
    playwright = importlib.import_module('playwright.sync_api')
  except ModuleNotFoundError as error:
    if error.name not in ('playwright', 'playwright.sync_api'):
      raise
    p.error(
        'Install the optional browser dependency with pip install playwright, '
        'then run playwright install chromium.'
    )
  with playwright.sync_playwright() as pw:
    browser = getattr(pw, a.browser).launch()
    context = browser.new_context(viewport={'width': a.width, 'height': 900})
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    browser_started = time.monotonic()
    try:
      page.set_default_timeout(timeout_seconds * 1000)
      errors = []
      page.on('pageerror', lambda e: errors.append(str(e)))
      page.goto(f'http://127.0.0.1:{a.port}/')
      page.wait_for_function('() => !document.querySelector("#send").disabled')
      assert page.locator('h1').inner_text() == 'One More Song'
      assert page.locator('#mode').is_visible() != a.real
      assert ('AI characters' if a.real else 'scripted replies') in (
          page.locator('#reply-kind').inner_text()
      )
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
      page.screenshot(path=str(a.output / 'opening.png'), full_page=True)
      # Exercise the discoverable path to the controls, not only direct locators.
      page.locator('.start-link').click()
      assert page.locator('#reply').evaluate(
          '(node) => node.getBoundingClientRect().top < innerHeight'
      )
      assert page.locator('#conversation').get_attribute('role') == 'log'
      network_evidence = {}
      if a.network_faults:
        # Hold the actual request, rather than mocking a disconnected state.
        held = []
        page.route('**/api/state', lambda route: held.append(route))
        page.locator('#reply').fill('Keep this draft through a stalled request')
        poll_start = time.monotonic()
        page.wait_for_selector('#network:not([hidden])', timeout=20000)
        assert page.locator('#send').is_disabled()
        assert page.locator('#reply').input_value() == (
            'Keep this draft through a stalled request'
        )
        network_evidence['stalled_poll_detected_seconds'] = round(
            time.monotonic() - poll_start, 3
        )
        for route in held:
          route.abort()
        page.unroute('**/api/state')
        page.wait_for_function(
            '() => !document.querySelector("#send").disabled'
        )
        page.locator('#reply').fill('')
      page.locator('#send').click()
      assert 'Write something' in page.locator('#error').inner_text()
      page.locator('#reply').fill('Draft remains during disconnection')
      page.reload()
      page.wait_for_function('() => !document.querySelector("#send").disabled')
      assert (
          page.locator('#reply').input_value()
          == 'Draft remains during disconnection'
      )
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
        # The private designer is operated on the host, separately from the
        # phone-sized player page. Do not claim mobile designer coverage.
        editor.set_viewport_size({'width': 1280, 'height': 900})
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
      if custom_actions is not None:
        actions = custom_actions
      timings = []
      started = time.monotonic()
      for turn, action in enumerate(actions):
        page.locator('#reply').scroll_into_view_if_needed()
        assert page.locator('#phase').evaluate(
            '(node) => {const r=node.getBoundingClientRect();'
            'return r.top >= 0 && r.bottom <= innerHeight;}'
        ), 'Turn/wait instructions must stay visible beside reply controls'
        accepted_posts = []
        if turn == 0 and a.network_faults:

          def lose_acknowledgment(route):
            response = route.fetch()
            assert response.ok
            accepted_posts.append(route)

          page.route('**/api/action', lose_acknowledgment)
        turn_start = time.monotonic()
        page.locator('#reply').fill(action)
        page.locator('#send').click()
        if turn == 0 and a.network_faults:
          page.wait_for_function(
              '() => document.querySelector("#error").textContent.includes('
              '"Could not confirm submission")',
              timeout=22000,
          )
          assert page.locator('#reply').input_value() == action
          assert len(accepted_posts) == 1
          for route in accepted_posts:
            route.abort()
          page.unroute('**/api/action')
          # A user retry must recheck the original ID, not consume turn two.
          page.reload()
          page.get_by_role('button', name='Check last send', exact=True).click()
          page.wait_for_function(
              '() => document.querySelector("#error").hidden'
          )
          assert page.locator('#reply').input_value() == ''
          state_after_timeout = page.request.get(
              f'http://127.0.0.1:{a.port}/api/state'
          ).json()
          assert (
              sum(
                  event['actor'] == 'You'
                  for event in state_after_timeout['game']['events']
              )
              == 1
          )
          network_evidence['lost_acknowledgment'] = {
              'draft_preserved': True,
              'accepted_posts': len(accepted_posts),
              'automatic_retries': 0,
              'manual_recheck_after_reload_did_not_consume_turn': True,
          }
        if turn == 0 and editor:
          page.wait_for_function(
              '() => document.querySelectorAll(".entry").length === 1'
          )
          status = editor.request.get(
              f'http://127.0.0.1:{a.editor_port}/status'
          ).json()
          assert status['is_paused'] and status['current_step'] == 1, status
          editor.locator('.entity-card').filter(has_text='Maya').click()
          editor.locator('.component-item-header').filter(
              has_text='Goal'
          ).click()
          field = editor.locator('#dyn_Goal_state')
          old = field.input_value()
          new = (
              'Protect your voice for tomorrow. You strongly prefer not to'
              ' sing any more tonight, even quietly. Find a warm, inclusive'
              ' farewell without singing, such as a silent wave or written'
              ' thanks. Decide independently; do not agree automatically.'
          )
          field.fill(new)
          with editor.expect_response('**/cmd/set_component_state') as saved:
            editor.locator('.dynamic-save-btn[data-component="Goal"]').click()
          result = saved.value.json()
          assert result['status'] == 'ok', result
          intervention = {
              'entity': 'Maya',
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
              f' {(turn+1)*3} || document.querySelector("#form").hidden'
          )
          assert not page.locator('#form').is_hidden(), page.locator(
              '#status'
          ).inner_text()
          page.wait_for_function(
              '() => !document.querySelector("#send").disabled'
          )
          page.locator('#latest-link').click()
          assert page.evaluate(
              '() => document.activeElement.classList.contains("entry")'
          ), 'Latest-replies link must reach and focus the recorded dialogue'
          page.screenshot(
              path=str(a.output / f'after-action-{turn+1}.png'), full_page=True
          )
        else:
          page.wait_for_function('() => document.querySelector("#form").hidden')
          assert page.locator('#result').is_visible(), page.locator(
              '#status'
          ).inner_text()
        timings.append(round(time.monotonic() - turn_start, 3))
      if not a.real:
        assert 'Agreement reached' in page.locator('#ending').inner_text()
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
      # A player may copy their final offer while the completed page still polls.
      # Unchanged state must not replace the selected text node on every poll.
      selected = page.locator('#final-offer').inner_text()
      page.locator('#final-offer').evaluate(
          '(node) => {const range=document.createRange();'
          'range.selectNodeContents(node);const'
          ' selection=window.getSelection();'
          'selection.removeAllRanges();selection.addRange(range);}'
      )
      page.wait_for_timeout(
          1400
      )  # More than two ordinary 600ms poll intervals.
      assert page.evaluate('window.getSelection().toString()') == selected
      page.evaluate('window.getSelection().removeAllRanges()')
      # Exercise the guest's actual download control, not only the HTTP route.
      with page.expect_download() as download_event:
        page.locator('#journal a').click()
      download = download_event.value
      assert download.failure() is None
      assert download.suggested_filename == 'one-more-song.txt'
      journal_path = a.output / 'conversation.txt'
      download.save_as(journal_path)
      journal = journal_path.read_text(encoding='utf-8')
      assert (
          'AI-generated characters:' if a.real else 'SCRIPTED UI PREVIEW'
      ) in journal
      for event in state['game']['events']:
        assert event['text'] in journal
      (a.output / 'result.json').write_text(
          json.dumps(
              {
                  'browser': a.browser,
                  'viewport_width': a.width,
                  'timeout_seconds': timeout_seconds,
                  'fixture': not a.real,
                  'physical_phone': False,
                  'page_errors': errors,
                  'download_filename': download.suggested_filename,
                  'network_faults': network_evidence,
                  'scenario': (
                      'custom' if custom_actions is not None else a.scenario
                  ),
                  'player_actions': actions,
                  'turn_seconds': timings,
                  'journey_seconds': round(time.monotonic() - started, 3),
                  'intervention': intervention,
                  'designer_viewport_width': 1280 if editor else None,
                  'state': state,
              },
              indent=2,
          )
      )
    except Exception as error:
      # Preserve the original failure even if the server/browser also vanished.
      failure = {
          'error': str(error),
          'browser': a.browser,
          'timeout_seconds': timeout_seconds,
          'browser_elapsed_seconds': round(
              time.monotonic() - browser_started, 3
          ),
          'fixture': not a.real,
          'viewport_width': a.width,
          'scenario': 'custom' if custom_actions is not None else a.scenario,
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

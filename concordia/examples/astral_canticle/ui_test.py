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

"""Browser integration against the real adapter with a fake input request.

These tests do not start a Concordia simulation or contact a model. Install the
optional example test dependencies and run `python -m playwright install chromium`.
"""

import concurrent.futures
import socket
import threading
import time
from unittest import mock

from concordia.examples.astral_canticle import adventure
from concordia.examples.astral_canticle import human_io
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('uvicorn')
pytest.importorskip('playwright.sync_api')
web = pytest.importorskip('concordia.examples.astral_canticle.web')
playwright = pytest.importorskip('playwright.sync_api')
expect = playwright.expect
sync_playwright = playwright.sync_playwright
uvicorn = pytest.importorskip('uvicorn')


@pytest.fixture
def serve():
  resources = []

  def start(
      role='player', output_type=entity_lib.OutputType.FREE, human_request=None
  ):
    session = human_io.HumanSession(role=role)
    pool = concurrent.futures.ThreadPoolExecutor()
    result = pool.submit(
        session,
        human_request
        or human_input.HumanInputRequest(
            request_id='ui-request',
            entity_name='Ilyra Venn',
            action_spec=entity_lib.ActionSpec(
                call_to_action='What do you do?', output_type=output_type
            ),
            context='My context:\nA silver door.\n<script>bad()</script>',
            contexts={
                '__observation__': (
                    '[observation] You stand in the Star-Loom Chamber. A brass'
                    ' tuning fork hangs beside a cracked resonance cradle. The'
                    ' west exit leads to the Reliquary Nave.'
                    ' <script>bad()</script>'
                )
            },
        ),
    )
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(
        uvicorn.Config(web.create_app(session), log_level='error')
    )
    thread = threading.Thread(
        target=server.run, kwargs={'sockets': [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
      time.sleep(0.01)
    assert server.started
    resources.append((server, thread, session, pool, sock))
    return f'http://127.0.0.1:{sock.getsockname()[1]}/', session, result

  yield start
  for server, thread, session, pool, sock in resources:
    session.finish('Test complete')
    server.should_exit = True
    thread.join(timeout=5)
    pool.shutdown(wait=True)
    sock.close()


def test_mobile_draft_reload_offline_recovery_and_single_submit(
    serve, tmp_path
):
  url, session, result = serve()
  with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True
    )
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    expect(page.locator('#connection')).to_have_text('CONNECTED')
    expect(page.locator('.passage')).to_contain_text('<script>bad()</script>')
    page.locator('#command').fill('Examine the resonance cradle')
    page.reload()
    expect(page.locator('#command')).to_have_value(
        'Examine the resonance cradle'
    )
    expect(page.locator('#send')).to_be_enabled()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(
        path=str(tmp_path / 'mobile.png'), full_page=True, animations='disabled'
    )
    page.context.set_offline(True)
    expect(page.locator('#connection')).to_have_text(
        'RECONNECTING', timeout=18000
    )
    expect(page.locator('#send')).to_be_disabled()
    page.context.set_offline(False)
    expect(page.locator('#connection')).to_have_text('CONNECTED', timeout=18000)
    expect(page.locator('#status')).to_have_text('Your move')
    expect(page.locator('#command')).to_have_value(
        'Examine the resonance cradle'
    )
    page.locator('#send').click()
    assert result.result(timeout=4) == 'Examine the resonance cradle'
    expect(page.locator('.player-action')).to_contain_text(
        'Examine the resonance cradle'
    )
    page.reload()
    expect(page.locator('#command')).to_have_value('')
    assert (
        len([e for e in session.snapshot()['entries'] if e['kind'] == 'action'])
        == 1
    )
    assert not errors
    browser.close()


def test_gm_action_spec_builder_and_desktop_layout(serve, tmp_path):
  url, _, result = serve('gm', entity_lib.OutputType.NEXT_ACTION_SPEC)
  with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    page.goto(url)
    expect(page.locator('#spec-builder')).to_be_visible()
    page.locator('#spec-type').select_option('choice')
    page.locator('#spec-options').fill(' North \nWest')
    page.locator('#build-spec').click()
    page.screenshot(
        path=str(tmp_path / 'desktop-gm.png'),
        full_page=True,
        animations='disabled',
    )
    page.locator('#send').click()
    assert ' North ' in result.result(timeout=4)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    browser.close()


def test_draft_edited_while_post_is_in_flight_is_not_lost(serve):
  url, _, result = serve()
  with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url)
    expect(page.locator('#connection')).to_have_text('CONNECTED')
    page.locator('#command').fill('Examine the cradle')

    def hold_reply(route):
      response = route.fetch()
      page.locator('#command').fill('Talk to Sable-9')
      route.fulfill(response=response)

    page.route('**/api/action', hold_reply)
    page.locator('#send').click()
    assert result.result(timeout=4) == 'Examine the cradle'
    expect(page.locator('.player-action')).to_contain_text('Examine the cradle')
    page.reload()
    expect(page.locator('#command')).to_have_value('Talk to Sable-9')
    expect(page.locator('#send')).to_be_disabled()
    browser.close()


@pytest.mark.parametrize('role', ['player', 'gm'])
def test_complete_basic_context_visible_verbatim_in_prefab_order(
    serve, tmp_path, role
):
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  model.sample_text.return_value = (
      'A basic perception answer.\nIts second line.'
  )
  reader = mock.Mock(return_value='LOOK')
  players, gm = adventure.build_cast(model, reader, player_prefab='basic')
  players[1].observe('NPC-PRIVATE-SENTINEL')
  gm.observe('GM-PRIVATE-SENTINEL')
  players[0].act()  # Unit fixture: mock model/input, no simulation launch.
  request = reader.call_args.args[0]
  url, session, result = serve(role=role, human_request=request)
  with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True
    )
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    expect(page.locator('#entity-context')).to_be_visible()
    displayed = page.locator('#context').text_content()
    assert displayed == request.context
    # These are the actual basic prefab labels, in its Concat order (not the
    # dependency/inference execution order). No new dict-key headings are added.
    labels = [
        'Question: What kind of person is Ilyra Venn?',
        'Question: What situation is Ilyra Venn in right now?',
        (
            'Question: What would a person like Ilyra Venn do in a situation'
            ' like this?'
        ),
    ]
    offsets = [displayed.index(label) for label in labels]
    assert offsets == sorted(offsets)
    assert displayed.count('A basic perception answer.\nIts second line.') == 3
    assert 'NPC-PRIVATE-SENTINEL' not in displayed
    assert 'GM-PRIVATE-SENTINEL' not in displayed
    assert (
        page.locator('#context').evaluate(
            'el => getComputedStyle(el).whiteSpace'
        )
        == 'pre-wrap'
    )
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.reload()
    expect(page.locator('#entity-context')).to_be_visible()
    assert page.locator('#context').text_content() == request.context
    page.screenshot(
        path=str(tmp_path / f'basic-context-{role}.png'), full_page=True
    )
    assert not errors
    assert not result.done()
    assert session.snapshot()['pending']['context'] == request.context
    browser.close()

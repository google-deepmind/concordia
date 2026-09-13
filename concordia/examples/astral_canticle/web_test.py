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

"""Real transport requests, no LLM calls or simulation launches."""

import concurrent.futures
import time

from concordia.examples.astral_canticle import human_io
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
web = pytest.importorskip('concordia.examples.astral_canticle.web')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

HEADERS = {'Origin': 'http://localhost', 'X-Astral-Client': '1'}


def request(spec=None, id='first'):
  return human_input.HumanInputRequest(
      request_id=id,
      entity_name='Ilyra Venn',
      action_spec=spec
      or entity_lib.free_action_spec(call_to_action='Your move?'),
      contexts={
          '__observation__': 'Observations:\n[observation] A silver door.',
          'SelfPerception': 'This controlled entity reflects on its own role.',
      },
      context=(
          'Observations:\n[observation] A silver door.\nThis controlled entity'
          ' reflects on its own role.'
      ),
  )


def wait_pending(session):
  end = time.monotonic() + 4
  while time.monotonic() < end:
    if session.snapshot()['pending']:
      return
    time.sleep(0.01)
  raise AssertionError('No pending input')


@pytest.fixture
def active():
  session = human_io.HumanSession()
  with concurrent.futures.ThreadPoolExecutor() as pool:
    result = pool.submit(session, request())
    wait_pending(session)
    with TestClient(
        web.create_app(session), base_url='http://localhost'
    ) as client:
      yield session, result, client
    session.finish('Stopped')


def test_reconnect_validate_then_idempotent_submit(active):
  session, result, client = active
  first = client.get('/api/state').json()
  assert client.get('/api/state').json() == first
  assert 'contexts' not in first['pending']
  assert first['pending']['context'] == request().context
  data = {'request_id': 'first', 'response': '   '}
  assert (
      client.post('/api/action', json=data, headers=HEADERS).status_code == 422
  )
  assert not result.done()
  data['response'] = 'Open the door'
  assert client.post('/api/action', json=data, headers=HEADERS).json() == {
      'accepted': True
  }
  assert result.result(timeout=3) == 'Open the door'
  assert client.post('/api/action', json=data, headers=HEADERS).json() == {
      'accepted': False
  }
  data['response'] = 'Actually go west'
  assert (
      client.post('/api/action', json=data, headers=HEADERS).status_code == 409
  )
  assert len(session.snapshot()['entries']) == 2
  assert '> Open the door' in client.get('/api/journal').text


@pytest.mark.parametrize(
    'headers',
    [
        {},
        {'Origin': 'https://attacker.example', 'X-Astral-Client': '1'},
        {'Origin': 'http://localhost'},
    ],
)
def test_cross_origin_or_simple_post_cannot_act(active, headers):
  _, result, client = active
  assert (
      client.post(
          '/api/action',
          json={'request_id': 'first', 'response': 'go'},
          headers=headers,
      ).status_code
      == 403
  )
  assert not result.done()


def test_bad_host_oversized_and_non_json_requests(active):
  _, result, client = active
  assert (
      client.get('/api/state', headers={'Host': 'attacker.example'}).status_code
      == 400
  )
  assert (
      client.post('/api/action', content='x', headers=HEADERS).status_code
      == 415
  )
  assert (
      client.post(
          '/api/action',
          json={'request_id': 'first', 'response': 'x' * 40000},
          headers=HEADERS,
      ).status_code
      == 413
  )
  assert not result.done()


@pytest.mark.parametrize(
    'spec,bad,good',
    [
        (
            entity_lib.choice_action_spec(
                call_to_action='Where?', options=(' North ', 'West')
            ),
            'North',
            ' North ',
        ),
        (
            entity_lib.float_action_spec(call_to_action='How much?'),
            'NaN',
            '1.5',
        ),
        (
            entity_lib.ActionSpec(
                call_to_action='Next spec?',
                output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
            ),
            '{}',
            '{"call_to_action":"Act", "output_type":"free"}',
        ),
    ],
)
def test_spec_validation_happens_before_unblocking(spec, bad, good):
  session = human_io.HumanSession(role='gm')
  with concurrent.futures.ThreadPoolExecutor() as pool:
    result = pool.submit(session, request(spec))
    wait_pending(session)
    try:
      with pytest.raises(ValueError):
        session.submit('first', bad)
      assert not result.done()
      assert session.snapshot()['pending']['context'] == request(spec).context
      session.submit('first', good)
      assert result.result(timeout=3) == good
    finally:
      session.finish('Done')


def test_close_unblocks_reader_and_restart_rejects_old_action():
  session = human_io.HumanSession()
  with concurrent.futures.ThreadPoolExecutor() as pool:
    result = pool.submit(session, request())
    wait_pending(session)
    session.finish('Closed')
    with pytest.raises(human_io.InputClosed):
      result.result(timeout=3)
  new_session = human_io.HumanSession()
  with pytest.raises(human_io.StaleRequest):
    new_session.submit('first', 'go')


def test_assets_csp_and_import_do_not_launch_a_simulation(active):
  _, _, client = active
  response = client.get('/')
  assert response.status_code == 200
  assert 'frame-ancestors' in response.headers['content-security-policy']
  assert response.headers['cache-control'] == 'no-store'
  assert client.get('/static/app.js').status_code == 200
  assert client.get('/docs').status_code == 404


@pytest.mark.parametrize('prefix', ['', '/astral-canticle-play'])
def test_proxy_mount_serves_index_api_and_assets_with_either_prefix(prefix):
  session = human_io.HumanSession()
  app = web.create_app(session, root_path='/astral-canticle-play')
  with TestClient(app, base_url='http://localhost') as client:
    for suffix in ('/', '/api/state', '/static/app.js', '/static/style.css'):
      response = client.get(prefix + suffix)
      assert response.status_code == 200, (prefix, suffix, response.text)

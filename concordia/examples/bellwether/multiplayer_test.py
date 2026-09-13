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

"""Privacy and role integrity using real HTTP/SSE; no simulation launches."""

import json
import pathlib
import threading
from unittest import mock
import uuid

from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether import multiplayer
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
from concordia.utils import browser_sessions
from concordia.utils import operation_service as ops
from concordia.utils import simulation_server
import httpx
import pytest


def envelope(game, operation, args=None):
  return dict(
      operation=operation,
      arguments=args or {},
      references=game.operations.references.copy(),
      revision=game.operations.revision,
      retry_key=str(uuid.uuid4()),
  )


def approve(game, label):
  row = next(r for r in game.sessions.pending() if r['label'] == label)
  game.operations.dispatch(
      'developer', envelope(game, 'session.approve', {'request_id': row['id']})
  )
  return row['id']


@pytest.fixture(name='hosted')
def hosted_fixture(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation in transport tests'),
  ):
    game = multiplayer.SharedGame(tmp_path, secure=False)
    server = simulation_server.SimulationServer(
        port=0,
        operation_service=game.operations,
        browser_sessions=game.sessions,
        html_content=pathlib.Path(__file__)
        .with_name('player.html')
        .read_text(encoding='utf-8'),
    )
    server.start()
    url = f'http://127.0.0.1:{server.bound_port}'
    try:
      yield game, server, url
    finally:
      game.close()
      server.stop()


def joined(game, url, role, label=None):
  client = httpx.Client()
  client.headers['Origin'] = url
  assert client.get(url).status_code == 200
  label = label or role
  assert (
      client.post(
          url + '/api/join', json={'label': label, 'role': role}
      ).status_code
      == 200
  )
  approve(game, label)
  return client


def test_only_two_humans_default_ai_policies(hosted):
  game, _, _ = hosted
  policies = {
      e.name: type(e.get_act_component()).__name__
      for e in game.simulation.get_entities()
  }
  assert policies == {
      'Coordinator': 'HumanActComponent',
      'Nell': 'HumanActComponent',
      'Mara': 'ConcatActComponent',
      'Ivo': 'ConcatActComponent',
      'Sam': 'ConcatActComponent',
  }


def test_unauthenticated_forged_and_pending_receive_no_private_data(hosted):
  game, _, url = hosted
  for path in ['/api/state', '/api/events', '/api/operations']:
    assert httpx.get(url + path).status_code == 403
  visitor = httpx.Client()
  assert visitor.get(url).status_code == 200
  for path in ['/api/state', '/api/operations']:
    assert 'PRIVATE_' not in visitor.get(url + path).text
  assert visitor.get(url + '/api/state').json()['result'] == {'lobby': True}
  result = visitor.post(
      url + '/api/join',
      json={'label': 'visitor', 'role': 'Nell'},
      headers={'Origin': url},
  )
  assert result.status_code == 200
  assert visitor.get(url + '/api/state').json()['result'] == {'lobby': True}
  result = visitor.post(
      url + '/api/dispatch',
      json=envelope(
          game,
          'session.approve',
          {'request_id': game.sessions.pending()[0]['id']},
      ),
      headers={'Origin': url},
  )
  assert result.status_code == 409
  assert result.json()['error']['code'] == 'unsupported_operation'
  for path in ['/events', '/status', '/cmd/set_component_state']:
    assert visitor.get(url + path).status_code == 404


def test_exact_origin_joins_and_mutations(hosted):
  game, _, url = hosted
  client = httpx.Client()
  client.get(url)
  for origin in [None, 'https://attacker.invalid', url + '.attacker.invalid']:
    headers = {'Origin': origin} if origin else {}
    response = client.post(
        url + '/api/join',
        json={'label': 'test', 'role': 'Nell'},
        headers=headers,
    )
    assert response.status_code == 403
  assert not game.sessions.pending()


def test_cannot_steal_role_or_inject_developer_or_actor(hosted):
  game, _, url = hosted
  joined(game, url, 'Nell')
  before = game.operations.revision
  client = httpx.Client()
  client.get(url)
  for role in ['Nell', 'developer', 'Mara']:
    response = client.post(
        url + '/api/join',
        json={'label': 'another', 'role': role},
        headers={'Origin': url},
    )
    assert response.status_code == 409
  assert game.operations.revision == before


def test_projections_public_private_spectator_and_revocation(hosted):
  game, _, url = hosted
  c = joined(game, url, 'Coordinator')
  n = joined(game, url, 'Nell')
  s = joined(game, url, 'spectator')
  with game.operations.lock:
    game.world.emit('speech', 'PUBLIC_MARKER')
    game.world.emit('speech', 'C_ONLY', ['Coordinator'])
    game.world.emit('speech', 'N_ONLY', ['Nell'])
    game.world.emit('speech', 'M_ONLY', ['Mara'])
    game.operations.publish({'kind': 'fixture.observations'})
  for client, visible, hidden in [
      (c, ['PUBLIC_MARKER', 'C_ONLY'], ['N_ONLY', 'M_ONLY', 'PRIVATE_NELL']),
      (
          n,
          ['PUBLIC_MARKER', 'N_ONLY', 'PRIVATE_NELL'],
          ['C_ONLY', 'M_ONLY', 'PRIVATE_MARA'],
      ),
      (s, ['PUBLIC_MARKER'], ['C_ONLY', 'N_ONLY', 'M_ONLY', 'PRIVATE_']),
  ]:
    content = client.get(url + '/api/state').text
    assert all(word in content for word in visible)
    assert all(word not in content for word in hidden)
  for operation in [
      'game.begin',
      'human.respond',
      'session.approve',
      'component.edit',
  ]:
    body = envelope(
        game,
        operation,
        {'request_id': 'x', 'response': 'wait'}
        if operation == 'human.respond'
        else {},
    )
    assert s.post(url + '/api/dispatch', json=body).status_code == 409
  with n.stream('GET', url + '/api/events', timeout=3) as stream:
    lines = stream.iter_lines()
    first = next(lines)
    assert 'role' in first
    row = next(r for r in game.sessions.pending() if r['role'] == 'Nell')
    game.operations.dispatch(
        'developer', envelope(game, 'session.revoke', {'request_id': row['id']})
    )
    next(lines)  # blank SSE separator
    message = next(lines)
    assert json.loads(message.removeprefix('data: '))['result'] == {
        'lobby': True
    }
  assert n.get(url + '/api/state').json()['result'] == {'lobby': True}


def test_nell_owns_pending_turn_retry_and_wrong_client_atomic(hosted):
  game, _, url = hosted
  c = joined(game, url, 'Coordinator')
  n = joined(game, url, 'Nell')
  game.world.data['agenda'] = [{
      'name': 'Nell',
      'purpose': 'request: release reserve',
      'audience': rules.NAMES,
      'watch': 0,
  }]
  request = human_input.HumanInputRequest(
      request_id='nell-request',
      entity_name='Nell',
      action_spec=entity_lib.free_action_spec(call_to_action='Your decision'),
      contexts={},
      context='NELL_PROMPT_ONLY',
  )
  result = []
  thread = threading.Thread(
      target=lambda: result.append(game.nell_inbox(request))
  )
  thread.start()
  try:
    assert 'NELL_PROMPT_ONLY' in n.get(url + '/api/state').text
    assert 'NELL_PROMPT_ONLY' not in c.get(url + '/api/state').text
    response = json.dumps(
        {'decision': 'decline', 'speech': 'I keep my reserve.'}
    )
    body = envelope(
        game,
        'human.respond',
        {'request_id': request.request_id, 'response': response},
    )
    before = game.operations.revision
    assert (
        c.post(url + '/api/dispatch', json=body).json()['error']['code']
        == 'wrong_turn'
    )
    assert game.operations.revision == before
    bad = envelope(
        game,
        'human.respond',
        {'request_id': request.request_id, 'response': 'not a decision'},
    )
    assert n.post(url + '/api/dispatch', json=bad).status_code == 409
    assert game.operations.revision == before
    stale = {**body, 'revision': -1}
    assert (
        n.post(url + '/api/dispatch', json=stale).json()['error']['code']
        == 'stale_revision'
    )
    accepted = n.post(url + '/api/dispatch', json=body)
    assert accepted.status_code == 200
    thread.join(2)
    assert result == [response]
    assert n.post(url + '/api/dispatch', json=body).json() == accepted.json()
    assert result == [response]
  finally:
    game.nell_inbox.finish('done')
    thread.join(2)


def test_cookie_attributes_and_unknown_session_fail_closed():

  store = browser_sessions.BrowserSessions(
      ('Nell',), cookie_path='/bellwether/'
  )
  _, cookie = store.identify('', create=True)
  assert all(
      word in cookie
      for word in ['HttpOnly', 'Secure', 'SameSite=Strict', 'Path=/bellwether/']
  )
  with pytest.raises(ops.OperationError):
    store.identify('unknown=forged')


def test_requires_both_players_before_start(hosted):
  game, _, url = hosted
  client = joined(game, url, 'Coordinator')
  response = client.post(
      url + '/api/dispatch', json=envelope(game, 'game.begin')
  )
  assert response.json()['error']['code'] == 'players_not_ready'
  assert game.developer_view()['quiescent']

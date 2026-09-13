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

"""Service integrity tests, without simulation or model execution."""

# Private worker probes deliberately test the quiescence guard.
# pylint: disable=protected-access

import copy
import json
import subprocess
import sys
import threading
from unittest import mock
import urllib.error
import urllib.request

from concordia.examples.bellwether import scenario
from concordia.examples.bellwether import service
from concordia.utils import operation_service as ops
from concordia.utils import simulation_server
import pytest


@pytest.fixture(name='game')
def game_fixture(tmp_path):
  with mock.patch(
      'concordia.prefabs.simulation.generic.Simulation.play',
      side_effect=AssertionError('No simulation in service tests'),
  ):
    value = service.Bellwether(tmp_path)
    yield value
    value.close()


def request(
    game, operation='component.edit', value='A revised account', key='edit1'
):
  current = game.operations.snapshot('developer')
  return {
      'operation': operation,
      'arguments': {'value': value},
      'references': current['references'],
      'revision': current['revision'],
      'retry_key': key,
  }


@pytest.mark.parametrize(
    'text',
    [
        '',
        '"quoted"\nnext line',
        '<script>literal & inert</script>',
        'Nell’s mémoire 🌧️',
    ],
)
def test_exact_edit_isolated_and_initial_unchanged(game, text):
  before = game.simulation.make_checkpoint_data()
  initial = copy.deepcopy(game.config.instances)
  result = game.operations.dispatch('developer', request(game, value=text))
  after = game.simulation.make_checkpoint_data()
  assert result['result']['value'] == text
  assert before['game_masters'] == after['game_masters']
  for name in ('Nell', 'Ivo', 'Sam', scenario.PLAYER):
    assert before['entities'][name] == after['entities'][name]
  assert game.config.instances == initial
  event = game.operations.events()[-1]
  assert event['informs'] == [] and event['after'] == text


def test_retry_stale_and_key_conflict_are_atomic(game):
  original = request(game)
  first = game.operations.dispatch('developer', original)
  state = game.operations.snapshot('developer')
  assert game.operations.dispatch('developer', original) == first
  assert game.operations.snapshot('developer') == state
  stale = {**original, 'retry_key': 'other'}
  with pytest.raises(ops.OperationError, match='State changed'):
    game.operations.dispatch('developer', stale)
  conflict = {**original, 'arguments': {'value': 'different'}}
  with pytest.raises(ops.OperationError, match='different input'):
    game.operations.dispatch('developer', conflict)
  assert game.operations.snapshot('developer') == state


@pytest.mark.parametrize(
    'value', [None, 3, False, ['text'], {'text': 'bad'}, 'x' * 8193]
)
def test_invalid_type_or_size_leaves_state_unchanged(game, value):
  before = game.operations.snapshot('developer')
  with pytest.raises(ops.OperationError):
    game.operations.dispatch('developer', request(game, value=value))
  assert game.operations.snapshot('developer') == before


def test_scope_and_read_only_operations(game):
  before = game.operations.snapshot('developer')
  bad = request(game)
  bad['references']['branch_id'] = 'other'
  with pytest.raises(ops.OperationError, match='Attach'):
    game.operations.dispatch('developer', bad)
  with pytest.raises(ops.OperationError):
    game.operations.dispatch('player', request(game))
  with pytest.raises(ops.OperationError):
    game.operations.dispatch('developer', {'operation': 'checkpoint.restore'})
  assert game.operations.snapshot('developer') == before


def test_real_inflight_thread_rejects_even_when_paused(game):
  release = threading.Event()
  game._worker = threading.Thread(
      target=release.wait
  )  # boundary probe, NOT engine work
  game._worker.start()
  try:
    assert game.server.step_controller.is_paused
    before = game.operations.snapshot('developer')
    with pytest.raises(ops.OperationError, match='still active'):
      game.operations.dispatch('developer', request(game))
    assert game.operations.snapshot('developer') == before
  finally:
    release.set()
    game._worker.join()


def test_snapshot_ownership_and_player_event_filter(game):
  snapshot = game.operations.snapshot('developer')
  snapshot['result']['initial']['reserve_fuel'] = 999
  assert (
      game.operations.snapshot('developer')['result']['initial']['reserve_fuel']
      == 2
  )
  client = game.operations.subscribe('player')
  first = client.get(timeout=1)
  game.operations.dispatch(
      'developer', request(game, value='PRIVATE_EDIT_NEVER_PLAYER')
  )
  second = client.get(timeout=1)
  game.operations.unsubscribe(client)
  for value in (first, second, game.operations.discover('player')):
    text = json.dumps(value)
    assert 'PRIVATE_' not in text
    assert 'reserve_fuel' not in text
    assert 'component.edit' not in text
    assert 'components_at_last_boundary' not in text


def test_fresh_component_ownership(game, tmp_path):
  other = service.Bellwether(tmp_path / 'other')
  try:
    game.operations.dispatch('developer', request(game, value='only A'))
    assert other.developer_view()['target'] != 'only A'
    assert game.operations.references != other.operations.references
  finally:
    other.close()


def test_player_http_boundary_blocks_all_legacy_routes(game):
  player = simulation_server.SimulationServer(
      port=0,
      html_content='<html>public only</html>',
      operation_service=game.operations,
      audience='player',
  )
  player.start()
  url = f'http://127.0.0.1:{player.bound_port}'
  try:
    for path in ('/', '/api/state', '/api/operations'):
      with urllib.request.urlopen(url + path) as response:
        assert 'PRIVATE_' not in response.read().decode()
        assert response.headers.get('Access-Control-Allow-Origin') is None
    for path in ('/events', '/status', '/cmd/play', '/runtime', '/project'):
      with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(url + path)
      assert error.value.code == 404
    for path, payload in [
        ('/cmd/set_component_state', {}),
        ('/api/dispatch', request(game)),
    ]:
      req = urllib.request.Request(
          url + path,
          data=json.dumps(payload).encode(),
          headers={'Content-Type': 'application/json'},
      )
      with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req)
      assert 'PRIVATE_' not in error.value.read().decode()
  finally:
    player.stop()


def test_cli_and_direct_dispatch_same_semantic_state_and_events(game, tmp_path):
  other = service.Bellwether(tmp_path / 'other')
  other.server.start()
  try:
    direct = request(game, value='Shared exact edit', key='same')
    attached = request(other, value='Shared exact edit', key='same')
    game.operations.dispatch('developer', direct)
    args = [
        sys.executable,
        '-m',
        'concordia.command_line_interface.concordia_session',
        '--url',
        f'http://127.0.0.1:{other.server.bound_port}',
        'call',
    ]
    result = subprocess.run(
        args,
        input=json.dumps(attached),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert game.developer_view() == other.developer_view()
    before = other.operations.snapshot('developer')
    retry = subprocess.run(
        args,
        input=json.dumps(attached),
        text=True,
        capture_output=True,
        check=False,
    )
    assert json.loads(retry.stdout) == json.loads(result.stdout)
    assert other.operations.snapshot('developer') == before
    bad = {**attached, 'retry_key': 'stale'}
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(bad))
    failure = subprocess.run(
        [*args, '--input', str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failure.returncode == 2 and 'stale_revision' in failure.stderr
  finally:
    other.close()


def test_simultaneous_retry_has_one_effect(game):
  from concurrent import futures  # pylint: disable=import-outside-toplevel

  payload = request(game)
  with futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(
        pool.map(
            lambda _: game.operations.dispatch('developer', payload), range(20)
        )
    )
  assert all(result == results[0] for result in results)
  assert len(game.operations.events()) == 1


def test_fixture_observer_has_no_fallback_or_spurious_content(game):
  from concordia.environment.engines import sequential  # pylint: disable=import-outside-toplevel

  with mock.patch(
      'concordia.language_model.no_language_model.NoLanguageModel.sample_text',
      side_effect=AssertionError('No model call'),
  ):
    gm = game.simulation.get_game_masters()[0]
    for actor in game.simulation.get_entities():
      assert sequential.Sequential().make_observation(gm, actor) == ''


def test_initial_safe_snapshot_includes_seeded_observations(game):
  snapshot = game.operations.snapshot('developer')['result']
  actual = game.simulation.make_checkpoint_data()
  recorded = snapshot['components_at_last_boundary']
  # Checkpoint creation increments its counter; compare entity state instead.
  for field in ('entities', 'game_masters', 'raw_log'):
    assert recorded[field] == actual[field]
  assert scenario.PUBLIC['opening'] in json.dumps(actual, ensure_ascii=False)
  assert 'PRIVATE_NELL' in json.dumps(actual, ensure_ascii=False)
  assert 'PRIVATE_NELL' not in json.dumps(game.operations.snapshot('player'))

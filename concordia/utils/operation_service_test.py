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

"""Coalesced SSE wakeups, unchanged direct snapshots, current authorization."""

import concurrent.futures
import json
import threading
import urllib.request

from concordia.utils import operation_service as ops
from concordia.utils import simulation_server
import pytest


def test_default_subscriber_keeps_owned_snapshot_contract():
  service = ops.OperationService(project_id='test')
  state = {'values': [0]}
  service.set_view('reader', lambda: state)
  client = service.subscribe('reader')
  try:
    initial = client.get_nowait()
    initial['result']['values'][0] = 99
    assert service.snapshot('reader')['result']['values'] == [0]
    state['values'][0] = 1
    service.publish({'kind': 'change'})
    result = client.get_nowait()
    assert result == service.snapshot('reader')
    assert result['revision'] == 1
  finally:
    service.unsubscribe(client)


def test_notification_burst_never_materializes_or_retains_private_views():
  service = ops.OperationService(project_id='test')
  calls = []
  state = {'step': 0, 'secret': 'PRIVATE_LARGE_VIEW' * 40000}

  def view():
    calls.append(True)
    return state

  service.set_view('reader', view)
  client = service.subscribe('reader', notifications_only=True)
  try:
    assert client.get_nowait() is None
    for step in range(40):
      state['step'] = step + 1
      service.publish({'kind': 'change', 'step': step + 1})
    assert not calls
    assert client.qsize() == 1
    assert client.get_nowait() is None
    assert client.empty()
    assert service.snapshot('reader')['result']['step'] == 40
    assert len(calls) == 1
    assert [e['revision'] for e in service.events()] == list(range(1, 41))
  finally:
    service.unsubscribe(client)
  service.publish({'kind': 'after-unsubscribe'})
  assert client.empty()


@pytest.mark.parametrize('notifications', [False, True])
def test_both_modes_validate_initial_scope(notifications):
  service = ops.OperationService(project_id='test')
  with pytest.raises(ops.OperationError, match='No view'):
    service.subscribe('forbidden', notifications_only=notifications)
  # Rejected subscription cannot poison a later publication.
  service.publish({'kind': 'valid-update'})
  assert service.revision == 1


def test_coalescing_keeps_latest_state_and_events_during_concurrent_updates():
  service = ops.OperationService(project_id='test')
  service.set_view('reader', lambda: {'latest': service.revision})
  client = service.subscribe('reader', notifications_only=True)
  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      list(
          pool.map(
              lambda i: service.publish({'kind': 'update', 'i': i}), range(80)
          )
      )
    assert client.qsize() == 1
    assert client.get_nowait() is None
    snapshot = service.snapshot('reader')
    assert snapshot['revision'] == snapshot['result']['latest'] == 80
    assert len(service.events()) == 80
    assert {e['i'] for e in service.events()} == set(range(80))
  finally:
    service.unsubscribe(client)


def test_http_delivery_resolves_current_role_after_backlog(monkeypatch):
  roles = {'browser': 'private'}
  service = ops.OperationService(
      project_id='test', audience_resolver=roles.__getitem__
  )
  service.set_view('private', lambda: {'private': 'NEVER_DELIVER_AFTER_REVOKE'})
  service.set_view('visitor', lambda: {'lobby': True})
  subscribed, calls = [], []
  original_subscribe = service.subscribe
  original_snapshot = service.snapshot
  waiting, release = threading.Event(), threading.Event()

  def subscribe(audience, **kwargs):
    subscribed.append(kwargs)
    return original_subscribe(audience, **kwargs)

  def snapshot(audience):
    waiting.set()
    assert release.wait(5), 'Test did not release delivery'
    value = original_snapshot(audience)
    calls.append(value)
    return value

  monkeypatch.setattr(service, 'subscribe', subscribe)
  monkeypatch.setattr(service, 'snapshot', snapshot)
  server = simulation_server.SimulationServer(
      port=0, operation_service=service, audience='browser'
  )
  server.start()
  try:
    with urllib.request.urlopen(
        f'http://127.0.0.1:{server.bound_port}/api/events', timeout=5
    ) as response:
      assert waiting.wait(2)
      assert subscribed == [{'notifications_only': True}]
      # Delivery is blocked before snapshotting: queued updates must not
      # freeze the previous private audience or contain a private payload.
      roles['browser'] = 'visitor'
      for i in range(40):
        service.publish({'kind': 'update', 'i': i})
      release.set()
      line = response.readline().decode('utf-8')
      assert line.startswith('data: ')
      received = json.loads(line[len('data: ') :])
      assert received['result'] == {'lobby': True}
      assert received['revision'] == 40
      assert set(received) == {'references', 'revision', 'result'}
      assert 'NEVER_DELIVER' not in line
      assert calls[0] == received
  finally:
    release.set()
    server.stop()

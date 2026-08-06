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

"""Tests for SimulationServer."""

import json
import queue
import urllib.error
import urllib.request

from absl.testing import absltest
from concordia.environment import step_controller as step_controller_lib
from concordia.utils import simulation_server


class _FakeSimulation:
  """Minimal stand-in for a Simulation, for exercising the edit endpoint."""

  def __init__(self):
    self.set_calls = []

  def set_component_dynamic_state(
      self, entity_name, component_name, key, value
  ):
    self.set_calls.append((entity_name, component_name, key, value))

  def make_checkpoint_data(self):
    return {'entities': {'alice': {}}, 'game_masters': {'gm': {}}}


def _request(url, method='GET', data=None):
  body = json.dumps(data).encode('utf-8') if data is not None else None
  req = urllib.request.Request(url, data=body, method=method)
  if body is not None:
    req.add_header('Content-Type', 'application/json')
  with urllib.request.urlopen(req, timeout=5) as response:
    return response.status, json.loads(response.read().decode('utf-8'))


class HostDefaultsTest(absltest.TestCase):

  def test_defaults_to_loopback_only(self):
    server = simulation_server.SimulationServer(port=0)
    self.assertEqual(server.host, '127.0.0.1')

  def test_host_is_configurable(self):
    server = simulation_server.SimulationServer(port=0, host='0.0.0.0')
    self.assertEqual(server.host, '0.0.0.0')

  def test_port_property_reflects_constructor_argument(self):
    server = simulation_server.SimulationServer(port=12345)
    self.assertEqual(server.port, 12345)

  def test_bound_port_raises_before_start(self):
    server = simulation_server.SimulationServer(port=0)
    with self.assertRaises(RuntimeError):
      _ = server.bound_port


class BroadcastTest(absltest.TestCase):

  def test_broadcast_step_delivers_to_all_queues(self):
    server = simulation_server.SimulationServer(port=0)
    q1: queue.Queue[str] = queue.Queue(maxsize=10)
    q2: queue.Queue[str] = queue.Queue(maxsize=10)
    server.server_sent_events_queues.extend([q1, q2])

    step_data = step_controller_lib.StepData(
        step=1,
        acting_entity='alice',
        action='wave',
        entity_actions={'alice': 'wave'},
        entity_logs={'alice': {}},
        game_master='gm',
    )
    server.broadcast_step(step_data)

    self.assertEqual(server.current_step_data['step'], 1)
    self.assertEqual(server.current_step_data['acting_entity'], 'alice')
    self.assertIn('data: ', q1.get_nowait())
    self.assertIn('data: ', q2.get_nowait())

  def test_broadcast_step_drops_full_queues(self):
    server = simulation_server.SimulationServer(port=0)
    full_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    full_queue.put_nowait('already full')
    server.server_sent_events_queues.append(full_queue)

    step_data = step_controller_lib.StepData(
        step=1,
        acting_entity='alice',
        action='wave',
        entity_actions={},
        entity_logs={},
    )
    server.broadcast_step(step_data)

    # The full queue should have been dropped from the list rather than
    # raising or blocking.
    self.assertNotIn(full_queue, server.server_sent_events_queues)

  def test_broadcast_entity_info_caches_payload(self):
    server = simulation_server.SimulationServer(port=0)
    server.broadcast_entity_info(
        {'entities': {'alice': {}}, 'game_masters': {}}
    )
    self.assertIsNotNone(server.cached_entity_info)
    self.assertEqual(server.cached_entity_info['entities'], {'alice': {}})


class HttpEndpointsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = simulation_server.SimulationServer(
        port=0, html_content='<html>hi</html>'
    )
    self.server.start()
    self.base_url = f'http://127.0.0.1:{self.server.bound_port}'

  def tearDown(self):
    self.server.stop()
    super().tearDown()

  def test_serves_configured_host(self):
    self.assertEqual(self.server.host, '127.0.0.1')

  def test_serves_html_at_root(self):
    with urllib.request.urlopen(self.base_url + '/', timeout=5) as response:
      self.assertEqual(response.status, 200)
      self.assertEqual(response.read().decode('utf-8'), '<html>hi</html>')

  def test_status_endpoint_reports_paused_by_default(self):
    status, payload = _request(self.base_url + '/status')
    self.assertEqual(status, 200)
    self.assertTrue(payload['is_paused'])
    self.assertFalse(payload['is_running'])

  def test_post_play_resumes_and_get_status_reflects_it(self):
    status, payload = _request(self.base_url + '/play', method='POST')
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'playing')
    self.assertTrue(self.server.step_controller.is_running)

  def test_post_pause(self):
    self.server.step_controller.play()
    status, payload = _request(self.base_url + '/pause', method='POST')
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'paused')
    self.assertTrue(self.server.step_controller.is_paused)

  def test_get_based_cmd_endpoints_mirror_post_endpoints(self):
    status, payload = _request(self.base_url + '/cmd/play')
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'playing')
    self.assertTrue(self.server.step_controller.is_running)

    status, payload = _request(self.base_url + '/cmd/pause')
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'paused')
    self.assertTrue(self.server.step_controller.is_paused)

  def test_unknown_path_returns_404(self):
    with self.assertRaises(urllib.error.HTTPError) as cm:
      urllib.request.urlopen(self.base_url + '/nonexistent', timeout=5)
    self.assertEqual(cm.exception.code, 404)

  def test_set_component_state_requires_paused(self):
    self.server.step_controller.play()
    status, payload = _request(
        self.base_url + '/cmd/set_component_state',
        method='POST',
        data={
            'entity_name': 'alice',
            'component_name': 'memory',
            'key': 'foo',
            'value': 'bar',
        },
    )
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'error')
    self.assertIn('paused', payload['message'])

  def test_set_component_state_requires_simulation(self):
    self.assertTrue(self.server.step_controller.is_paused)
    status, payload = _request(
        self.base_url + '/cmd/set_component_state',
        method='POST',
        data={
            'entity_name': 'alice',
            'component_name': 'memory',
            'key': 'foo',
            'value': 'bar',
        },
    )
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'error')
    self.assertIn('No simulation', payload['message'])

  def test_set_component_state_success_updates_simulation_and_broadcasts(self):
    fake_sim = _FakeSimulation()
    self.server.set_simulation(fake_sim)
    self.assertTrue(self.server.step_controller.is_paused)

    status, payload = _request(
        self.base_url + '/cmd/set_component_state',
        method='POST',
        data={
            'entity_name': 'alice',
            'component_name': 'memory',
            'key': 'foo',
            'value': 'bar',
        },
    )
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'ok')
    self.assertEqual(fake_sim.set_calls, [('alice', 'memory', 'foo', 'bar')])
    self.assertIsNotNone(self.server.cached_entity_info)

  def test_set_component_state_missing_field_returns_error(self):
    self.server.set_simulation(_FakeSimulation())
    status, payload = _request(
        self.base_url + '/cmd/set_component_state',
        method='POST',
        data={'entity_name': 'alice'},  # missing component_name/key/value
    )
    self.assertEqual(status, 200)
    self.assertEqual(payload['status'], 'error')


if __name__ == '__main__':
  absltest.main()

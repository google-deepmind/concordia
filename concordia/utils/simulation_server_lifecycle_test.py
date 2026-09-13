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

"""Lifecycle HTTP/SSE regressions; no simulation or model is launched."""

import json
import queue
import threading
import urllib.request

from absl.testing import absltest
from absl.testing import parameterized
from concordia.environment import step_controller
from concordia.utils import simulation_server


def _request(server, path, method='GET'):
  request = urllib.request.Request(
      f'http://127.0.0.1:{server.bound_port}{path}', method=method
  )
  with urllib.request.urlopen(request, timeout=5) as response:
    return json.load(response)


def _step_data(step):
  return step_controller.StepData(
      step=step,
      acting_entity='Alice',
      action='Mock callback',
      entity_actions={},
      entity_logs={},
  )


class LifecycleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = simulation_server.SimulationServer(port=0)
    self.server.start()
    self.addCleanup(self.server.stop)

  @parameterized.parameters('/cmd/step', '/cmd/play', '/cmd/pause')
  def test_empty_commands_are_rejected_without_permission(self, path):
    response = _request(self.server, path)
    self.assertEqual(response['status'], 'error')
    self.assertEqual(response['control_status']['state'], 'empty')
    self.assertTrue(self.server.step_controller.is_paused)
    self.assertEqual(_request(self.server, '/status')['current_step'], 0)

  def test_commands_publish_authoritative_status_to_all_clients(self):
    self.server.set_simulation(object())  # Binding only; no mock engine.
    queues = [queue.Queue(), queue.Queue()]
    self.server.server_sent_events_queues.extend(queues)
    revision = -1
    for path, method, state in [
        ('/cmd/play', 'GET', 'running'),
        ('/pause', 'POST', 'paused'),
        ('/play', 'POST', 'running'),
        ('/cmd/pause', 'GET', 'paused'),
        ('/step', 'POST', 'paused'),
    ]:
      response = _request(self.server, path, method)
      status = _request(self.server, '/status')
      self.assertEqual(status['state'], state)
      self.assertEqual(response['control_status'], status)
      self.assertGreater(status['revision'], revision)
      revision = status['revision']
      for client in queues:
        event = json.loads(client.get(timeout=2).removeprefix('data: '))
        self.assertEqual(event['control_status'], status)
    # A requested step is not a completed step.
    self.assertEqual(status['current_step'], 0)
    self.assertTrue(self.server.step_controller.wait_for_step_permission())
    self.assertTrue(self.server.step_controller.is_paused)

  def test_completion_retains_final_step_and_rejects_resume(self):
    self.server.set_simulation(object())
    _request(self.server, '/play', 'POST')
    self.server.broadcast_step(_step_data(7))
    self.server.broadcast_completion()
    status = _request(self.server, '/status')
    self.assertEqual(status['state'], 'completed')
    self.assertEqual(status['current_step'], 7)
    self.assertTrue(status['is_completed'])
    self.assertFalse(status['is_running'])
    self.assertTrue(self.server.step_controller.is_paused)
    for path, method in [
        ('/cmd/step', 'GET'),
        ('/play', 'POST'),
        ('/pause', 'POST'),
        ('/stop', 'POST'),
    ]:
      response = _request(self.server, path, method)
      self.assertEqual(response['status'], 'error')
      self.assertEqual(response['control_status'], status)

  def test_sse_replays_completion_after_last_step_for_new_clients(self):
    self.server.set_simulation(object())
    self.server.broadcast_step(_step_data(3))
    self.server.broadcast_completion()
    expected = _request(self.server, '/status')
    url = f'http://127.0.0.1:{self.server.bound_port}/events'
    with urllib.request.urlopen(url, timeout=5) as response:
      events = []
      while len(events) < 2:
        line = response.readline().decode()
        if line.startswith('data: '):
          events.append(json.loads(line[6:]))
      self.assertEqual(events[0]['step'], 3)
      self.assertEqual(events[1]['control_status'], expected)
      self.assertTrue(events[1]['completion'])
    # Wake the closed stream so its handler can detect disconnect promptly.
    for _ in range(3):
      self.server.broadcast_completion()

  def test_stopped_is_not_reported_as_running(self):
    self.server.set_simulation(object())
    response = _request(self.server, '/stop', 'POST')
    self.assertEqual(response['control_status']['state'], 'stopped')
    self.assertFalse(response['control_status']['is_running'])
    self.assertFalse(self.server.step_controller.wait_for_step_permission())
    self.assertEqual(_request(self.server, '/cmd/play')['status'], 'error')

  def test_completion_and_concurrent_command_have_ordered_terminal_status(self):
    self.server.set_simulation(object())
    client = self.server.subscribe_to_events()
    client.get_nowait()  # Initial paused snapshot.
    barrier = threading.Barrier(3)

    def complete():
      barrier.wait()
      self.server.broadcast_completion()

    def play():
      barrier.wait()
      self.server.execute_command('play')

    threads = [threading.Thread(target=complete), threading.Thread(target=play)]
    for thread in threads:
      thread.start()
    barrier.wait()
    for thread in threads:
      thread.join(2)
      self.assertFalse(thread.is_alive())
    events = []
    while not client.empty():
      events.append(json.loads(client.get_nowait().removeprefix('data: ')))
    statuses = [event['control_status'] for event in events]
    revisions = [status['revision'] for status in statuses]
    self.assertEqual(revisions, sorted(set(revisions)))
    self.assertEqual(statuses[-1]['state'], 'completed')
    self.assertEqual(self.server.get_status(), statuses[-1])

  def test_step_while_playing_does_not_claim_a_step(self):
    self.server.set_simulation(object())
    _request(self.server, '/play', 'POST')
    response = _request(self.server, '/cmd/step')
    self.assertEqual(response['status'], 'error')
    self.assertTrue(self.server.step_controller.is_running)
    self.assertEqual(response['control_status']['current_step'], 0)


class ControllerSemanticsTest(absltest.TestCase):
  """Exercise real permission waits, not a simulated run loop."""

  def test_one_step_grants_one_permission_then_blocks_until_play(self):
    controller = step_controller.StepController()
    first = threading.Event()
    second = threading.Event()

    def wait_twice():
      self.assertTrue(controller.wait_for_step_permission())
      first.set()
      self.assertTrue(controller.wait_for_step_permission())
      second.set()

    waiter = threading.Thread(target=wait_twice, daemon=True)
    waiter.start()
    try:
      self.assertFalse(first.wait(0.05))
      controller.step()
      self.assertTrue(first.wait(2))
      self.assertTrue(controller.is_paused)
      self.assertFalse(second.wait(0.05))
      controller.play()
      self.assertTrue(second.wait(2))
    finally:
      controller.stop()
      waiter.join(2)
    self.assertFalse(waiter.is_alive())

  def test_stop_releases_paused_waiter_without_permission(self):
    controller = step_controller.StepController()
    results = []
    waiter = threading.Thread(
        target=lambda: results.append(controller.wait_for_step_permission()),
        daemon=True,
    )
    waiter.start()
    controller.stop()
    waiter.join(2)
    self.assertEqual(results, [False])


if __name__ == '__main__':
  absltest.main()

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

"""Draft transactions and explicit runner boundaries; no simulation launches."""

import dataclasses
import json
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.environment import step_controller
from concordia.utils import simulation_server

from examples.project_editor import template


class ProjectServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.registry = template.registry()
    self.doc = self.registry.default_document(template.TEMPLATE_KEY)
    self.server = simulation_server.SimulationServer(port=0)
    self.runner = mock.Mock()
    self.server.configure_project(self.registry, self.doc, self.runner)

  @parameterized.parameters('{', '{}', '{"template":"unknown"}')
  def test_invalid_import_leaves_complete_draft_intact(self, text):
    before = self.server.get_project()
    with self.assertRaises(ValueError):
      self.server.replace_project(text, 0)
    self.assertEqual(self.server.get_project(), before)
    self.runner.assert_not_called()

  def test_configure_cannot_replace_an_existing_runtime(self):
    server = simulation_server.SimulationServer(port=0)
    runtime = object()
    server.set_simulation(runtime)
    with self.assertRaisesRegex(RuntimeError, 'before binding runtime'):
      server.configure_project(self.registry, self.doc, self.runner)
    self.assertIs(server.simulation, runtime)
    self.runner.assert_not_called()

  def test_saved_initial_state_does_not_touch_runtime(self):
    runtime = object()
    self.server.set_simulation(runtime)
    self.doc['instances'][0]['params']['goal'] = 'Reopened initial goal'
    self.server.replace_project(self.registry.dumps(self.doc), 0)
    self.assertIs(self.server.simulation, runtime)
    self.runner.assert_not_called()
    snapshot = self.server.get_project()
    snapshot['document']['premise'] = 'external mutation'
    self.assertNotEqual(
        self.server.get_project()['document']['premise'], 'external mutation'
    )

  def test_stale_save_and_run_rejected(self):
    self.server.replace_project(self.registry.dumps(self.doc), 0)
    before = self.server.get_project()
    for operation in (
        lambda: self.server.replace_project(self.registry.dumps(self.doc), 0),
        lambda: self.server.run_project(0),
    ):
      with self.assertRaisesRegex(ValueError, 'another tab'):
        operation()
    self.assertEqual(self.server.get_project(), before)
    self.runner.assert_not_called()

  def test_active_run_rejects_import_and_duplicate_run_even_when_paused(self):
    entered = threading.Event()
    release = threading.Event()
    received = []

    def runner(config):
      received.append(config)
      entered.set()
      release.wait(5)

    server = simulation_server.SimulationServer(port=0)
    server.configure_project(self.registry, self.doc, runner)
    self.doc['premise'] = 'Reopened draft'
    server.replace_project(self.registry.dumps(self.doc), 0)
    server.run_project(1)
    try:
      self.assertTrue(entered.wait(2))
      server.step_controller.pause()
      before = server.get_project()
      with self.assertRaisesRegex(ValueError, 'active'):
        server.run_project(1)
      with self.assertRaisesRegex(ValueError, 'active'):
        server.replace_project(self.registry.dumps(self.doc), 1)
      self.assertEqual(server.get_project(), before)
      self.assertLen(received, 1)
      self.assertEqual(
          dataclasses.asdict(received[0]),
          dataclasses.asdict(self.registry.to_config(self.doc)),
      )
    finally:
      release.set()
      thread = server._project_thread  # pylint: disable=protected-access
      assert thread is not None
      thread.join(2)
    self.assertEqual(server.get_project()['run']['status'], 'completed')

  def test_callback_failure_retains_draft_and_reports_error(self):
    self.runner.side_effect = RuntimeError('Example failure')
    self.server.run_project(0)
    thread = self.server._project_thread  # pylint: disable=protected-access
    assert thread is not None
    thread.join(2)
    self.assertEqual(self.server.get_project()['run']['status'], 'failed')
    self.assertEqual(
        self.server.get_project()['run']['message'], 'Example failure'
    )
    self.assertEqual(self.server.get_project()['document'], self.doc)
    # Correction is available only after the callback has returned.
    self.server.replace_project(self.registry.dumps(self.doc), 0)

  @parameterized.parameters('completed', 'stopped')
  def test_new_run_replaces_terminal_runtime_and_retained_state(self, terminal):
    old_runtime = object()

    def first_run(_):
      self.server.set_simulation(old_runtime)
      self.server.broadcast_step(
          step_controller.StepData(7, 'Old actor', 'Old action', {}, {})
      )
      self.server.broadcast_entity_info({'entities': {'Old actor': {}}})
      if terminal == 'completed':
        self.server.broadcast_completion()
      else:
        self.server.execute_command('stop')

    self.runner.side_effect = first_run
    self.server.run_project(0)
    thread = self.server._project_thread  # pylint: disable=protected-access
    assert thread is not None
    thread.join(2)
    self.assertEqual(self.server.get_status()['state'], terminal)

    entered = threading.Event()
    bind = threading.Event()
    bound = threading.Event()
    finish = threading.Event()
    new_runtime = object()

    def next_run(_):
      entered.set()
      bind.wait(5)
      self.server.set_simulation(new_runtime)
      bound.set()
      finish.wait(5)
      self.server.broadcast_completion()

    self.runner.side_effect = next_run
    self.server.run_project(0)
    try:
      self.assertTrue(entered.wait(2))
      self.assertIsNone(self.server.simulation)
      self.assertEqual(self.server.get_status()['state'], 'empty')
      self.assertEqual(self.server.get_status()['current_step'], 0)
      self.assertFalse(self.server.get_status()['is_completed'])
      retained = self.server.subscribe_to_events()
      event = json.loads(retained.get(timeout=2).removeprefix('data: '))
      self.assertEqual(event['control_status']['state'], 'empty')
      self.assertNotIn('entities', event)
      self.assertTrue(retained.empty())
      bind.set()
      self.assertTrue(bound.wait(2))
      self.assertIs(self.server.simulation, new_runtime)
      self.assertEqual(self.server.get_status()['state'], 'running')
      self.assertEqual(self.server.execute_command('pause')['status'], 'paused')
    finally:
      bind.set()
      finish.set()
      thread = self.server._project_thread  # pylint: disable=protected-access
      assert thread is not None
      thread.join(2)
    self.assertEqual(self.server.get_status()['state'], 'completed')


if __name__ == '__main__':
  absltest.main()

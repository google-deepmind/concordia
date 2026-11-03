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

"""Tests for the blocking HumanInput reader."""

import threading

from absl.testing import absltest
from concordia.components.agent import human_act_component
from concordia.contrib.components.agent import blocking_input
from concordia.typing import entity as entity_lib
from concordia.typing import human_input


def _request(request_id: str = 'r1') -> human_input.HumanInputRequest:
  return human_input.HumanInputRequest(
      request_id=request_id,
      entity_name='Alice',
      action_spec=entity_lib.free_action_spec(call_to_action='Go?'),
      contexts={},
      context='You are Alice.',
  )


class _Entity:
  name = 'Alice'


class BlockingInputTest(absltest.TestCase):

  def _ask(self, reader, request, results):
    try:
      results.append(reader(request))
    except blocking_input.InputClosedError as error:
      results.append(error)

  def test_answer_from_another_thread(self):
    reader = blocking_input.BlockingInput()
    results = []
    thread = threading.Thread(
        target=self._ask, args=(reader, _request('r1'), results)
    )
    thread.start()
    request = reader.wait_for_request(timeout=5)
    assert request is not None
    self.assertEqual(request.request_id, 'r1')
    self.assertFalse(reader.submit('stale', 'nope'))
    self.assertTrue(reader.submit('r1', 'hello'))
    thread.join(timeout=5)
    self.assertEqual(results, ['hello'])
    self.assertIsNone(reader.pending())
    self.assertEqual(reader.answered, 1)

  def test_close_releases_the_waiting_thread(self):
    reader = blocking_input.BlockingInput()
    results = []
    thread = threading.Thread(
        target=self._ask, args=(reader, _request(), results)
    )
    thread.start()
    reader.wait_for_request(timeout=5)
    reader.close()
    thread.join(timeout=5)
    self.assertIsInstance(results[0], blocking_input.InputClosedError)
    self.assertFalse(reader.submit('r1', 'late'))
    with self.assertRaises(blocking_input.InputClosedError):
      reader(_request('r2'))

  def test_validation_retry_replaces_the_pending_request(self):
    reader = blocking_input.BlockingInput()
    act = human_act_component.HumanActComponent(reader)
    act.set_entity(_Entity())  # pyrefly: ignore[bad-argument-type]
    spec = entity_lib.choice_action_spec(
        call_to_action='Agree?', options=('yes', 'no')
    )
    results = []
    thread = threading.Thread(
        target=lambda: results.append(act.get_action_attempt({}, spec))
    )
    thread.start()
    first = reader.wait_for_request(timeout=5)
    assert first is not None
    self.assertTrue(reader.submit(first.request_id, 'maybe'))
    # HumanActComponent asks again with the same id and an error.
    retry = reader.wait_for_request(timeout=5)
    assert retry is not None
    self.assertIsNotNone(retry.error)
    self.assertEqual(retry.request_id, first.request_id)
    self.assertTrue(reader.submit(retry.request_id, 'yes'))
    thread.join(timeout=5)
    self.assertEqual(results, ['yes'])


if __name__ == '__main__':
  absltest.main()

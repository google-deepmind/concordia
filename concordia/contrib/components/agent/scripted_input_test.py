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

"""Tests for the scripted and policy HumanInput readers."""

from absl.testing import absltest
from concordia.components.agent import human_act_component
from concordia.contrib.components.agent import scripted_input
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


class ScriptedInputTest(absltest.TestCase):

  def test_answers_in_order_then_raises(self):
    reader = scripted_input.ScriptedInput(['a', 'b'])
    self.assertEqual(reader(_request()), 'a')
    self.assertEqual(reader.remaining, 1)
    self.assertEqual(reader(_request()), 'b')
    with self.assertRaises(IndexError):
      reader(_request())

  def test_policy_sees_the_request_and_is_recorded(self):
    reader = scripted_input.PolicyInput(lambda r: r.context.upper())
    self.assertEqual(reader(_request()), 'YOU ARE ALICE.')
    self.assertEqual(reader.answers, ['YOU ARE ALICE.'])
    self.assertEqual(reader.requests[0].entity_name, 'Alice')

  def test_correction_after_validation_error(self):
    # The first answer is not an option; HumanActComponent retries and the
    # script supplies the correction.
    reader = scripted_input.ScriptedInput(['maybe', 'yes'])
    act = human_act_component.HumanActComponent(reader)
    spec = entity_lib.choice_action_spec(
        call_to_action='Agree?', options=('yes', 'no')
    )

    class _Entity:
      name = 'Alice'

    act.set_entity(_Entity())  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(act.get_action_attempt({}, spec), 'yes')
    self.assertIsNotNone(reader.requests[1].error)


if __name__ == '__main__':
  absltest.main()

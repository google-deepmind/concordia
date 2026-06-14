# Copyright 2024 DeepMind Technologies Limited.
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

from absl.testing import absltest
from absl.testing import parameterized
from concordia.utils import sampling


class ExtractChoiceResponseTest(parameterized.TestCase):

  @parameterized.parameters(
      ('a', 'a'),
      ('a)', 'a'),
      ('(a)', 'a'),
      ('foo(a)bar', 'a'),
  )
  def test_extracts_choice(self, response, expected):
    self.assertEqual(sampling.extract_choice_response(response), expected)


class DynamicallyAdjustTemperatureTest(parameterized.TestCase):

  @parameterized.parameters(
      # (attempts, max_attempts, expected_temperature)
      (1, 10, 0.0),  # first attempt: no temperature bump
      (2, 10, 0.5),  # early retries: moderate temperature
      (4, 10, 0.5),  # still below the midpoint
      (5, 10, 0.75),  # exact midpoint must escalate, not reset to 0.0
      (6, 10, 0.75),  # past the midpoint
      (10, 10, 0.75),
  )
  def test_temperature_schedule(self, attempts, max_attempts, expected):
    self.assertEqual(
        sampling.dynamically_adjust_temperature(attempts, max_attempts),
        expected,
    )

  def test_midpoint_does_not_reset_temperature(self):
    # Regression: at attempts == max_attempts / 2 (even max_attempts), both the
    # `< max/2` and `> max/2` checks were false, so the temperature silently
    # dropped back to 0.0 instead of escalating.
    self.assertEqual(sampling.dynamically_adjust_temperature(3, 6), 0.75)
    self.assertEqual(sampling.dynamically_adjust_temperature(4, 8), 0.75)


if __name__ == '__main__':
  absltest.main()

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

"""The matrix audit finds no problems, and the mock passes the golden set."""

from absl.testing import absltest
from examples.crucible import audit
from examples.crucible import golden
from examples.crucible import model_setup


class MatrixTest(absltest.TestCase):

  def test_matrix_has_no_problems(self):
    result = audit.audit()
    self.assertEqual(result['problems'], [])
    self.assertGreaterEqual(result['counts']['matrix_cells'], 25)

  def test_mock_passes_every_golden_case(self):
    result = golden.run(model_setup.mock_model())
    self.assertEqual(result['failures'], [])
    self.assertTrue(result['passed'])


if __name__ == '__main__':
  absltest.main()

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
from concordia.language_model import language_model
from examples.crucible import audit
from examples.crucible import golden
from examples.crucible import golden_passiton
from examples.crucible import model_setup


class MatrixTest(absltest.TestCase):

  def test_matrix_has_no_problems(self):
    result = audit.audit()
    self.assertEqual(result['problems'], [])
    self.assertGreaterEqual(result['counts']['matrix_cells'], 25)

  def test_pass_it_on_matrix_has_no_problems(self):
    result = audit.audit('passiton')
    self.assertEqual(result['problems'], [])
    self.assertGreaterEqual(result['counts']['matrix_cells'], 25)

  def test_mock_passes_every_pass_it_on_golden_case(self):
    result = golden_passiton.run(model_setup.mock_model(), repeats=2)
    self.assertEqual(result['failures'], [])
    self.assertTrue(result['passed'])
    self.assertEqual(result['flaky'], [])

  def test_night_wire_matrix_has_no_problems_and_every_cell_is_taught(self):
    result = audit.audit('nightwire')
    self.assertEqual(result['problems'], [])
    self.assertGreaterEqual(result['counts']['matrix_cells'], 25)
    self.assertEqual(
        result['untaught'], [ref for ref in result['untaught'] if '->' in ref]
    )  # every reading cell is a chapter's lesson

  def test_mock_passes_every_night_wire_golden_case(self):
    result = golden_passiton.run(
        model_setup.mock_model(), repeats=2, toy='nightwire'
    )
    self.assertEqual(result['failures'], [])
    self.assertTrue(result['passed'])
    self.assertEqual(result['flaky'], [])

  def test_mock_passes_every_golden_case(self):
    result = golden.run(model_setup.mock_model(), repeats=2)
    self.assertEqual(result['failures'], [])
    self.assertTrue(result['passed'])
    self.assertEqual(result['flaky'], [])  # the mock is deterministic

  def test_a_failing_model_fails_the_case_not_the_run(self):
    class Down(language_model.LanguageModel):

      def sample_text(self, prompt, **kwargs):
        raise TimeoutError('no reply')

      def sample_choice(self, prompt, responses, **kwargs):
        raise TimeoutError('no reply')

    case = golden.load_cases()[0]
    result = golden.run_case(Down(), case)
    self.assertFalse(result['ok'])
    self.assertEqual(result['answer'], 'error: TimeoutError')


if __name__ == '__main__':
  absltest.main()

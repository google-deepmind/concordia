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

"""Tests for golden sets."""

import itertools

from absl.testing import absltest
from concordia.contrib.testing import golden_set

CASES = [
    golden_set.Case('law', expect=['rest'], hard=True),
    golden_set.Case('style-1', expect=['build']),
    golden_set.Case('style-2', expect=['build', 'gather']),
]


class GoldenSetTest(absltest.TestCase):

  def test_a_model_that_keeps_the_laws_passes(self):
    answers = {'law': 'rest', 'style-1': 'build', 'style-2': 'gather'}
    report = golden_set.run(CASES, lambda c: answers[c.id])
    self.assertTrue(report.passed)
    self.assertEqual([r.hard_passed for r in report.runs], [1, 1])
    self.assertEqual(report.failures, {})
    self.assertEqual(report.flaky, [])

  def test_one_hard_miss_fails_however_good_the_rest(self):
    answers = {'law': 'build', 'style-1': 'build', 'style-2': 'build'}
    report = golden_set.run(CASES, lambda c: answers[c.id], repeats=1)
    self.assertFalse(report.passed)
    self.assertEqual(report.failures, {'law': ['build']})

  def test_soft_agreement_needs_its_share(self):
    answers = {'law': 'rest', 'style-1': 'rest', 'style-2': 'rest'}
    run = golden_set.run_once(CASES, lambda c: answers[c.id], soft_pass=0.5)
    self.assertEqual(run.soft_agreement, 0.0)
    self.assertFalse(run.passed)

  def test_every_run_must_pass_and_changed_answers_are_flaky(self):
    replies = itertools.cycle(['build', 'rest'])  # style-1 flips each run

    def answer(case):
      return next(replies) if case.id == 'style-1' else case.expect[0]

    report = golden_set.run(CASES, answer, repeats=2, soft_pass=1.0)
    self.assertEqual([r.passed for r in report.runs], [True, False])
    self.assertFalse(report.passed)
    self.assertEqual(report.flaky, ['style-1'])

  def test_a_failing_provider_fails_the_case_not_the_run(self):
    def answer(case):
      if case.id == 'law':
        raise TimeoutError('no reply')
      return case.expect[0]

    report = golden_set.run(CASES, answer, repeats=1)
    self.assertEqual(report.runs[0].answers['law'], 'error: TimeoutError')
    self.assertFalse(report.passed)

  def test_compare_names_what_a_change_fixed_and_broke(self):
    old = {'law': 'build', 'style-1': 'build', 'style-2': 'rest'}
    new = {'law': 'rest', 'style-1': 'rest', 'style-2': 'rest'}
    before = golden_set.run(CASES, lambda c: old[c.id], repeats=1)
    after = golden_set.run(CASES, lambda c: new[c.id], repeats=1)
    diff = golden_set.compare(CASES, before, after)
    self.assertEqual(diff['fixed'], ['law'])
    self.assertEqual(diff['broken'], ['style-1'])
    self.assertEqual(diff['still_failing'], ['style-2'])
    self.assertEqual(diff['flaky_after'], [])

  def test_ids_must_be_unique(self):
    with self.assertRaises(ValueError):
      golden_set.run([CASES[0], CASES[0]], lambda c: 'rest')


if __name__ == '__main__':
  absltest.main()

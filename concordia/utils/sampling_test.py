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

"""Tests for the sampling helper functions."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.utils import sampling


class ExtractChoiceResponseTest(parameterized.TestCase):

  @parameterized.parameters(
      ("don't)", None),
      ('non-aa)', None),
      ('a. or (aa)', None),
      ('a.\n(aa)', None),
      ('(a) or aa.', None),
      ('( a)', 'a'),
      ('( aa )', 'aa'),
      ('The answer is ( aa).', 'aa'),
  )
  def test_reviewed_word_boundaries_and_explicit_forms(self, sample, expected):
    self.assertEqual(
        sampling.extract_choice_response(sample, ['a', 't', 'aa']), expected
    )

  @parameterized.parameters(
      ('aa', 'aa'),
      ('aa)', 'aa'),
      ('aa.', 'aa'),
      (' (aa) ', 'aa'),
      ('foo(aa)bar', 'aa'),
      ('aa) because', 'aa'),
      ('(aa) then (aa)', 'aa'),
      ('(probably) (ba)', 'ba'),
      ('a', 'a'),
      ('a)', 'a'),
      ('a.', 'a'),
      ('a or aa', None),
      ('(a) or (aa)', None),
      ('(aa) (ba)', None),
      ('(aaa)', None),
      ('aaa', None),
      ('aardvark)', None),
      ('(aardvark)', None),
      ('I have a preference.', None),
      ('(not aa)', None),
      ('x_aa)', None),
      ('', None),
  )
  def test_supplied_choices(self, sample, expected):
    self.assertEqual(
        sampling.extract_choice_response(
            sample, responses=['a', 'b', 'aa', 'ba']
        ),
        expected,
    )

  @parameterized.parameters('aa', 'aa)', '(aa)', '(a) (b)', 'ab', 'a or b')
  def test_legacy_does_not_accept_fragments_or_ambiguity(self, sample):
    self.assertIsNone(sampling.extract_choice_response(sample))

  @parameterized.parameters('aa', 'aa)', '(aa)', 'aaa)', '(banana)')
  def test_unavailable_label_is_not_a_shorter_choice(self, sample):
    self.assertIsNone(
        sampling.extract_choice_response(sample, responses=['a', 'b', 'n'])
    )

  def test_supplied_choices_are_literal_and_may_be_empty(self):
    self.assertEqual(
        sampling.extract_choice_response('(a+)', responses=['a', 'a+']), 'a+'
    )
    self.assertEqual(
        sampling.extract_choice_response(
            'go home', responses=['go', 'go home']
        ),
        'go home',
    )
    self.assertIsNone(sampling.extract_choice_response('(a)', responses=[]))

  @parameterized.named_parameters(
      ('single_letter', 'a', 'a'),
      ('single_digit', '5', '5'),
      ('letter_then_paren', 'a)', 'a'),
      ('letter_then_period', 'a.', 'a'),
      ('parenthesized_at_start', '(a)bar', 'a'),
      ('parenthesized_in_middle', 'foo(a)bar', 'a'),
      ('parenthesized_at_end', 'The answer is (b).', 'b'),
      ('letter_then_paren_in_longer_string', 'a) because', 'a'),
      (
          'parenthesized_word_before_choice',
          'I think (probably) the answer is (b)',
          'b',
      ),
      ('unopened_word_before_choice', 'maybe) (a)', 'a'),
  )
  def test_extracts_choice(self, sample, expected):
    self.assertEqual(sampling.extract_choice_response(sample), expected)

  @parameterized.named_parameters(
      ('empty_string', ''),
      ('no_parenthesized_choice', 'no choice here'),
      ('word_without_closing_paren', '(abc'),
      ('parenthesized_word_only', '(yes) a'),
  )
  def test_returns_none_when_no_choice(self, sample):
    self.assertIsNone(sampling.extract_choice_response(sample))


class DynamicallyAdjustTemperatureTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # max_attempts == 10, so the midpoint is 5.0.
      ('first_attempt', 1, 10, 0.0),
      ('second_attempt_below_midpoint', 2, 10, 0.5),
      ('just_below_midpoint', 4, 10, 0.5),
      ('exactly_at_midpoint', 5, 10, 0.75),
      ('just_above_midpoint', 6, 10, 0.75),
      ('final_attempt', 10, 10, 0.75),
      ('zero_attempts', 0, 10, 0.0),
  )
  def test_temperature(self, attempts, max_attempts, expected):
    self.assertEqual(
        sampling.dynamically_adjust_temperature(attempts, max_attempts),
        expected,
    )


if __name__ == '__main__':
  absltest.main()

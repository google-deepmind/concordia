# Copyright 2023 DeepMind Technologies Limited.
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

"""Tests for the string formatting utilities."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.utils import text


class WrapTest(parameterized.TestCase):

  def test_short_string_is_unchanged(self):
    self.assertEqual(text.wrap('hello world', width=70), 'hello world')

  def test_empty_string(self):
    self.assertEqual(text.wrap(''), '')

  def test_wraps_at_width(self):
    self.assertEqual(text.wrap('aaaa bbbb cccc', width=5), 'aaaa\nbbbb\ncccc')

  def test_preserves_existing_newlines(self):
    self.assertEqual(text.wrap('line one\nline two', width=70),
                     'line one\nline two')

  def test_no_line_exceeds_width(self):
    string = 'the quick brown fox jumps over the lazy dog ' * 5
    width = 20
    for line in text.wrap(string, width=width).split('\n'):
      self.assertLessEqual(len(line), width)


class TruncateTest(parameterized.TestCase):

  def test_no_truncation_by_default(self):
    self.assertEqual(text.truncate('hello world'), 'hello world')

  def test_truncates_to_max_length(self):
    self.assertEqual(text.truncate('hello world', max_length=5), 'hello')

  def test_max_length_longer_than_string(self):
    self.assertEqual(text.truncate('hello', max_length=100), 'hello')

  @parameterized.named_parameters(
      ('single_delimiter', 'hello world', (' ',), 'hello'),
      ('delimiter_absent', 'hello', (' ',), 'hello'),
      ('multiple_delimiters', 'a.b,c', (',', '.'), 'a'),
      ('stops_at_first_occurrence', 'a-b-c', ('-',), 'a'),
  )
  def test_delimiters(self, string, delimiters, expected):
    self.assertEqual(
        text.truncate(string, delimiters=delimiters), expected)

  def test_max_length_and_delimiters_combined(self):
    self.assertEqual(
        text.truncate('hello world foo', max_length=11, delimiters=(' ',)),
        'hello',
    )


if __name__ == '__main__':
  absltest.main()

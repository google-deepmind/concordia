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

"""Tests for markdown stripping utilities."""

from concordia.language_model import markdown_stripper
from absl.testing import absltest


class MarkdownStripperTest(absltest.TestCase):

  def test_strip_triple_backticks(self):
    text = '```json\n{"key": "value"}\n```'
    result = markdown_stripper.strip_markdown(text)
    self.assertEqual(result, '{"key": "value"}')

  def test_strip_triple_backticks_no_language(self):
    text = '```\nsome code\n```'
    result = markdown_stripper.strip_markdown(text)
    self.assertEqual(result, 'some code')

  def test_strip_inline_code(self):
    text = 'Use the `print()` function.'
    result = markdown_stripper.strip_markdown(text)
    self.assertEqual(result, 'Use the print() function.')

  def test_strip_leading_trailing_whitespace(self):
    text = '   some text   '
    result = markdown_stripper.strip_markdown(text)
    self.assertEqual(result, 'some text')

  def test_no_markdown(self):
    text = 'Just plain text.'
    result = markdown_stripper.strip_markdown(text)
    self.assertEqual(result, 'Just plain text.')

  def test_mixed_markdown(self):
    text = '```json\n{"name": "test"}\n```\nUse `print()` here.   '
    result = markdown_stripper.strip_markdown(text)
    self.assertEqual(result, '{"name": "test"}\nUse print() here.')


if __name__ == '__main__':
  absltest.main()
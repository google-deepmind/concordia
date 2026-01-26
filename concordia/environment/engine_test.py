# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for the engine module, including action spec parsing and formatting."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.environment import engine
from concordia.typing import entity as entity_lib


class EngineTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'free_no_prompt',
          'type: free',
          entity_lib.ActionSpec(
              call_to_action=entity_lib.DEFAULT_CALL_TO_ACTION,
              output_type=entity_lib.OutputType.FREE,
          ),
      ),
      (
          'free_with_prompt',
          'prompt: Describe the scene.;;type: free',
          entity_lib.ActionSpec(
              call_to_action='Describe the scene.',
              output_type=entity_lib.OutputType.FREE,
          ),
      ),
      (
          'choice_no_prompt_no_options',
          'type: choice',
          entity_lib.ActionSpec(
              call_to_action=entity_lib.DEFAULT_CALL_TO_ACTION,
              output_type=entity_lib.OutputType.FREE,
          ),
      ),
      (
          'choice_with_prompt_no_options',
          'prompt: Pick one.;;type: choice',
          entity_lib.ActionSpec(
              call_to_action='Pick one.',
              output_type=entity_lib.OutputType.FREE,
          ),
      ),
      (
          'choice_with_options',
          'type: choice options: a, b',
          entity_lib.ActionSpec(
              call_to_action=entity_lib.DEFAULT_CALL_TO_ACTION,
              output_type=entity_lib.OutputType.CHOICE,
              options=('a', 'b'),
          ),
      ),
      (
          'choice_with_prompt_and_options',
          'prompt: Pick one.;;type: choice options: a, b',
          entity_lib.ActionSpec(
              call_to_action='Pick one.',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a', 'b'),
          ),
      ),
      (
          'choice_with_escaped_comma',
          r'type: choice options: a\,b, c',
          entity_lib.ActionSpec(
              call_to_action=entity_lib.DEFAULT_CALL_TO_ACTION,
              output_type=entity_lib.OutputType.CHOICE,
              options=('a,b', 'c'),
          ),
      ),
      (
          'skip',
          'type: __SKIP_THIS_STEP__',
          entity_lib.ActionSpec(
              call_to_action='',
              output_type=entity_lib.OutputType.SKIP_THIS_STEP,
          ),
      ),
  )
  def test_action_spec_parser(self, action_spec_string, expected_spec):
    spec = engine.action_spec_parser(action_spec_string)
    self.assertEqual(spec, expected_spec)

  @parameterized.named_parameters(
      (
          'free',
          entity_lib.ActionSpec(
              call_to_action='Describe the scene.',
              output_type=entity_lib.OutputType.FREE,
          ),
          'prompt: Describe the scene.;;type: free',
      ),
      (
          'choice',
          entity_lib.ActionSpec(
              call_to_action='Pick one.',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a', 'b'),
          ),
          'prompt: Pick one.;;type: choice options: a, b',
      ),
      (
          'choice_escaped',
          entity_lib.ActionSpec(
              call_to_action='Pick one.',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a,b', 'c'),
          ),
          r'prompt: Pick one.;;type: choice options: a\,b, c',
      ),
      (
          'skip',
          entity_lib.ActionSpec(
              call_to_action='Wait.',
              output_type=entity_lib.OutputType.SKIP_THIS_STEP,
          ),
          'prompt: Wait.;;type: __SKIP_THIS_STEP__',
      ),
  )
  def test_action_spec_to_string(self, action_spec, expected_string):
    self.assertEqual(engine.action_spec_to_string(action_spec), expected_string)

  def test_split_options(self):
    self.assertEqual(engine.split_options('a, b, c'), ('a', 'b', 'c'))
    self.assertEqual(engine.split_options('a'), ('a',))
    self.assertEqual(engine.split_options(r'a\,b, c'), ('a,b', 'c'))
    self.assertEqual(
        engine.split_options(r'a\\,b'), (r'a\,b',)
    )  # Double backslash is not handled as escape for escape


if __name__ == '__main__':
  absltest.main()

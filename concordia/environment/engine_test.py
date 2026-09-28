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

import json

from absl.testing import absltest
from absl.testing import parameterized
from concordia.environment import engine
from concordia.typing import entity as entity_lib


class EngineTest(parameterized.TestCase):

  @parameterized.parameters(
      ({},), ({'Alice': 'Alice: HARVEST 1\nBob: a quote'},)
  )
  def test_resolution_actions_round_trip(self, actions):
    spec = entity_lib.ActionSpec(
        call_to_action='Resolve the decisions.',
        output_type=entity_lib.OutputType.RESOLVE,
        entity_actions=actions,
    )
    serialized = engine.action_spec_to_string(spec)
    self.assertEqual(json.loads(serialized)['entity_actions'], actions)
    restored = engine.action_spec_parser(serialized)
    self.assertEqual(restored, spec)
    self.assertIsNotNone(restored.entity_actions)

  def test_resolution_actions_are_an_immutable_snapshot(self):
    actions = {'Alice': 'HARVEST 1'}
    spec = entity_lib.ActionSpec(
        call_to_action='Resolve.',
        output_type=entity_lib.OutputType.RESOLVE,
        entity_actions=actions,
    )
    actions['Alice'] = 'HARVEST 9'
    assert spec.entity_actions is not None
    self.assertEqual(spec.entity_actions, {'Alice': 'HARVEST 1'})
    with self.assertRaises(TypeError):
      spec.entity_actions['Alice'] = 'HARVEST 9'  # pyrefly: ignore[unsupported-operation]
    encoded = spec.to_dict()
    encoded['entity_actions']['Alice'] = 'HARVEST 9'
    self.assertEqual(spec.entity_actions['Alice'], 'HARVEST 1')

  def test_legacy_resolution_spec_omits_actions(self):
    spec = entity_lib.ActionSpec(
        call_to_action='Resolve.', output_type=entity_lib.OutputType.RESOLVE
    )
    self.assertEqual(spec.to_dict(), {
        'call_to_action': 'Resolve.', 'output_type': 'resolve',
        'options': [], 'tag': None,
    })

  def test_entity_actions_require_resolution(self):
    with self.assertRaisesRegex(ValueError, 'only supported for RESOLVE'):
      entity_lib.ActionSpec(
          call_to_action='Act.', output_type=entity_lib.OutputType.FREE,
          entity_actions={},
      )

  @parameterized.named_parameters(
      (
          'free',
          {
              'call_to_action': 'Describe the scene.',
              'output_type': 'free',
              'options': [],
              'tag': None,
          },
          entity_lib.ActionSpec(
              call_to_action='Describe the scene.',
              output_type=entity_lib.OutputType.FREE,
          ),
      ),
      (
          'choice',
          {
              'call_to_action': 'Pick one.',
              'output_type': 'choice',
              'options': ['a', 'b'],
              'tag': None,
          },
          entity_lib.ActionSpec(
              call_to_action='Pick one.',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a', 'b'),
          ),
      ),
      (
          'choice_with_special_chars',
          {
              'call_to_action': 'Pick: type: free;;options: test',
              'output_type': 'choice',
              'options': ['a,b', 'c;;d', 'type: free'],
              'tag': None,
          },
          entity_lib.ActionSpec(
              call_to_action='Pick: type: free;;options: test',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a,b', 'c;;d', 'type: free'),
          ),
      ),
      (
          'skip',
          {
              'call_to_action': '',
              'output_type': 'skip_this_step',
              'options': [],
              'tag': None,
          },
          entity_lib.ActionSpec(
              call_to_action='',
              output_type=entity_lib.OutputType.SKIP_THIS_STEP,
          ),
      ),
  )
  def test_action_spec_parser_json(self, spec_dict, expected_spec):
    json_string = json.dumps(spec_dict)
    spec = engine.action_spec_parser(json_string)
    self.assertEqual(spec, expected_spec)

  @parameterized.named_parameters(
      (
          'free',
          entity_lib.ActionSpec(
              call_to_action='Describe the scene.',
              output_type=entity_lib.OutputType.FREE,
          ),
      ),
      (
          'choice',
          entity_lib.ActionSpec(
              call_to_action='Pick one.',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a', 'b'),
          ),
      ),
      (
          'choice_with_special_chars',
          entity_lib.ActionSpec(
              call_to_action='Pick: type: free;;options: test',
              output_type=entity_lib.OutputType.CHOICE,
              options=('a,b', 'c;;d'),
          ),
      ),
      (
          'skip',
          entity_lib.ActionSpec(
              call_to_action='Wait.',
              output_type=entity_lib.OutputType.SKIP_THIS_STEP,
          ),
      ),
  )
  def test_action_spec_to_string_roundtrip(self, action_spec):
    json_string = engine.action_spec_to_string(action_spec)
    parsed_spec = engine.action_spec_parser(json_string)
    self.assertEqual(parsed_spec, action_spec)

  def test_action_spec_to_string_outputs_json(self):
    action_spec = entity_lib.ActionSpec(
        call_to_action='Test',
        output_type=entity_lib.OutputType.FREE,
    )
    result = engine.action_spec_to_string(action_spec)
    parsed = json.loads(result)
    self.assertEqual(parsed['call_to_action'], 'Test')
    self.assertEqual(parsed['output_type'], 'free')


if __name__ == '__main__':
  absltest.main()

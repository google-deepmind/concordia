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

"""Tests for interrupt_response_parsing module."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.game_master import interrupt_response_parsing
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model


class ParseDurationSecondsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('0m', 0),
      ('0', 0),
      ('5m', 300),
      ('2h', 7200),
      ('1h30m', 5400),
      ('', 0),
  )
  def test_parse_duration_seconds(self, input_str, expected):
    self.assertEqual(
        interrupt_time_model.parse_duration_seconds(input_str),
        expected,
    )


class ParseEntityResponseTest(absltest.TestCase):

  def test_full_response(self):
    response = (
        'Alice says hello to everyone.\n'
        '{"mask": ["chat.", "alarm."], "timer":'
        ' {"time": "30m", "reason": "waiting for reply"}}'
    )
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.action_text, 'Alice says hello to everyone.')
    self.assertEqual(parsed.mask.prefixes, ('chat.', 'alarm.'))
    self.assertEqual(parsed.timer_duration_str, '30m')
    self.assertEqual(parsed.timer_description, 'waiting for reply')

  def test_missing_json_defaults_to_match_all(self):
    response = 'I do something.'
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.mask, interrupt_scheduling.MATCH_ALL)

  def test_missing_timer_defaults_to_none(self):
    response = 'I do something.\n{"mask": [""]}'
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertIsNone(parsed.timer_duration_str)
    self.assertEqual(parsed.timer_description, 'default timer')

  def test_empty_action_text(self):
    response = '{"mask": [], "timer": {"time": "2h", "reason": "deep focus"}}'
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.action_text, '')

  def test_match_none_mask(self):
    response = (
        'Focusing.\n'
        '{"mask": [], "timer":'
        ' {"time": "2h", "reason": "deep work"}}'
    )
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.mask, interrupt_scheduling.MATCH_NONE)

  def test_tags_parsed(self):
    response = (
        'I announce dinner!\n'
        '{"mask": [""], "tags": ["food.ready", "announcement."],'
        ' "timer": {"time": "30m", "reason": "cleanup"}}'
    )
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.event_tags, ('food.ready', 'announcement.'))
    self.assertEqual(parsed.action_text, 'I announce dinner!')

  def test_single_tag_parsed(self):
    response = (
        'Fire alarm!\n'
        '{"mask": [""], "tags": ["alert.fire"],'
        ' "timer": {"time": "0m", "reason": "evacuate"}}'
    )
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.event_tags, ('alert.fire',))

  def test_no_tags_defaults_to_empty(self):
    response = (
        'I do something.\n'
        '{"mask": [""], "timer":'
        ' {"time": "1h", "reason": "idle"}}'
    )
    parsed = interrupt_response_parsing.parse_entity_response(response)
    self.assertEqual(parsed.event_tags, ())


if __name__ == '__main__':
  absltest.main()

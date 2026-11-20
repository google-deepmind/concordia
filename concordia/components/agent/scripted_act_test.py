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

"""Tests name-prefix behavior for scripted actions with typed action specs."""

import json
import unittest

from concordia.agents import entity_agent
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import scripted_act
from concordia.environment import engine
from concordia.language_model import no_language_model
from concordia.typing import entity


class ScriptedActTest(unittest.TestCase):

  def _act(
      self,
      line: str,
      output_type: entity.OutputType,
      prefix_entity_name: bool = True,
  ) -> str:
    spec = engine.action_spec_parser(
        json.dumps({
            'call_to_action': 'Act now, {name}.',
            'output_type': output_type.value,
            'options': (
                (
                    ['Yes', 'No']
                    if output_type == entity.OutputType.TERMINATE
                    else ['left', 'right']
                )
                if output_type in entity.CHOICE_ACTION_TYPES
                else []
            ),
        })
    )
    policy = scripted_act.ScriptedActComponent(
        no_language_model.NoLanguageModel(),
        [{'name': 'Alice', 'line': line}],
        prefix_entity_name=prefix_entity_name,
    )
    agent = entity_agent.EntityAgent('Alice', policy)
    action = agent.act(spec)
    if output_type in entity.PLAYER_ACTION_TYPES:
      spec.validate(action)
    return action

  def test_choice_retains_the_scripted_option(self):
    self.assertEqual(self._act('left', entity.OutputType.CHOICE), 'left')

  def test_float_retains_the_scripted_number(self):
    self.assertEqual(self._act('1.25', entity.OutputType.FLOAT), '1.25')

  def test_make_observation_keeps_entity_prefix(self):
    self.assertEqual(
        self._act('waves', entity.OutputType.MAKE_OBSERVATION), 'Alice waves'
    )

  def test_next_action_spec_keeps_entity_prefix(self):
    self.assertEqual(
        self._act('waves', entity.OutputType.NEXT_ACTION_SPEC), 'Alice waves'
    )

  def test_resolve_keeps_entity_prefix(self):
    self.assertEqual(
        self._act('waves', entity.OutputType.RESOLVE), 'Alice waves'
    )

  def test_next_acting_keeps_the_scripted_choice(self):
    self.assertEqual(self._act('left', entity.OutputType.NEXT_ACTING), 'left')

  def test_terminate_keeps_the_scripted_choice(self):
    self.assertEqual(self._act('Yes', entity.OutputType.TERMINATE), 'Yes')

  def test_next_game_master_keeps_the_scripted_choice(self):
    self.assertEqual(
        self._act('left', entity.OutputType.NEXT_GAME_MASTER), 'left'
    )

  def test_script_advancement_exhaustion_and_training_log(self):
    policy = scripted_act.ScriptedActComponent(
        no_language_model.NoLanguageModel(),
        [{'name': 'Alice', 'line': 'waves'}, {'name': 'Alice', 'line': 'nods'}],
    )
    agent = entity_agent_with_logging.EntityAgentWithLogging('Alice', policy)
    self.assertEqual(agent.act(), 'Alice waves')
    self.assertEqual(policy.get_state()['line_index'], 1)
    self.assertEqual(agent.act(), 'Alice nods')
    self.assertEqual(policy.get_state()['line_index'], 2)
    self.assertEqual(agent.act(), '')
    self.assertEqual(policy.get_state()['line_index'], 2)
    logs = agent.get_all_logs()['__act__']
    self.assertEqual(
        [item['SFT datum']['target'] for item in logs], ['waves', 'nods']
    )

  def test_free_action_keeps_entity_prefix_by_default(self):
    self.assertEqual(self._act('waves', entity.OutputType.FREE), 'Alice waves')

  def test_free_action_can_disable_entity_prefix(self):
    self.assertEqual(self._act('waves', entity.OutputType.FREE, False), 'waves')

  def test_choice_can_disable_entity_prefix(self):
    self.assertEqual(self._act('left', entity.OutputType.CHOICE, False), 'left')

  def test_float_can_disable_entity_prefix(self):
    self.assertEqual(self._act('1.25', entity.OutputType.FLOAT, False), '1.25')


if __name__ == '__main__':
  unittest.main()

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

"""Tests of the rule-played acting component."""

from collections.abc import Mapping
import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import constant
from concordia.contrib.components.agent import rule_act
from concordia.typing import entity as entity_lib


def _entity(
    rule, name: str = 'Bot'
) -> entity_agent_with_logging.EntityAgentWithLogging:
  return entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=name,
      act_component=rule_act.RuleActComponent(rule),
      context_components={'Scene': constant.Constant('A door, shut.')},
  )


class Counter:
  """A rule with state: says how many times it has been asked."""

  def __init__(self):
    self.asked = 0

  def __call__(
      self, action_spec: entity_lib.ActionSpec, contexts: Mapping[str, str]
  ) -> str:
    del action_spec, contexts
    self.asked += 1
    return f'asked {self.asked} times'

  def get_state(self):
    return {'asked': self.asked}

  def set_state(self, state):
    self.asked = int(state['asked'])


class RuleActComponentTest(absltest.TestCase):

  def test_it_answers_by_the_rule_from_what_the_entity_knows(self):
    seen = []

    def rule(action_spec, contexts):
      seen.append((action_spec.call_to_action, dict(contexts)))
      return 'open the door'

    bot = _entity(rule)

    answer = bot.act(
        entity_lib.free_action_spec(call_to_action='What does {name} do?')
    )

    self.assertEqual(answer, 'open the door')
    ((question, contexts),) = seen
    self.assertEqual(question, 'What does Bot do?')
    self.assertIn('A door, shut.', contexts['Scene'])

  def test_a_choice_not_offered_is_refused(self):
    bot = _entity(lambda action_spec, contexts: 'fly')
    spec = entity_lib.choice_action_spec(
        call_to_action='Which way?', options=('left', 'right')
    )

    with self.assertRaises(ValueError):
      bot.act(spec)
    self.assertEqual(
        _entity(lambda action_spec, contexts: 'left').act(spec), 'left'
    )

  def test_a_rule_with_state_is_restored_with_its_entity(self):
    counter = Counter()
    bot = _entity(counter)
    question = entity_lib.free_action_spec(call_to_action='Again?')
    bot.act(question)
    bot.act(question)
    saved = json.loads(json.dumps(bot.get_state()))

    restored = _entity(Counter())
    restored.set_state(saved)

    self.assertEqual(restored.act(question), 'asked 3 times')

  def test_a_rule_without_state_saves_nothing(self):
    act = rule_act.RuleActComponent(lambda action_spec, contexts: 'wait')
    self.assertEqual(act.get_state(), {})
    act.set_state({})


if __name__ == '__main__':
  absltest.main()

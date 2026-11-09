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

"""Tests logging through a question component that hides its pre-act output."""

from collections.abc import Mapping
from typing import Any
import unittest

from concordia.agents import entity_agent
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat
from concordia.components.agent import memory as memory_component
from concordia.components.agent import question_of_recent_memories as questions
from concordia.language_model import no_language_model
from concordia.typing import entity
from concordia.typing import entity_component


class _ContextAct(entity_component.ActingComponent):

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity.ActionSpec,
  ) -> str:
    del action_spec
    return '|'.join(value for value in context.values() if value)

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _question(
    wrapped: bool,
) -> (
    questions.QuestionOfRecentMemories
    | questions.QuestionOfRecentMemoriesWithoutPreAct
):
  cls = (
      questions.QuestionOfRecentMemoriesWithoutPreAct
      if wrapped
      else questions.QuestionOfRecentMemories
  )
  return cls(
      model=no_language_model.NoLanguageModel(),
      pre_act_label='Reflection',
      question='What should {agent_name} remember?',
      answer_prefix='remembered',
      add_to_memory=False,
      num_memories_to_retrieve=0,
  )


def _contexts(
    reflection: entity_component.ContextComponent, dependent: bool = True
) -> dict[str, entity_component.ContextComponent]:
  contexts: dict[str, entity_component.ContextComponent] = {
      '__memory__': memory_component.ListMemory([]),
      'reflection': reflection,
  }
  if dependent:
    contexts['display'] = concat.Concatenate(
        ['reflection'], pre_act_label='Display'
    )
  return contexts


class QuestionLoggingTest(unittest.TestCase):

  def test_wrapped_question_logs_once_when_requested_by_dependency(self):
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        'Alice', _ContextAct(), _contexts(_question(wrapped=True))
    )
    action = agent.act()
    self.assertEqual(action, 'Display:\nReflection:\nremembered\n')
    self.assertEqual(agent.get_last_log()['reflection']['State'], 'remembered')
    self.assertEqual(len(agent.get_all_logs()['reflection']), 1)
    agent.act()
    self.assertEqual(len(agent.get_all_logs()['reflection']), 2)

  def test_replacing_logging_channel_replaces_inner_callback(self):
    reflection = _question(wrapped=True)
    first: list[Mapping[str, Any]] = []
    second: list[Mapping[str, Any]] = []
    reflection.set_logging_channel(first.append)
    agent = entity_agent.EntityAgent(
        'Alice', _ContextAct(), _contexts(reflection)
    )
    agent.act()
    self.assertEqual(len(first), 1)
    reflection.set_logging_channel(second.append)
    agent.act()
    self.assertEqual(len(first), 1)
    self.assertEqual(len(second), 1)
    self.assertEqual(second[0]['State'], 'remembered')

  def test_derived_self_perception_keeps_its_logging_channel(self):
    reflection = questions.SelfPerceptionWithoutPreAct(
        model=no_language_model.NoLanguageModel(), num_memories_to_retrieve=0
    )
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        'Alice', _ContextAct(), _contexts(reflection)
    )
    agent.act()
    self.assertEqual(agent.get_last_log()['reflection']['State'], 'Alice is ')
    self.assertEqual(len(agent.get_all_logs()['reflection']), 1)

  def test_unused_wrapper_remains_hidden_and_uncomputed(self):
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        'Alice',
        _ContextAct(),
        _contexts(_question(wrapped=True), dependent=False),
    )
    self.assertEqual(agent.act(), '')
    self.assertEqual(agent.get_last_log(), {})

  def test_unwrapped_question_still_logs_once(self):
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        'Alice', _ContextAct(), _contexts(_question(wrapped=False))
    )
    agent.act()
    self.assertEqual(agent.get_last_log()['reflection']['State'], 'remembered')
    self.assertEqual(len(agent.get_all_logs()['reflection']), 1)


if __name__ == '__main__':
  unittest.main()

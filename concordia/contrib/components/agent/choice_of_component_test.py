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

"""Tests the label interface of a hidden component choice."""

from collections.abc import Mapping, Sequence
from typing import Any
import unittest

from concordia.agents import entity_agent
from concordia.components.agent import concat
from concordia.components.agent import constant
from concordia.contrib.components.agent import choice_of_component
from concordia.language_model import no_language_model
from concordia.typing import entity
from concordia.typing import entity_component


class _RecordingModel(no_language_model.NoLanguageModel):

  def __init__(self):
    super().__init__()
    self.choice_calls = 0

  def sample_choice(
      self, prompt: str, responses: Sequence[str], **kwargs: Any
  ) -> tuple[int, str, Mapping[str, Any]]:
    self.choice_calls += 1
    return super().sample_choice(prompt, responses, **kwargs)


class _Act(entity_component.ActingComponent):

  def __init__(self):
    self.context: dict[str, str] = {}

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity.ActionSpec,
  ) -> str:
    del action_spec
    self.context = dict(context)
    return context.get('consumer', '')

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _agent(
    choice: entity_component.ContextComponent, dependent: bool
) -> tuple[entity_agent.EntityAgent, _Act]:
  contexts: dict[str, entity_component.ContextComponent] = {
      'obs': constant.Constant('OBSERVATION'),
      'selected': constant.Constant('SELECTED'),
      'other': constant.Constant('OTHER'),
      'choice': choice,
  }
  if dependent:
    contexts['consumer'] = concat.Concatenate(
        ['choice'], pre_act_label='Consumer'
    )
  policy = _Act()
  return entity_agent.EntityAgent('Alice', policy, contexts), policy


class ChoiceOfComponentTest(unittest.TestCase):

  def test_hidden_choice_exposes_default_label_before_binding(self):
    model = _RecordingModel()
    choice = choice_of_component.ChoiceOfComponentWithoutPreAct(
        model, 'obs', ['selected']
    )
    self.assertEqual(choice.get_pre_act_label(), 'ChoiceComponent')
    self.assertEqual(model.choice_calls, 0)

  def test_hidden_choice_exposes_custom_label_before_binding(self):
    choice = choice_of_component.ChoiceOfComponentWithoutPreAct(
        _RecordingModel(), 'obs', ['selected'], pre_act_label='Choice view'
    )
    self.assertEqual(choice.get_pre_act_label(), 'Choice view')

  def test_stock_dependency_can_read_the_selected_value_and_label(self):
    model = _RecordingModel()
    choice = choice_of_component.ChoiceOfComponentWithoutPreAct(
        model, 'obs', ['selected', 'other'], pre_act_label='Choice view'
    )
    agent, policy = _agent(choice, dependent=True)
    self.assertIn(
        agent.act(),
        (
            'Consumer:\nChoice view:\nSELECTED\n',
            'Consumer:\nChoice view:\nOTHER\n',
        ),
    )
    self.assertEqual(model.choice_calls, 1)
    self.assertEqual(policy.context['choice'], '')
    self.assertEqual(choice.get_pre_act_label(), 'Choice view')

  def test_unused_wrapper_stays_hidden_without_model_sampling(self):
    model = _RecordingModel()
    choice = choice_of_component.ChoiceOfComponentWithoutPreAct(
        model, 'obs', ['selected', 'other'], pre_act_label='Choice view'
    )
    agent, policy = _agent(choice, dependent=False)
    self.assertEqual(agent.act(), '')
    self.assertEqual(policy.context['choice'], '')
    self.assertEqual(model.choice_calls, 0)

  def test_unwrapped_choice_keeps_its_existing_dependency_output(self):
    choice = choice_of_component.ChoiceOfComponent(
        _RecordingModel(), 'obs', ['selected'], pre_act_label='Choice view'
    )
    agent, _ = _agent(choice, dependent=True)
    self.assertEqual(agent.act(), 'Consumer:\nChoice view:\nSELECTED\n')
    self.assertEqual(choice.get_pre_act_label(), 'Choice view')


if __name__ == '__main__':
  unittest.main()

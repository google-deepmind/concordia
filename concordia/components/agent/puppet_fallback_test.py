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

"""Tests name substitution in PuppetAct's model fallback prompts."""

from collections.abc import Mapping, Sequence
from typing import Any
import unittest

from concordia.agents import entity_agent
from concordia.components.agent import puppet_act
from concordia.language_model import no_language_model
from concordia.typing import entity


class _RecordingModel(no_language_model.NoLanguageModel):

  def __init__(self):
    super().__init__()
    self.prompts: list[str] = []

  def sample_text(self, prompt: str, **kwargs: Any) -> str:
    del kwargs
    self.prompts.append(prompt)
    return '1.25'

  def sample_choice(
      self, prompt: str, responses: Sequence[str], **kwargs: Any
  ) -> tuple[int, str, Mapping[str, Any]]:
    del kwargs
    self.prompts.append(prompt)
    return 0, responses[0], {}


class PuppetFallbackTest(unittest.TestCase):

  def _act(
      self,
      output_type: entity.OutputType,
      question: str = 'What should {name} do?',
  ) -> tuple[str, list[str]]:
    model = _RecordingModel()
    component = puppet_act.PuppetActComponent(model, fixed_responses={})
    agent = entity_agent.EntityAgent('Alice', component)
    spec = entity.ActionSpec(
        call_to_action=question,
        output_type=output_type,
        options=('wait', 'leave')
        if output_type == entity.OutputType.CHOICE
        else (),
    )
    result = agent.act(spec)
    return result, model.prompts

  def test_free_fallback_substitutes_name(self):
    result, prompts = self._act(entity.OutputType.FREE)
    self.assertEqual(result, 'Alice 1.25')
    self.assertEqual(len(prompts), 1)
    self.assertIn('What should Alice do?', prompts[0])
    self.assertNotIn('{name}', prompts[0])

  def test_choice_fallback_substitutes_name(self):
    result, prompts = self._act(entity.OutputType.CHOICE)
    self.assertIn(result, ('wait', 'leave'))
    self.assertEqual(len(prompts), 1)
    self.assertIn('What should Alice do?', prompts[0])
    self.assertNotIn('{name}', prompts[0])

  def test_float_fallback_substitutes_name(self):
    result, prompts = self._act(entity.OutputType.FLOAT)
    self.assertEqual(result, '1.25')
    self.assertEqual(len(prompts), 1)
    self.assertIn('What should Alice do?', prompts[0])
    self.assertNotIn('{name}', prompts[0])

  def test_fixed_response_still_avoids_model(self):
    model = _RecordingModel()
    component = puppet_act.PuppetActComponent(
        model, fixed_responses={'What should {name} do?': 'fixed'}
    )
    agent = entity_agent.EntityAgent('Alice', component)
    result = agent.act(
        entity.ActionSpec(
            call_to_action='What should {name} do?',
            output_type=entity.OutputType.FREE,
        )
    )
    self.assertEqual(result, 'fixed')
    self.assertEqual(model.prompts, [])

  def test_unknown_placeholder_keeps_existing_fallback_behavior(self):
    _, prompts = self._act(entity.OutputType.FREE, 'What should {other} do?')
    self.assertIn('What should {other} do?', prompts[0])

  def test_question_without_placeholder_is_unchanged(self):
    _, prompts = self._act(entity.OutputType.FREE, 'Pick an action.')
    self.assertIn('Pick an action.', prompts[0])


if __name__ == '__main__':
  unittest.main()

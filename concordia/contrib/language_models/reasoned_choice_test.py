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

"""Tests for the reasoned-choice wrapper."""

from collections.abc import Sequence

from absl.testing import absltest
from concordia.contrib.language_models import reasoned_choice
from concordia.document import interactive_document
from concordia.language_model import no_language_model


class _Fake(no_language_model.NoLanguageModel):
  """Replies with fixed text; records prompts; bare choice picks the last."""

  def __init__(self, reply: str):
    super().__init__()
    self.reply = reply
    self.prompts: list[str] = []
    self.temperatures: list[float | None] = []
    self.choices = 0

  def sample_text(self, prompt: str, **kwargs) -> str:
    self.prompts.append(prompt)
    self.temperatures.append(kwargs.get('temperature'))
    return self.reply

  def sample_choice(self, prompt: str, responses: Sequence[str], **kwargs):
    del prompt, kwargs
    self.choices += 1
    return len(responses) - 1, responses[-1], {}


class ReasonedChoiceTest(absltest.TestCase):

  def test_choice_comes_from_the_reasoned_reply(self):
    fake = _Fake('Cy follows the order to build, so (c).')
    model = reasoned_choice.ReasonedChoice(fake)
    doc = interactive_document.InteractiveDocument(model)
    doc.statement('The order is: build.')
    idx = doc.multiple_choice_question(
        'What does Cy do?', ['repair', 'gather', 'build', 'rest'],
        randomize_choices=False,
    )  # fmt: skip
    self.assertEqual(idx, 2)
    self.assertEqual(fake.choices, 0)
    # The bare-answer tail is replaced by the instruction.
    self.assertTrue(
        fake.prompts[0].endswith(reasoned_choice.DEFAULT_INSTRUCTION)
    )
    self.assertNotIn('Answer: (', fake.prompts[0])

  def test_the_last_option_letter_wins(self):
    fake = _Fake('Not (a), and not (z) either; the answer is (b).')
    model = reasoned_choice.ReasonedChoice(fake)
    idx, response, debug = model.sample_choice('Q?\nAnswer: (', ['a', 'b'])
    self.assertEqual((idx, response), (1, 'b'))
    self.assertIn('reasoning', debug)

  def test_falls_back_to_the_bare_choice(self):
    fake = _Fake('I cannot decide.')
    model = reasoned_choice.ReasonedChoice(fake)
    idx, _, _ = model.sample_choice('Q?\nAnswer: (', ['a', 'b', 'c'])
    self.assertEqual((idx, fake.choices), (2, 1))

  def test_temperature_reaches_the_reasoned_reply(self):
    fake = _Fake('So (a).')
    model = reasoned_choice.ReasonedChoice(fake, temperature=0.0)
    model.sample_choice('Q?\nAnswer: (', ['a', 'b'])
    self.assertEqual(fake.temperatures, [0.0])

  def test_text_passes_through(self):
    fake = _Fake('hello')
    self.assertEqual(
        reasoned_choice.ReasonedChoice(fake).sample_text('hi'), 'hello'
    )


if __name__ == '__main__':
  absltest.main()

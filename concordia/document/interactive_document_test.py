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


import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.document import document
from concordia.document import interactive_document
from concordia.language_model import language_model
import numpy as np


DEBUG = functools.partial(document.Content, tags=frozenset({'debug'}))
STATEMENT = functools.partial(document.Content, tags=frozenset({'statement'}))
QUESTION = functools.partial(document.Content, tags=frozenset({'question'}))
RESPONSE = functools.partial(document.Content, tags=frozenset({'response'}))
MODEL_RESPONSE = functools.partial(
    document.Content, tags=frozenset({'response', 'model'})
)


class InteractiveDocumentTest(parameterized.TestCase):

  def test_open_question(self):
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.return_value = 'This is a long answer'

    doc = interactive_document.InteractiveDocument(model)
    doc.statement('Hello')
    response = doc.open_question(
        question='What is 1+1?',
        answer_prefix='Well...',
        max_tokens=mock.sentinel.max_tokens,
        terminators=mock.sentinel.terminators,
    )

    with self.subTest('response'):
      self.assertEqual(response, 'This is a long answer')

    with self.subTest('model'):
      prompt = """Hello
Question: What is 1+1?
Answer: Well..."""
      model.sample_text.assert_called_once_with(
          prompt=prompt,
          max_tokens=mock.sentinel.max_tokens,
          terminators=mock.sentinel.terminators,
      )

    with self.subTest('text'):
      expected = """Hello
Question: What is 1+1?
Answer: Well...This is a long answer
"""
      self.assertEqual(doc.text(), expected)

    with self.subTest('contents'):
      expected = (
          STATEMENT('Hello\n'),
          QUESTION('Question: What is 1+1?\n'),
          RESPONSE('Answer: Well...'),
          MODEL_RESPONSE('This is a long answer'),
          RESPONSE('\n'),
      )
      self.assertEqual(doc.contents(), expected)

  def test_open_question_with_forced_answer(self):
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_text.return_value = 'This is a long answer'

    doc = interactive_document.InteractiveDocument(model)
    doc.statement('Hi!')
    response = doc.open_question(
        question='What is 1+1?',
        forced_response='I hereby declare the answer to be 7',
        answer_prefix='OK then...',
    )

    with self.subTest('response'):
      self.assertEqual(response, 'I hereby declare the answer to be 7')

    with self.subTest('text'):
      expected = """Hi!
Question: What is 1+1?
Answer: OK then...I hereby declare the answer to be 7
"""
      self.assertEqual(doc.text(), expected)

    with self.subTest('contents'):
      expected = (
          STATEMENT('Hi!\n'),
          QUESTION('Question: What is 1+1?\n'),
          RESPONSE('Answer: OK then...'),
          MODEL_RESPONSE('I hereby declare the answer to be 7'),
          RESPONSE('\n'),
      )
      self.assertEqual(doc.contents(), expected)

  def test_multiple_choice_question(self):
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_choice.return_value = (2, 'c', mock.sentinel.debug)
    rng = mock.create_autospec(
        np.random.Generator, instance=True, spec_set=True
    )
    rng.permutation.return_value = np.arange(3)[::-1]

    doc = interactive_document.InteractiveDocument(model, rng=rng)
    doc.statement('Hello')
    response = doc.multiple_choice_question(
        question='What is 1+1?',
        answers=['1', '2', '3'],
    )

    with self.subTest('response'):
      self.assertEqual(response, 0)

    with self.subTest('model'):
      prompt = """Hello
Question: What is 1+1?
  (a) 3
  (b) 2
  (c) 1
Answer: ("""
      model.sample_choice.assert_called_once_with(
          prompt=prompt, responses=['a', 'b', 'c']
      )

    with self.subTest('text'):
      expected = """Hello
Question: What is 1+1?
  (a) 3
  (b) 2
  (c) 1
Answer: (c)
[sentinel.debug]
"""
      self.assertEqual(doc.text(), expected)

    with self.subTest('contents'):
      expected = (
          STATEMENT('Hello\n'),
          QUESTION('Question: What is 1+1?\n'),
          QUESTION('  (a) 3\n'),
          QUESTION('  (b) 2\n'),
          QUESTION('  (c) 1\n'),
          RESPONSE('Answer: ('),
          MODEL_RESPONSE('c'),
          RESPONSE(')\n'),
          DEBUG('[sentinel.debug]\n'),
      )
      self.assertSequenceEqual(doc.contents(), expected)

  def test_debug_hidden_from_default_view(self):
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )
    model.sample_choice.return_value = (2, 'c', mock.sentinel.debug)
    rng = mock.create_autospec(
        np.random.Generator, instance=True, spec_set=True
    )
    rng.permutation.return_value = np.arange(3)[::-1]

    doc = interactive_document.InteractiveDocument(model, rng=rng)
    doc.statement('Hello')
    doc.multiple_choice_question(
        question='What is 1+1?',
        answers=['1', '2', '3'],
    )

    with self.subTest('view'):
      expected = """Hello
Question: What is 1+1?
  (a) 3
  (b) 2
  (c) 1
Answer: (c)
"""
      self.assertEqual(doc.view().text(), expected)

  def test_yes_no_question_answer_yes(self):
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )

    rng = mock.create_autospec(
        np.random.Generator, instance=True, spec_set=True
    )
    rng.permutation.return_value = [0, 1]

    doc = interactive_document.InteractiveDocument(model, rng=rng)
    doc.statement('Hello')
    model.sample_choice.return_value = (1, 'b', mock.sentinel.debug)
    response = doc.yes_no_question(question='Does 1+1 equal 2?')
    self.assertTrue(response)

  def test_yes_no_question_answer_no(self):
    model = mock.create_autospec(
        language_model.LanguageModel, instance=True, spec_set=True
    )

    rng = mock.create_autospec(
        np.random.Generator, instance=True, spec_set=True
    )
    rng.permutation.return_value = [0, 1]

    doc = interactive_document.InteractiveDocument(model, rng=rng)
    doc.statement('Hello')
    model.sample_choice.return_value = (0, 'a', mock.sentinel.debug)
    response = doc.yes_no_question(question='Does 1+1 equal 3?')
    self.assertFalse(response)


if __name__ == '__main__':
  absltest.main()

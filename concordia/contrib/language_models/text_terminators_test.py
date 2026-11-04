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

"""Text terminator contracts with scripted completions, no provider calls."""

import types
from typing import cast
from unittest import mock

from concordia.document import interactive_document
import pytest


@pytest.fixture(params=['gpt', 'groq'])
def model_and_client(request):
  client = types.SimpleNamespace(
      chat=types.SimpleNamespace(
          completions=types.SimpleNamespace(create=mock.Mock())
      )
  )
  if request.param == 'gpt':
    pytest.importorskip('openai')
    from concordia.contrib.language_models.openai import base_gpt_model

    model = base_gpt_model.BaseGPTModel(
        'gpt-5', client=cast(base_gpt_model.OpenAI, client)
    )
  else:
    pytest.importorskip('groq')
    from concordia.contrib.language_models.groq import groq_model

    with mock.patch.object(groq_model, 'Groq', return_value=client):
      model = groq_model.GroqModel('fixture-model', api_key='fixture')
  return model, client


def completion(client, text):
  client.chat.completions.create.return_value = types.SimpleNamespace(
      choices=[
          types.SimpleNamespace(message=types.SimpleNamespace(content=text))
      ]
  )


@pytest.mark.parametrize(
    'text,terminators,expected',
    [
        ('First.\nQuestion: next?', ('\n',), 'First.'),
        ('first STOP second END third', ('END', 'STOP'), 'first '),
        ('first\nsecond', (), 'first\nsecond'),
        ('complete', ('STOP',), 'complete'),
        ('STOP extra', ('STOP',), ''),
        ('first<END>second', ('<END>',), 'first'),
    ],
)
def test_sample_text_honors_terminators(
    model_and_client, text, terminators, expected
):
  model, client = model_and_client
  completion(client, text)
  assert model.sample_text('Continue', terminators=terminators) == expected
  client.chat.completions.create.assert_called_once()
  # Truncate locally, without adding a provider-specific stop parameter.
  assert 'stop' not in client.chat.completions.create.call_args.kwargs


def test_document_default_newline_does_not_record_another_question(
    model_and_client,
):
  model, client = model_and_client
  completion(client, 'Alice waits.\nQuestion: What happens next?')
  doc = interactive_document.InteractiveDocument(model)
  assert doc.open_question('What does Alice do?') == 'Alice waits.'
  assert 'What happens next?' not in doc.view().text()
  client.chat.completions.create.assert_called_once()


def test_measurement_retains_raw_completion_length(model_and_client):
  model, client = model_and_client
  text = 'Alice waits.\nQuestion: What happens next?'
  completion(client, text)
  measurement = mock.Mock()
  model._measurements = measurement
  assert model.sample_text('Continue', terminators=('\n',)) == 'Alice waits.'
  measurement.publish_datum.assert_called_once_with(
      model._channel, {'raw_text_length': len(text)}
  )


def test_choice_uses_unmodified_raw_response(model_and_client):
  model, client = model_and_client
  completion(client, 'first\nsecond')
  assert model.sample_choice('Which?', ['other', 'first\nsecond'], seed=17) == (
      1,
      'first\nsecond',
      {},
  )
  client.chat.completions.create.assert_called_once()
  assert client.chat.completions.create.call_args.kwargs['seed'] == 17

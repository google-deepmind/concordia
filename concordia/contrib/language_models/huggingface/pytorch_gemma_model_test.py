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

"""Gemma wrapper text contracts with scripted generation, no model loading."""

import types
from typing import Any, cast
from unittest import mock

from concordia.document import interactive_document
import numpy as np
import pytest


@pytest.fixture
def model(monkeypatch):
  monkeypatch.setenv('HF_HUB_OFFLINE', '1')
  pytest.importorskip('transformers')
  from concordia.contrib.language_models.huggingface import pytorch_gemma_model
  # Bypass only the model-loading constructor. Methods under test stay real;
  # the tokenizer and generation objects below are explicit backend doubles.
  model = cast(
      Any, object.__new__(pytorch_gemma_model.PyTorchGemmaLanguageModel)
  )
  model._device = 'cpu'
  model._measurements = None
  model._channel = 'fixture'
  model._text_system_message = 'Synthetic system'
  model._tokenizer = mock.Mock(
      return_value=types.SimpleNamespace(
          input_ids=types.SimpleNamespace(to=lambda _device: 'input-ids')
      )
  )
  sequence = types.SimpleNamespace(cpu=lambda: np.array([1, 2, 3]))
  score = types.SimpleNamespace(cpu=lambda: np.array([-2.0, -0.1]))
  model._model = types.SimpleNamespace(
      generate=mock.Mock(
          return_value=types.SimpleNamespace(
              sequences=[sequence], scores=[[score]]
          )
      )
  )
  return model


def completion(model, text):
  model._tokenizer.decode.side_effect = lambda *_args, **_kwargs: (
      model._tokenizer.call_args.args[0] + text
  )


@pytest.mark.parametrize(
    'text,terminators,expected',
    [
        ('Complete.', ('\n',), 'Complete.'),
        ('First STOP second', ('STOP',), 'First '),
        ('First STOP second', ('STOP', 'absent'), 'First '),
        ('First STOP second END third', ('END', 'STOP'), 'First '),
        ('Complete.', (), 'Complete.'),
        ('STOP extra', ('STOP',), ''),
    ],
)
def test_terminators_keep_characters_when_absent(
    model, text, terminators, expected
):
  completion(model, text)
  assert model.sample_text('Continue', terminators=terminators) == expected
  model._model.generate.assert_called_once()


def test_document_default_newline_keeps_complete_answer(model):
  completion(model, 'Complete.')
  doc = interactive_document.InteractiveDocument(model)
  assert doc.open_question('Continue') == 'Complete.'
  assert 'Answer: Complete.\n' in doc.text()


def test_existing_measurement_records_returned_text_length(model):
  completion(model, 'First STOP second')
  measurement = mock.Mock()
  model._measurements = measurement
  assert model.sample_text('Continue', terminators=('STOP',)) == 'First '
  measurement.publish_datum.assert_called_once_with(
      'fixture', {'raw_text_length': len('First ')}
  )


def test_choice_scoring_path_is_unchanged(model):
  model._tokenizer.batch_decode.return_value = ['b']
  assert model.sample_choice('Choose', ['a', 'b']) == (1, 'b', {})
  assert model._model.generate.call_args.kwargs['max_new_tokens'] == 1
  model._tokenizer.decode.assert_not_called()


def test_generation_arguments_are_unchanged(model):
  completion(model, 'Complete.')
  assert (
      model.sample_text(
          'Continue',
          terminators=(),
          max_tokens=17,
          temperature=0.7,
          top_p=0.3,
          top_k=2,
      )
      == 'Complete.'
  )
  model._model.generate.assert_called_once_with(
      'input-ids',
      max_new_tokens=17,
      temperature=0.7,
      top_p=0.3,
      top_k=2,
      return_dict_in_generate=True,
      output_scores=True,
  )

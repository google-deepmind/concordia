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

"""Bounded local generation request checks; no model calls."""

from unittest import mock

from concordia.contrib.language_models.ollama import ollama_model
import pytest


@pytest.mark.parametrize(
    'bound,requested,expected',
    [(None, 25, 25), (256, 2200, 256), (256, 12, 12)],
)
def test_generation_honors_requested_and_configured_limits(
    bound, requested, expected
):
  with mock.patch.object(ollama_model.ollama, 'Client') as client:
    client.return_value.generate.return_value = {'response': 'local response'}
    model = ollama_model.OllamaLanguageModel(
        'local-model', request_timeout=90, max_output_tokens=bound
    )
    assert model.sample_text('prompt', max_tokens=requested) == 'local response'
    client.assert_called_once_with(timeout=90)
    assert (
        client.return_value.generate.call_args.kwargs['options']['num_predict']
        == expected
    )


def test_nonpositive_limit_rejected():
  with mock.patch.object(ollama_model.ollama, 'Client'):
    with pytest.raises(ValueError, match='positive'):
      ollama_model.OllamaLanguageModel('local-model', max_output_tokens=0)


@pytest.mark.parametrize('bound', [None, 256])
def test_choice_generation_obeys_optional_limit(bound):
  with mock.patch.object(ollama_model.ollama, 'Client') as client:
    client.return_value.generate.return_value = {'response': '{"choice": "a"}'}
    model = ollama_model.OllamaLanguageModel(
        'local-model', max_output_tokens=bound
    )
    assert model.sample_choice('prompt', ['a', 'b'])[:2] == (0, 'a')
    options = client.return_value.generate.call_args.kwargs['options']
    if bound is None:
      assert 'num_predict' not in options
    else:
      assert options['num_predict'] == bound


@pytest.mark.parametrize('response_format', [None, 'json', {'type': 'object'}])
def test_text_format_is_opt_in_and_choice_keeps_its_own_json_contract(
    response_format,
):
  with mock.patch.object(ollama_model.ollama, 'Client') as client:
    client.return_value.generate.return_value = {'response': '{"choice":"a"}'}
    model = ollama_model.OllamaLanguageModel(
        'local-model', response_format=response_format
    )
    model.sample_text('prompt')
    args = client.return_value.generate.call_args.kwargs
    if response_format is None:
      assert 'format' not in args
    else:
      assert args['format'] == response_format
    assert model.sample_choice('choice prompt', ['a', 'b'])[:2] == (0, 'a')
    assert client.return_value.generate.call_args.kwargs['format'] == 'json'


def test_caller_schema_mutation_cannot_change_subsequent_requests():
  with mock.patch.object(ollama_model.ollama, 'Client') as client:
    client.return_value.generate.return_value = {'response': '{}'}
    schema = {'type': 'object', 'required': ['answer']}
    model = ollama_model.OllamaLanguageModel(
        'local-model', response_format=schema
    )
    schema['required'].clear()
    model.sample_text('prompt')
    assert client.return_value.generate.call_args.kwargs['format'] == {
        'type': 'object',
        'required': ['answer'],
    }


def test_unknown_text_format_rejected_before_client_creation():
  with mock.patch.object(ollama_model.ollama, 'Client') as client:
    with pytest.raises(ValueError, match='response_format'):
      ollama_model.OllamaLanguageModel('local-model', response_format='xml')
    client.assert_not_called()

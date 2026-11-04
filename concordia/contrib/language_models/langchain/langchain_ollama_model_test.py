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

"""Registered Langchain adapter with the real SDK and scripted HTTP replies."""

import importlib
import json
import socket
from unittest import mock

from concordia.document import interactive_document
import pytest


@pytest.fixture
def runtime(monkeypatch):
  monkeypatch.setenv('LANGSMITH_TRACING', 'false')
  monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
  pytest.importorskip('langchain_community')
  sdk = importlib.import_module('langchain_community.llms.ollama')
  adapter = importlib.import_module(
      'concordia.contrib.language_models.langchain.langchain_ollama_model'
  )
  content = {'text': 'Complete.'}

  def reply(**kwargs):
    assert kwargs['stream'] is True
    response = sdk.requests.Response()
    response.status_code = 200
    # A complete, local streaming body for the SDK's own JSONL reader.
    response._content = (
        json.dumps({
            'response': content['text'],
            'done': True,
        })
        + '\n'
    ).encode()
    response._content_consumed = True
    response.encoding = 'utf-8'
    return response

  post = mock.Mock(side_effect=reply)
  monkeypatch.setattr(sdk.requests, 'post', post)

  def reject_socket(*_args, **_kwargs):
    raise AssertionError('Only scripted HTTP is permitted in this test')

  monkeypatch.setattr(socket.socket, 'connect', reject_socket)
  return adapter.LangchainOllamaLanguageModel, content, post


def test_public_sampling_uses_supported_sdk_entrypoint(runtime):
  model_type, _, post = runtime
  model = model_type('fixture-model')
  assert model.sample_text('Continue', terminators=()) == 'Complete.'
  assert post.call_count == 1
  payload = post.call_args.kwargs['json']
  assert payload['model'] == 'fixture-model'
  assert payload['options']['stop'] == []


@pytest.mark.parametrize(
    'raw,stops,expected',
    [
        ('First STOP second', ['STOP'], 'First '),
        ('First STOP second END third', ['END', 'STOP'], 'First '),
        ('First\nsecond', [], 'First\nsecond'),
        ('Complete.', ['absent'], 'Complete.'),
        ('STOP extra', ['STOP'], ''),
    ],
)
def test_requested_stops_reach_sdk_and_bound_return(
    runtime, raw, stops, expected
):
  model_type, content, post = runtime
  content['text'] = raw
  model = model_type('fixture-model')
  assert model.sample_text('Continue', terminators=stops) == expected
  assert post.call_args.kwargs['json']['options']['stop'] == stops


def test_document_default_newline_is_respected(runtime):
  model_type, content, post = runtime
  content['text'] = 'Complete.\nQuestion: next?'
  doc = interactive_document.InteractiveDocument(model_type('fixture-model'))
  assert doc.open_question('Continue') == 'Complete.'
  assert 'Question: next?' not in doc.text()
  assert post.call_args.kwargs['json']['options']['stop'] == ['\n']


def test_stops_do_not_accumulate_across_calls(runtime):
  model_type, content, post = runtime
  model = model_type('llama3-fixture')
  stops = ['STOP']
  content['text'] = 'Complete.'
  model.sample_text('First', terminators=stops)
  model.sample_text('Second', terminators=['END'])
  content['text'] = 'First STOP second END third'
  assert model.sample_text('Third', terminators=()) == content['text']
  assert stops == ['STOP']
  assert [
      call.kwargs['json']['options']['stop'] for call in post.call_args_list
  ] == [
      ['<|eot_id|>', 'STOP'],
      ['<|eot_id|>', 'END'],
      ['<|eot_id|>'],
  ]


def test_builtin_stop_and_raw_length_measurement_are_preserved(runtime):
  model_type, content, post = runtime
  content['text'] = 'Complete.<|eot_id|>trailing'
  measurement = mock.Mock()
  model = model_type('llama3-fixture', measurements=measurement)
  assert model.sample_text('Continue') == 'Complete.'
  assert post.call_args.kwargs['json']['options']['stop'] == ['<|eot_id|>']
  measurement.publish_datum.assert_called_once_with(
      model._channel, {'raw_text_length': len(content['text'])}
  )


def test_sampling_options_and_choice_path_are_preserved(runtime):
  model_type, content, post = runtime
  model = model_type('fixture-model')
  model.sample_text('Continue', temperature=0.2, top_p=0.4)
  options = post.call_args.kwargs['json']['options']
  assert options['temperature'] == 0.2 and options['top_p'] == 0.4
  content['text'] = 'b'
  assert model.sample_choice('Choose', ['a', 'b']) == (1, 'b', {})
  assert post.call_count == 2

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

"""Tests how the Gemini wrappers turn API responses into text.

The Gemini client is replaced by a mock, so no API key or network is needed.
"""

import importlib
from typing import Any
from unittest import mock

import pytest

genai: Any = None
types: Any = None
gemini_model: Any = None
gemini_model_multimodal: Any = None


@pytest.fixture(autouse=True)
def _load_modules():
  """Skips these tests when the optional google-genai package is missing."""
  global genai, types, gemini_model, gemini_model_multimodal
  pytest.importorskip('google.genai')
  if gemini_model is None:
    genai = importlib.import_module('google.genai')
    types = importlib.import_module('google.genai.types')
    gemini_model = importlib.import_module(
        'concordia.contrib.language_models.google.gemini_model'
    )
    gemini_model_multimodal = importlib.import_module(
        'concordia.contrib.language_models.google.gemini_model_multimodal'
    )


def _response_with_parts(parts):
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(content=types.Content(role='model', parts=parts))
      ]
  )


def _text_response(text):
  return _response_with_parts([types.Part(text=text)])


def _make_text_model(response):
  """Returns a GeminiModel whose chat API replies with `response`."""
  with mock.patch.object(genai, 'Client'):
    model = gemini_model.GeminiModel(model_name='test-model', api_key='key')
  chat = model._client.chats.create.return_value  # pylint: disable=protected-access
  chat.send_message.return_value = response
  return model


def _make_vision_model(response):
  """Returns a GeminiModelVision whose API replies with `response`."""
  with mock.patch.object(genai, 'Client'):
    model = gemini_model_multimodal.GeminiModelVision(
        model_name='test-model', api_key='key'
    )
  model._client.models.generate_content.return_value = response  # pylint: disable=protected-access
  return model


# Responses the API can return that contain no usable text. Each must become an
# empty string rather than an exception, because callers retry or move on.
def _empty_responses():
  return {
      'parts_none': lambda: _response_with_parts(None),
      'parts_empty': lambda: _response_with_parts([]),
      'part_text_none': lambda: _response_with_parts([types.Part(text=None)]),
      'candidates_none': lambda: types.GenerateContentResponse(candidates=None),
      'candidates_empty': lambda: types.GenerateContentResponse(candidates=[]),
      'content_none': lambda: types.GenerateContentResponse(
          candidates=[types.Candidate(content=None)]
      ),
  }


EMPTY_RESPONSE_NAMES = sorted([
    'parts_none',
    'parts_empty',
    'part_text_none',
    'candidates_none',
    'candidates_empty',
    'content_none',
])


class TestGeminiModel:

  def test_plain_text_is_returned_unchanged(self):
    model = _make_text_model(_text_response('Jake is not a turtle.'))
    assert model.sample_text('prompt') == 'Jake is not a turtle.'

  def test_surrounding_whitespace_is_trimmed(self):
    # `_strip_markdown` has always trimmed surrounding whitespace.
    model = _make_text_model(_text_response('  not a turtle.\n'))
    assert model.sample_text('prompt') == 'not a turtle.'

  def test_markdown_code_fence_is_stripped(self):
    model = _make_text_model(_text_response('```json\n{"a": 1}\n```'))
    assert model.sample_text('prompt') == '{"a": 1}'

  @pytest.mark.parametrize('name', EMPTY_RESPONSE_NAMES)
  def test_response_without_text_gives_empty_string(self, name):
    model = _make_text_model(_empty_responses()[name]())
    assert model.sample_text('prompt') == ''

  def test_sample_choice_survives_an_empty_response_then_succeeds(self):
    model = _make_text_model(None)
    chat = model._client.chats.create.return_value  # pylint: disable=protected-access
    chat.send_message.side_effect = [
        _response_with_parts(None),
        _text_response('(b)'),
    ]
    idx, response, _ = model.sample_choice('prompt', ['a', 'b'])
    assert (idx, response) == (1, 'b')


class TestGeminiModelVision:

  def test_plain_text_is_returned_unchanged(self):
    model = _make_vision_model(_text_response('Jake is not a turtle.'))
    assert model.sample_text('prompt') == 'Jake is not a turtle.'

  @pytest.mark.parametrize('name', EMPTY_RESPONSE_NAMES)
  def test_response_without_text_gives_empty_string(self, name):
    model = _make_vision_model(_empty_responses()[name]())
    assert model.sample_text('prompt') == ''

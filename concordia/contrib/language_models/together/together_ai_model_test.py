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

"""Together request/visible-text contracts; no provider clients or calls."""

from types import SimpleNamespace
from unittest import mock

from concordia.contrib import language_models
from concordia.contrib.language_models.together import together_ai_model
from concordia.language_model import language_model
import pytest

_FLASH = 'deepseek-ai/DeepSeek-V4.1-Flash'


@pytest.fixture(autouse=True)
def no_provider_or_backoff(caplog):
  caplog.set_level('DEBUG')
  with (
      mock.patch.object(
          together_ai_model.together,
          'Together',
          side_effect=AssertionError('No provider client'),
      ),
      mock.patch.object(
          together_ai_model.time,
          'sleep',
          side_effect=AssertionError('No retry expected'),
      ),
  ):
    yield


def client_with(content):
  client = mock.Mock()
  client.chat.completions.create.return_value = SimpleNamespace(
      choices=[
          SimpleNamespace(
              message=SimpleNamespace(
                  content=content, reasoning='private reasoning fixture'
              ),
              finish_reason='length',
          )
      ]
  )
  return client


@pytest.mark.parametrize('budget', [1, 16, 50])
def test_flash_short_text_keeps_budget_and_stop_with_reasoning_disabled(budget):
  client = client_with('Hello.')
  model = together_ai_model.Base(_FLASH, api_key='fixture', client=client)
  assert (
      model.sample_text(
          'Reply with Hello.', max_tokens=budget, terminators=['\n'], seed=7
      )
      == 'Hello.'
  )
  client.chat.completions.create.assert_called_once()
  values = client.chat.completions.create.call_args.kwargs
  assert values['model'] == _FLASH
  assert values['max_tokens'] == budget
  assert values['stop'] == ['\n']
  assert values['reasoning'] == {'enabled': False}
  assert values['seed'] == 7 and values['stream'] is False
  assert 'reasoning_effort' not in values


def test_flash_choice_uses_same_reasoning_policy_once():
  client = client_with('Bob')
  model = together_ai_model.Base(_FLASH, api_key='fixture', client=client)
  assert model.sample_choice('Next actor?', ['Alice', 'Bob'], seed=5) == (
      1,
      'Bob',
      {},
  )
  client.chat.completions.create.assert_called_once()
  values = client.chat.completions.create.call_args.kwargs
  assert values['reasoning'] == {'enabled': False}
  assert values['max_tokens'] == language_model.DEFAULT_MAX_TOKENS
  assert values['seed'] == 5


@pytest.mark.parametrize(
    'name',
    [
        'deepseek-ai/DeepSeek-V3',
        'deepseek-ai/DeepSeek-V4-Pro',
        _FLASH + '-unverified',
        'google/gemma-4-31B-it',
        'openai/gpt-oss-120b',
    ],
)
def test_other_model_request_options_unchanged(name):
  client = client_with('Hello.')
  model = together_ai_model.Base(name, api_key='fixture', client=client)
  assert (
      model.sample_text('Hello', max_tokens=16, terminators=['\n']) == 'Hello.'
  )
  values = client.chat.completions.create.call_args.kwargs
  assert 'reasoning' not in values
  assert values['max_tokens'] == (2048 if name.startswith('google/') else 16)
  assert values['stop'] == (None if name.startswith('google/') else ['\n'])
  if name.startswith(('google/', 'openai/')):
    assert values['reasoning_effort'] == 'low'
  else:
    assert 'reasoning_effort' not in values


@pytest.mark.parametrize(
    'name',
    [
        _FLASH,
        'deepseek-ai/DeepSeek-V3',
        'google/gemma-4-31B-it',
        'openai/gpt-oss-120b',
    ],
)
@pytest.mark.parametrize('choice', [False, True])
def test_none_content_fails_once_before_choice_retry_or_measurement(
    name, choice
):
  client = client_with(None)
  measurements = mock.Mock()
  model = together_ai_model.Base(
      name, api_key='fixture', client=client, measurements=measurements
  )
  with pytest.raises(
      language_model.InvalidResponseError, match='no visible text'
  ) as error:
    if choice:
      model.sample_choice('PRIVATE PROMPT', ['Alice', 'Bob'])
    else:
      model.sample_text('PRIVATE PROMPT', max_tokens=16)
  assert 'Request not retried' in str(error.value)
  assert 'private reasoning fixture' not in str(error.value)
  assert 'PRIVATE PROMPT' not in str(error.value)
  client.chat.completions.create.assert_called_once()
  measurements.publish_datum.assert_not_called()


def test_non_string_and_empty_string_rejected():
  client = client_with(123)
  model = together_ai_model.Base(_FLASH, api_key='fixture', client=client)
  with pytest.raises(language_model.InvalidResponseError):
    model.sample_text('Hello')
  client.chat.completions.create.return_value.choices[0].message.content = ''
  with pytest.raises(language_model.InvalidResponseError):
    model.sample_text('Hello')


def test_standard_factory_routes_flash_through_adapter():
  client = client_with('Hello.')
  with mock.patch.object(
      together_ai_model, '_create_together_client', return_value=client
  ) as create:
    model = language_models.language_model_setup(
        api_type='together_ai', model_name=_FLASH, api_key='fixture'
    )
  create.assert_called_once_with('fixture')
  assert isinstance(model, together_ai_model.Base)
  assert model.sample_text('Hello', max_tokens=16) == 'Hello.'
  assert client.chat.completions.create.call_args.kwargs['reasoning'] == {
      'enabled': False
  }


def sdk_error(status):
  import httpx

  request = httpx.Request(
      'POST',
      'https://fixture.invalid/DUMMY_SECRET',
      headers={'Authorization': 'Bearer DUMMY_SECRET'},
  )
  response = httpx.Response(status, request=request)
  return together_ai_model.together.APIStatusError(
      'DUMMY_SECRET raw exception',
      response=response,
      body={'key': 'DUMMY_SECRET'},
  )


@pytest.mark.parametrize(
    'name', [_FLASH, 'google/gemma-4-31B-it', 'openai/gpt-oss-120b']
)
@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
@pytest.mark.parametrize('choice', [False, True])
def test_permanent_failures_once_sanitized(
    name, status, choice, caplog, capsys
):
  client = client_with('unused')
  client.chat.completions.create.side_effect = sdk_error(status)
  model = together_ai_model.Base(name, api_key='fixture', client=client)
  with pytest.raises(language_model.InvalidResponseError) as error:
    if choice:
      model.sample_choice('DUMMY_SECRET prompt', ['Alice', 'Bob'])
    else:
      model.sample_text('DUMMY_SECRET prompt')
  assert f'HTTP {status}' in str(error.value)
  assert 'No automatic retry' in str(error.value)
  assert 'TOGETHER_API_KEY' not in str(error.value)
  assert 'TOGETHER_AI_API_KEY' not in str(error.value)
  assert 'DUMMY_SECRET' not in str(error.value) + caplog.text + ''.join(
      capsys.readouterr()
  )
  assert error.value.__suppress_context__
  client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    'name', [_FLASH, 'google/gemma-4-31B-it', 'openai/gpt-oss-120b']
)
@pytest.mark.parametrize(
    'status', [408, 429, 500, 503, 'connection', 'timeout']
)
@pytest.mark.parametrize('exhausted', [False, True])
def test_transient_retries_bounded_and_never_empty(
    name, status, exhausted, caplog
):
  import httpx

  client = client_with('Bob')
  success = client.chat.completions.create.return_value
  if status in ('connection', 'timeout'):
    request = httpx.Request('POST', 'https://fixture.invalid/DUMMY_SECRET')
    error = (
        together_ai_model.together.APIConnectionError(
            request=request, message='DUMMY_SECRET'
        )
        if status == 'connection'
        else together_ai_model.together.APITimeoutError(request=request)
    )
  else:
    error = sdk_error(status)
  client.chat.completions.create.side_effect = (
      [error] * 3 if exhausted else [error, success]
  )
  model = together_ai_model.Base(name, api_key='fixture', client=client)
  with mock.patch.object(together_ai_model.time, 'sleep') as sleep:
    if exhausted:
      with pytest.raises(
          language_model.InvalidResponseError, match='after 3 attempts'
      ) as caught:
        model.sample_choice('DUMMY_SECRET', ['Alice', 'Bob'])
      assert 'DUMMY_SECRET' not in str(caught.value)
    else:
      assert model.sample_choice('DUMMY_SECRET', ['Alice', 'Bob']) == (
          1,
          'Bob',
          {},
      )
  assert client.chat.completions.create.call_count == (3 if exhausted else 2)
  assert sleep.call_count == (2 if exhausted else 1)
  assert 'DUMMY_SECRET' not in caplog.text


@pytest.mark.parametrize(
    'name', [_FLASH, 'google/gemma-4-31B-it', 'openai/gpt-oss-120b']
)
@pytest.mark.parametrize('content', ['', ' \n ', None])
def test_no_visible_content_never_becomes_action_or_choice(name, content):
  client = client_with(content)
  model = together_ai_model.Base(name, api_key='fixture', client=client)
  with pytest.raises(
      language_model.InvalidResponseError, match='no visible text'
  ):
    model.sample_choice('Next', ['Alice', 'Bob'])
  client.chat.completions.create.assert_called_once()


def test_sdk_retries_disabled_and_choice_exhaustion_redacted():
  with mock.patch.object(together_ai_model.together, 'Together') as factory:
    together_ai_model._create_together_client('fixture')
  factory.assert_called_once_with(api_key='fixture', max_retries=0)
  client = client_with('DUMMY_SECRET unrecognized answer')
  model = together_ai_model.Base(_FLASH, api_key='fixture', client=client)
  with pytest.raises(
      language_model.InvalidResponseError,
      match='no valid choice after 3 attempts',
  ) as error:
    model.sample_choice('Next', ['Alice', 'Bob'])
  assert 'DUMMY_SECRET' not in str(error.value)
  assert client.chat.completions.create.call_count == 3


@pytest.mark.parametrize('response', [None, SimpleNamespace(choices=[])])
def test_malformed_response_fails_once(response):
  client = client_with('unused')
  client.chat.completions.create.return_value = response
  with pytest.raises(
      language_model.InvalidResponseError, match='invalid response structure'
  ):
    together_ai_model.Base(
        _FLASH, api_key='fixture', client=client
    ).sample_text('Prompt')
  client.chat.completions.create.assert_called_once()


def test_gemma_stop_cannot_reduce_reply_to_fabricated_empty_action():
  client = client_with('\nSome content')
  with pytest.raises(
      language_model.InvalidResponseError, match='no visible text'
  ):
    together_ai_model.Base(
        'google/gemma-4-31B-it', api_key='fixture', client=client
    ).sample_text('Prompt', terminators=['\n'])
  client.chat.completions.create.assert_called_once()

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

"""Language Model wrapper for Mistral models."""

from collections.abc import Collection, Sequence
import os
import time

from concordia.language_model import language_model
from concordia.utils import sampling
from concordia.utils.deprecated import measurements as measurements_lib
import mistralai
from mistralai import models
from typing_extensions import override


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_MAX_CHAT_ATTEMPTS = 20

# At least one Mistral model supports completion mode.
COMPLETION_MODELS = (
    'codestral-latest',
    'codestral-2405',
)

_NUM_SILENT_ATTEMPTS = 3


class MistralLanguageModel(language_model.LanguageModel):
  """Language Model wrapper that uses Mistral models."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      use_codestral_for_choices: bool = False,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The language model to use. For more details, see
        https://docs.mistral.ai/getting-started/models/.
      api_key: The API key to use when accessing the OpenAI API, if None will
        use the MISTRAL_API_KEY environment variable.
      use_codestral_for_choices: When enabled, use codestral for multiple choice
        questions. Otherwise, use the model specified in the param `model_name`.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    if api_key is None:
      api_key = os.environ['MISTRAL_API_KEY']
    self._api_key = api_key
    self._text_model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._client = mistralai.Mistral(api_key=self._api_key)

    self._choice_model_name = self._text_model_name
    if use_codestral_for_choices:
      self._choice_model_name = 'codestral-latest'

    self._completion_for_text = False
    if self._text_model_name in COMPLETION_MODELS:
      self._completion_for_text = True

    self._completion_for_choice = False
    if self._choice_model_name in COMPLETION_MODELS:
      self._completion_for_choice = True

  def _complete_text(
      self,
      prompt: str,
      *,
      suffix: str | None = None,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      seed: int | None = None,
  ) -> str:
    if not terminators:
      # It is essential to set a terminator since these models otherwise always
      # continue till max_tokens.
      terminators = ('\n\n',)

    result = ''
    for attempts in range(_MAX_CHAT_ATTEMPTS):
      if attempts > 0:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print('Sleeping for 10 seconds... ' +
                f'attempt: {attempts} / {_MAX_CHAT_ATTEMPTS}')
        time.sleep(10)
      try:
        response = self._client.fim.complete(
            model=self._choice_model_name,
            prompt=prompt,
            suffix=suffix,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=terminators,
            random_seed=seed,
        )
      except mistralai.models.sdkerror.SDKError as err:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(f'  Exception: {err}')
        continue
      else:
        result = response.choices[0].message.content
        break

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)},
      )

    return result

  def _chat_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      seed: int | None = None,
  ) -> str:
    del terminators
    messages = [
        models.SystemMessage(
            role='system',
            content=(
                'You always continue sentences provided by the user and you '
                'never repeat what the user already said.'
            )
        ),
        models.UserMessage(
            role='user',
            content='Question: Is Jake a turtle?\nAnswer: Jake is ',
        ),
        models.AssistantMessage(role='assistant', content='not a turtle.'),
        models.UserMessage(
            role='user',
            content=(
                'Question: What is Priya doing right now?\n'
                'Answer: Priya is currently '
            ),
        ),
        models.AssistantMessage(role='assistant', content='sleeping.'),
        models.UserMessage(role='user', content=prompt),
    ]
    for attempts in range(_MAX_CHAT_ATTEMPTS):
      if attempts > 0:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print('Sleeping for 10 seconds... ' +
                f'attempt: {attempts} / {_MAX_CHAT_ATTEMPTS}')
        time.sleep(10)
      try:
        response = self._client.chat.complete(
            model=self._text_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            random_seed=seed,
        )
      except mistralai.models.sdkerror.SDKError as err:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(f'  Exception: {err}')
        continue
      else:
        return response.choices[0].message.content

    raise language_model.InvalidResponseError(
        (f'Too many chat attempts.\n Prompt: {prompt}')
    )

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del timeout

    if self._completion_for_text:
      response = self._complete_text(
          prompt=prompt,
          suffix='.\n',
          max_tokens=max_tokens,
          terminators=terminators,
          temperature=temperature,
          seed=seed,
      )
    else:
      response = self._chat_text(
          prompt=prompt,
          max_tokens=max_tokens,
          terminators=terminators,
          temperature=temperature,
          seed=seed,
      )

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(response)},
      )
    return response

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    prompt = (
        prompt
        + '\nChoose one:\n'
        + '\n'.join(responses)
        + '\nchoice=('
    )

    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

      if self._completion_for_choice:
        sample = self._complete_text(
            prompt=prompt,
            suffix=')',
            max_tokens=256,
            terminators=(' ', '\n'),
            temperature=temperature,
            seed=seed,
        )
      else:
        sample = self._chat_text(
            prompt,
            max_tokens=3,
            temperature=temperature,
            seed=seed,
        )
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError(
        (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
         f'extracted: {answer}')
    )

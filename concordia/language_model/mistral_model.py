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
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from typing_extensions import override

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

# At least one Mistral model supports completion mode.
COMPLETION_MODELS = (
    'codestral-latest',
)


class MistralLanguageModel(language_model.LanguageModel):
  """Language Model wrapper that uses Mistral models."""

  def __init__(
      self,
      api_key: str,
      model_name: str,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      api_key: The API key to use when accessing the OpenAI API.
      model_name: The language model to use. For more details, see
        https://docs.mistral.ai/getting-started/models/.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._api_key = api_key
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._client = MistralClient(api_key=api_key)

    self._completion = False
    if self._model_name in COMPLETION_MODELS:
      self._completion = True

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

    response = self._client.completion(
        model=self._model_name,
        prompt=prompt,
        suffix=suffix,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=terminators,
        random_seed=seed,
    )

    result = response.choices[0].message.content

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)},
      )
    # Remove the occasional sentence fragment from the end of the result.
    last_stop = result.rfind('.')
    if last_stop >= 0:
      result = result[:last_stop + 1]
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
        ChatMessage(role='system',
                    content=('You always continue sentences provided ' +
                             'by the user and you never repeat what ' +
                             'the user already said.')),
        ChatMessage(role='user',
                    content='Question: Is Jake a turtle?\nAnswer: Jake is '),
        ChatMessage(role='assistant',
                    content='not a turtle.'),
        ChatMessage(role='user',
                    content=('Question: What is Priya doing right '
                             'now?\nAnswer: Priya is currently ')),
        ChatMessage(role='assistant',
                    content='sleeping.'),
        ChatMessage(role='user',
                    content=prompt)
    ]
    response = self._client.chat(
        model=self._model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        random_seed=seed,
    )
    return response.choices[0].message.content

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

    if self._completion:
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

      if self._completion:
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

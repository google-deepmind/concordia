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

"""Language Model that uses Together AI api.

Recommended model name is 'google/gemma-2-9b-it'
"""

from collections.abc import Collection, Sequence
import concurrent.futures
import os
import time
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import numpy as np
import together
from typing_extensions import override

_MAX_ATTEMPTS = 20
_NUM_SILENT_ATTEMPTS = 3
_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED = 2
_DEFAULT_MAX_TOKENS = 5000


class Gemma2(language_model.LanguageModel):
  """Language Model that uses Together AI models."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The language model to use. For more details, see
        https://api.together.xyz/models.
      api_key: The API key to use when accessing the Together AI API. If None,
        will use the TOGETHER_AI_API_KEY environment variable.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    if api_key is None:
      api_key = os.environ['TOGETHER_AI_API_KEY']
    self._api_key = api_key
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._client = together.Together(api_key=self._api_key)

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
    messages = [
        {
            'role': 'system',
            'content': (
                'You always continue sentences provided '
                'by the user and you never repeat what '
                'the user has already said. All responses must end with a '
                'period. Try not to use lists, but if you must, then '
                'always delimit list items using either '
                r"semicolons or single newline characters ('\n'), never "
                r"delimit list items with double carriage returns ('\n\n')."
            ),
        },
        {
            'role': 'user',
            'content': 'Question: Is Jake a turtle?\nAnswer: Jake is ',
        },
        {'role': 'assistant', 'content': 'not a turtle.'},
        {
            'role': 'user',
            'content': (
                'Question: What is Priya doing right now?\nAnswer: '
                + 'Priya is currently '
            ),
        },
        {'role': 'assistant', 'content': 'sleeping.'},
        {'role': 'user', 'content': prompt},
    ]

    # gemma2 does not support `tokens` + `max_new_tokens` > 8193.
    # gemma2 interprets our `max_tokens`` as their `max_new_tokens`.
    max_tokens = min(max_tokens, _DEFAULT_MAX_TOKENS)

    result = ''
    for attempts in range(_MAX_ATTEMPTS):
      if attempts > 0:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(
              f'Sleeping for {_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED} seconds... '
              + f'attempt: {attempts} / {_MAX_ATTEMPTS}'
          )
        time.sleep(_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED)
      try:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stop=terminators,
            seed=seed,
            stream=False,
        )
      except together.error.RateLimitError as err:
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
    # Remove the occasional sentence fragment from the end of the result.
    last_stop = result.rfind('.')
    if last_stop >= 0:
      result = result[: last_stop + 1]
    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:

    def _sample_choice(response: str) -> float:
      augmented_prompt = prompt + response
      messages = [
          {
              'role': 'system',
              'content': (
                  'You always continue sentences provided '
                  + 'by the user and you never repeat what '
                  + 'the user already said.'
              ),
          },
          {
              'role': 'user',
              'content': 'Question: Is Jake a turtle?\nAnswer: Jake is ',
          },
          {'role': 'assistant', 'content': 'not a turtle.'},
          {
              'role': 'user',
              'content': (
                  'Question: What is Priya doing right now?\nAnswer: '
                  + 'Priya is currently '
              ),
          },
          {'role': 'assistant', 'content': 'sleeping.'},
          {'role': 'user', 'content': augmented_prompt},
      ]

      result = None
      for attempts in range(_MAX_ATTEMPTS):
        if attempts > 0:
          if attempts >= _NUM_SILENT_ATTEMPTS:
            print(
                f'Sleeping for {_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED} seconds.. '
                + f'attempt: {attempts} / {_MAX_ATTEMPTS}'
            )
          time.sleep(_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED)
        try:
          result = self._client.chat.completions.create(
              model=self._model_name,
              messages=messages,
              seed=seed,
              logprobs=1,
              stream=False,
          )
        except together.error.RateLimitError as err:
          if attempts >= _NUM_SILENT_ATTEMPTS:
            print(f'  Exception: {err}')
            print(f'  Exception prompt: {augmented_prompt}')
          continue
        else:
          break

      if result:
        lp = sum(result.choices[0].logprobs.token_logprobs)
      else:
        raise ValueError(
            f'Failed to get logprobs.\nException prompt: {augmented_prompt}')

      return lp

    with concurrent.futures.ThreadPoolExecutor() as executor:
      logprobs_np = np.array(
          list(executor.map(_sample_choice, responses))
      ).reshape(-1)

    idx = np.argmax(logprobs_np)

    # Get the corresponding response string
    max_str = responses[idx]

    return idx, max_str, {r: logprobs_np[i] for i, r in enumerate(responses)}

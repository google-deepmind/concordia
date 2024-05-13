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


"""Language Model that uses OpenAI's GPT models."""

from collections.abc import Collection, Sequence
import re
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import openai
from typing_extensions import override

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


def extract_response(sample: str):
  """Given text formatted as 'lorum(a)ipsum', return 'a'."""
  match = re.search(r'\(?(\w)\)', sample)
  if match:
    return match.group(1)
  else:
    return None


class GptLanguageModel(language_model.LanguageModel):
  """Language Model that uses OpenAI GPT models."""

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
        https://platform.openai.com/docs/guides/text-generation/which-model-should-i-use.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._api_key = api_key
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._client = openai.OpenAI(
        api_key=api_key,
    )

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    # gpt models do not support `max_characters`.
    del max_characters
    # gpt models do not support `max_tokens` > 4096.
    max_tokens = min(max_tokens, 4000)

    messages = [
        {'role': 'system',
         'content': ('You always continue sentences provided ' +
                     'by the user and you never repeat what ' +
                     'the user already said.')},
        {'role': 'user',
         'content': 'Question: Is Jake a turtle?\nAnswer: Jake is '},
        {'role': 'assistant',
         'content': 'not a turtle.'},
        {'role': 'user',
         'content': ('Question: What is Priya doing right now?\nAnswer: ' +
                     'Priya is currently ')},
        {'role': 'assistant',
         'content': 'sleeping.'},
        {'role': 'user',
         'content': prompt}
    ]

    response = self._client.chat.completions.create(
        model=self._model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        stop=terminators,
        seed=seed,
    )

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(response.choices[0].message.content)},
      )
    return response.choices[0].message.content

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:

    attempts = 1
    prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings:\n'
        + '\n'.join(responses) + '.'
    )

    sample = ''
    answer = ''
    for _ in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = 0.0
      if attempts > 1:
        temperature = 0.5

      sample = self.sample_text(
          prompt,
          temperature=temperature,
          seed=seed,
      )
      if len(sample) == 1:
        # i.e. this would be a sample such as "a"
        answer = sample
      elif len(sample) == 2:
        # i.e. this would be a sample such as "a)"
        answer = sample[0]
      else:
        # extract a substring like "(a)" wherever it may be in a longer string
        answer = extract_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        attempts += 1
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

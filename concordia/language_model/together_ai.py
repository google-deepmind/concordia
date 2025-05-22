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

Recommended model names are:
  'google/gemma-2-9b-it'
  'google/gemma-2-27b-it'
"""

from collections.abc import Collection, Sequence
import concurrent.futures
import os
import random
import time

from absl import logging
from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib
import numpy as np
import together
from typing_extensions import override


_MAX_ATTEMPTS = 20
_NUM_SILENT_ATTEMPTS = 3
_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED = 2
_JITTER_SECONDS = 0.25
_DEFAULT_NUM_RESPONSE_TOKENS = 5000

_GUESS_CHARS_PER_TOKEN = 4
# Max tokens is really 8193, but we leave substantial margin since estimates
# of the number of tokens are imprecise and also calculated before adding the
# system messages.
_MAX_ALLOWED_TOKENS = 7000
# Use `_NUM_INITIAL_TOKENS` from the start of the prompt if possible when
# trimming to fit the whole sequence into `_MAX_ALLOWED_TOKENS`.
_NUM_INITIAL_TOKENS = 500


def _find_response_start_index(tokens):
  r"""Finds the start of the response in the prompt.

  Args:
    tokens: A list of strings.

  Returns:
    The index of the last occurrence of '<start_of_turn>' followed by 'model'
    and '\n', or 1 if the sequence is not found. This corresponds to the start 
    of the response.
  """
  assert len(tokens) >= 3, "Response doesn't match expectation."
  for i in range(len(tokens) - 3, -1, -1):
    if (
        tokens[i] == '<start_of_turn>'
        and tokens[i + 1] == 'model'
        and tokens[i + 2] == '\n'
    ):
      return i + 3  # Return the index after the sequence
  raise ValueError("Response doesn't match expectation.")


def _ensure_prompt_not_too_long(
    prompt: str,
    num_response_tokens: int,
    guess_chars_per_token: int = _GUESS_CHARS_PER_TOKEN) -> str:
  r"""Ensures the prompt is not too long for Together AI\'s Gemma-2 models."""
  num_initial_chars = _NUM_INITIAL_TOKENS * guess_chars_per_token
  max_prompt_tokens = _MAX_ALLOWED_TOKENS - num_response_tokens
  if max_prompt_tokens <= 0:
    raise ValueError(
        f'Cannot reserve {num_response_tokens} of {_MAX_ALLOWED_TOKENS} tokens.'
    )
  max_prompt_chars = max_prompt_tokens * guess_chars_per_token
  if len(prompt) <= max_prompt_chars:
    return prompt

  # Keep the first _NUM_INITIAL_TOKENS tokens and then skip to the last tokens
  # and take as many as we can from the end.
  if max_prompt_chars > num_initial_chars:
    num_final_chars = max_prompt_chars - num_initial_chars
    new_prompt = prompt[:num_initial_chars] + prompt[-num_final_chars:]
    logging.info('Prompt too long, trimmed it down, while keeping start and '
                 'end, resulting in %d characters', len(new_prompt))
    logging.debug('Trimmed prompt: %s', new_prompt)
    return new_prompt

  # This happens if len(prompt) > max_prompt_chars <= num_initial_chars.
  new_prompt = prompt[-max_prompt_chars:]
  logging.info(
      'Prompt too long, truncated it to last %d characters.',
      max_prompt_chars
  )
  logging.debug('Truncated prompt: %s', new_prompt)
  return new_prompt


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
    original_prompt = prompt
    prompt = _ensure_prompt_not_too_long(prompt, max_tokens)
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
    max_tokens = min(max_tokens, _DEFAULT_NUM_RESPONSE_TOKENS)

    result = ''
    for attempts in range(_MAX_ATTEMPTS):
      if attempts > 0:
        seconds_to_sleep = (_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED +
                            random.uniform(-_JITTER_SECONDS, _JITTER_SECONDS))
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(
              f'Sleeping for {seconds_to_sleep} seconds... '
              + f'attempt: {attempts} / {_MAX_ATTEMPTS}'
          )
        time.sleep(seconds_to_sleep)
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
      except (together.error.RateLimitError,
              together.error.APIError,
              together.error.ServiceUnavailableError,
              together.error.InvalidRequestError) as err:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(f'  Exception: {err}')
          print(f'  Text exception prompt: {prompt}')
        if isinstance(err, together.error.APIError) or isinstance(
            err, together.error.InvalidRequestError
        ):
          # If hit the error that arises from a prompt that is too long then
          # re-run the trimming function with a more pessimistic guess of the
          # the number of characters per token.
          prompt = _ensure_prompt_not_too_long(original_prompt,
                                               max_tokens,
                                               guess_chars_per_token=1)
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

  def _sample_choice(
      self, prompt: str, response: str) -> float:
    """Returns the log probability of the prompt and response."""
    original_prompt = prompt
    augmented_prompt = _ensure_prompt_not_too_long(prompt, len(response))
    attempts = 0
    for attempts in range(_MAX_ATTEMPTS):
      if attempts > 0:
        seconds_to_sleep = (_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED +
                            random.uniform(-_JITTER_SECONDS, _JITTER_SECONDS))
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(
              f'Sleeping for {seconds_to_sleep} seconds.. '
              + f'attempt: {attempts} / {_MAX_ATTEMPTS}'
          )
        time.sleep(seconds_to_sleep)
      try:
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
            {'role': 'assistant', 'content': response},
        ]
        result = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=1,
            seed=None,
            logprobs=1,
            stream=False,
            echo=True,
        )
      except (together.error.RateLimitError,
              together.error.APIError,
              together.error.ServiceUnavailableError,
              together.error.InvalidRequestError) as err:
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(f'  Exception: {err}')
          print(f'  Choice exception prompt: {augmented_prompt}')
        if isinstance(err, together.error.APIError) or isinstance(
            err, together.error.InvalidRequestError
        ):
          # If hit the error that arises from a prompt that is too long then
          # re-run the trimming function with a more pessimistic guess of the
          # the number of characters per token.
          augmented_prompt = _ensure_prompt_not_too_long(
              original_prompt, 1, guess_chars_per_token=1
          )
        continue
      else:
        logprobs = result.prompt[0].logprobs
        response_idx = _find_response_start_index(logprobs.tokens)
        response_log_probs = logprobs.token_logprobs[response_idx:]
        score = sum(response_log_probs)
        return score

    raise language_model.InvalidResponseError(
        f'Failed to get logprobs after {attempts+1} attempts.\n Exception'
        f' prompt: {augmented_prompt}'
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:

    sample_choice_for_prompt = lambda x: self._sample_choice(prompt, x)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      logprobs_np = np.array(
          list(executor.map(sample_choice_for_prompt, responses))
      ).reshape(-1)

    idx = np.argmax(logprobs_np)

    # Get the corresponding response string
    max_str = responses[idx]

    return idx, max_str, {r: logprobs_np[i] for i, r in enumerate(responses)}

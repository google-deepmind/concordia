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

"""Gemini Language Model via the unified google-genai SDK.

Supports both Google AI Studio (API key) and Vertex AI (GCP project) backends
through a single class using the google-genai SDK.

See https://ai.google.dev/gemini-api/docs/migrate for migration details.
"""

from collections.abc import Collection, Sequence
import copy
import os
import time
from typing import override

from absl import logging
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from concordia.utils import text
from google import genai
from google.genai import types


MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

DEFAULT_HISTORY = [
    types.Content(
        role='user',
        parts=[
            types.Part(
                text=(
                    'You always continue sentences provided by the user, and'
                    ' you never repeat what the user already said.'
                ),
            ),
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(
                text=(
                    'I always continue user-provided text and never repeat'
                    ' what the user already said.'
                ),
            ),
        ],
    ),
    types.Content(
        role='user',
        parts=[
            types.Part(
                text='Question: Is Jake a turtle?\nAnswer: Jake is ',
            ),
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(text='not a turtle.'),
        ],
    ),
    types.Content(
        role='user',
        parts=[
            types.Part(
                text=(
                    'Question: What is Priya doing right now?\n'
                    'Answer: Priya is currently '
                ),
            ),
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(text='sleeping.'),
        ],
    ),
]


DEFAULT_SAFETY_SETTINGS = [
    types.SafetySetting(
        category='HARM_CATEGORY_HARASSMENT',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_HATE_SPEECH',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_DANGEROUS_CONTENT',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
]


class GeminiModel(language_model.LanguageModel):
  """Language model using the unified google-genai SDK.

  This class supports both Google AI Studio and Vertex AI backends.
  To use AI Studio, provide an api_key (or set the GEMINI_API_KEY env var).
  To use Vertex AI, provide project and location parameters.
  """

  def __init__(
      self,
      model_name: str = 'gemini-2.5-pro',
      *,
      api_key: str | None = None,
      project: str | None = None,
      location: str | None = None,
      safety_settings: Sequence[types.SafetySetting] = DEFAULT_SAFETY_SETTINGS,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      sleep_periodically: bool = False,
  ) -> None:
    """Initializes the Gemini language model.

    Provide either api_key for AI Studio or project/location for Vertex AI.
    If neither is provided, the api_key will be read from the GEMINI_API_KEY
    environment variable for AI Studio access.

    Args:
      model_name: which language model to use (e.g. 'gemini-2.5-pro').
        For available models see https://ai.google.dev/gemini-api/docs/models
      api_key: API key for Gemini API access. If None and project is also
        None, will try the GEMINI_API_KEY environment variable.
      project: GCP project ID for Vertex AI access.
      location: GCP region for Vertex AI (e.g. 'us-central1'). Required if
        project is provided.
      safety_settings: Safety settings for content filtering.
        See https://ai.google.dev/gemini-api/docs/safety-settings
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
      sleep_periodically: Whether to sleep between API calls to avoid rate
        limits.
    """
    if project and api_key:
      raise ValueError(
          'Provide either api_key (for AI Studio) or project/location '
          '(for Vertex AI), not both.'
      )

    if project:
      if not location:
        raise ValueError(
            'location is required when using Vertex AI (project is set).'
        )
      self._client = genai.Client(
          vertexai=True, project=project, location=location
      )
    else:
      if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
          raise ValueError(
              'GEMINI_API_KEY not found. Please provide it via the api_key '
              'parameter or set the GEMINI_API_KEY environment variable.'
          )
      self._client = genai.Client(api_key=api_key)

    self._model_name = model_name
    self._safety_settings = list(safety_settings)
    self._sleep_periodically = sleep_periodically
    self._measurements = measurements
    self._channel = channel

    self._calls_between_sleeping = 10
    self._n_calls = 0

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del timeout

    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0
    ):
      logging.info('Sleeping for 10 seconds...')
      time.sleep(10)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        stop_sequences=list(terminators),
        candidate_count=1,
        top_p=top_p,
        top_k=top_k,
        response_mime_type='text/plain',
        safety_settings=self._safety_settings,
        seed=seed,
    )

    chat = self._client.chats.create(
        model=self._model_name,
        history=copy.deepcopy(DEFAULT_HISTORY),
        config=config,
    )
    sample = chat.send_message(message=prompt)

    try:
      response = sample.candidates[0].content.parts[0].text
    except (ValueError, IndexError, AttributeError) as e:
      logging.error('An error occurred: %s', e)
      logging.debug('prompt: %s', prompt)
      logging.debug('sample: %s', sample)
      response = ''
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {'raw_text_length': len(response)}
      )
    return text.truncate(response, delimiters=terminators)

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    sample = ''
    answer = ''
    for attempts in range(MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, MAX_MULTIPLE_CHOICE_ATTEMPTS
      )

      question = (
          'The following is a multiple choice question. Respond '
          + 'with one of the possible choices, such as (a) or (b). '
          + f'Do not include reasoning.\n{prompt}'
      )
      sample = self.sample_text(
          question,
          max_tokens=256,  # This is wasteful, but Gemini blocks lower values.
          temperature=temperature,
          seed=seed,
      )
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        logging.debug(
            'Sample choice fail: %s extracted from %s.', answer, sample
        )
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError((
        f'Too many multiple choice attempts.\nLast attempt: {sample}, '
        + f'extracted: {answer}'
    ))

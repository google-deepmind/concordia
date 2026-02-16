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
"""Google Cloud Vertex Language Model."""

from collections.abc import Collection, Sequence
import copy
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
            types.Part(text='Continue my sentences. Never repeat their starts.')
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(
                text=(
                    'I always continue user-provided text and never repeat '
                    + 'what the user already said.'
                )
            )
        ],
    ),
    types.Content(
        role='user',
        parts=[
            types.Part(text='Question: Is Jake a turtle?\nAnswer: Jake is ')
        ],
    ),
    types.Content(role='model', parts=[types.Part(text='not a turtle.')]),
    types.Content(
        role='user',
        parts=[
            types.Part(
                text=(
                    'Question: What is Priya doing right now?\nAnswer: '
                    + 'Priya is currently '
                )
            )
        ],
    ),
    types.Content(role='model', parts=[types.Part(text='sleeping.')]),
]


class VertexLanguageModel(language_model.LanguageModel):
  """Language model via the vertex API for Google Cloud."""

  def __init__(
      self,
      model_name: str = 'gemini-2.5-pro',
      *,
      harm_block_threshold: str = 'BLOCK_NONE',
      project: str | None = None,
      location: str = 'us-central1',
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      sleep_periodically: bool = False,
  ) -> None:
    """Initializes a model instance using the Google Cloud language model API.

    Args:
      model_name: which language model to use
      For a list of available Vertex AI models, see:
        https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
      harm_block_threshold: Safety threshold. Choose from {'BLOCK_ONLY_HIGH',
        'BLOCK_MEDIUM_AND_ABOVE', 'BLOCK_LOW_AND_ABOVE', 'BLOCK_NONE'}
      project: Google Cloud project ID. If None, uses GOOGLE_CLOUD_PROJECT
        environment variable or default credentials.
      location: Google Cloud location for Vertex AI (default: 'us-central1')
      measurements: The measurements object to log usage statistics to
      channel: The channel to write the statistics to
      sleep_periodically: Whether to sleep between API calls to avoid rate limit
    """
    self._model_name = model_name
    self._client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
    self._measurements = measurements
    self._channel = channel
    self._safety_settings = [
        types.SafetySetting(
            category='HARM_CATEGORY_HARASSMENT',
            threshold=harm_block_threshold,
        ),
        types.SafetySetting(
            category='HARM_CATEGORY_HATE_SPEECH',
            threshold=harm_block_threshold,
        ),
        types.SafetySetting(
            category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
            threshold=harm_block_threshold,
        ),
        types.SafetySetting(
            category='HARM_CATEGORY_DANGEROUS_CONTENT',
            threshold=harm_block_threshold,
        ),
    ]
    self._sleep_periodically = sleep_periodically
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
    if seed is not None:
      raise NotImplementedError('Unclear how to set seed for cloud models.')
    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0
    ):
      logging.info('Sleeping for 10 seconds...')
      time.sleep(10)

    chat = self._client.chats.create(
        model=self._model_name,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
            stop_sequences=terminators,
            candidate_count=1,
            safety_settings=self._safety_settings,
        ),
        history=copy.deepcopy(DEFAULT_HISTORY),
    )
    sample = chat.send_message(prompt)
    try:
      response = sample.text
    except (ValueError, AttributeError) as e:
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

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

"""Gemini Language Model with vision/image generation support.

This model uses the google-genai SDK and is designed for multimodal use:
  - Parses inline markdown images from prompts and sends them as multimodal
    input parts, allowing the model to "see" images.
  - Extracts image data from model responses and returns it as inline
    markdown image tags.

Unlike the text-only GeminiModel, it does NOT use text-completion history
or system instructions, which allows models like gemini-2.5-flash-image to
generate images.

Usage:
    model = GeminiModelVision(
        model_name='gemini-2.5-flash-image',
        api_key='YOUR_KEY',
    )
    result = model.sample_text('Generate an image of a cat.')
    # result contains ![image](data:image/png;base64,...) if image generated
"""

import base64
from collections.abc import Collection, Sequence
import os
import re
import time
from typing import override

from absl import logging
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from concordia.utils import text as text_utils
from google import genai
from google.genai import types


_IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(data:(image/[a-zA-Z0-9.\-]+);base64,([^)]+)\)'
)

DEFAULT_SAFETY_SETTINGS = (
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
)


def _parse_prompt_to_parts(prompt: str) -> list[types.Part]:
  """Parses a prompt string, extracting inline markdown images as Parts."""
  parts = []
  last_end = 0

  for match in _IMAGE_PATTERN.finditer(prompt):
    start, end = match.span()

    text_before = prompt[last_end:start]
    if text_before:
      parts.append(types.Part(text=text_before))

    mime_type = match.group(2)
    image_bytes = base64.b64decode(match.group(3))
    parts.append(
        types.Part(
            inline_data=types.Blob(mime_type=mime_type, data=image_bytes)
        )
    )

    last_end = end

  remaining = prompt[last_end:]
  if remaining:
    parts.append(types.Part(text=remaining))

  return parts


class GeminiModelVision(language_model.LanguageModel):
  """Gemini language model with vision/image generation support.

  This class is designed for models that can generate images (e.g.,
  gemini-2.5-flash-image). It passes prompts without text-completion
  history or system instructions to allow image generation.

  Image data is returned inline as markdown image tags:
      ![image](data:image/png;base64,...)
  """

  def __init__(
      self,
      model_name: str = 'gemini-2.5-flash-image',
      *,
      api_key: str | None = None,
      project: str | None = None,
      location: str | None = None,
      safety_settings: Sequence[types.SafetySetting] = DEFAULT_SAFETY_SETTINGS,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      sleep_periodically: bool = False,
  ) -> None:
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
      temperature: float = 1.0,
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
        candidate_count=1,
        top_p=top_p,
        top_k=top_k,
        safety_settings=self._safety_settings,
        seed=seed,
        response_modalities=['TEXT', 'IMAGE'],
    )

    contents = _parse_prompt_to_parts(prompt)

    response = self._client.models.generate_content(
        model=self._model_name,
        contents=contents,
        config=config,
    )

    output_parts = []
    try:
      for part in response.candidates[0].content.parts:
        if part.text:
          output_parts.append(part.text)
        elif part.inline_data:
          b64_data = base64.b64encode(part.inline_data.data).decode('utf-8')
          mime = part.inline_data.mime_type
          output_parts.append(f'![image](data:{mime};base64,{b64_data})')
    except (ValueError, IndexError, AttributeError) as e:
      logging.error('An error occurred: %s', e)
      logging.debug('prompt: %s', prompt)
      logging.debug('response: %s', response)

    sample = ''.join(output_parts)
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {'raw_text_length': len(sample)}
      )
    return text_utils.truncate(sample, delimiters=terminators)

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    max_attempts = 20
    sample = ''
    answer = ''
    for attempts in range(max_attempts):
      temperature = sampling.dynamically_adjust_temperature(
          attempts, max_attempts
      )
      question = (
          'The following is a multiple choice question. Respond '
          + 'with one of the possible choices, such as (a) or (b). '
          + f'Do not include reasoning.\n{prompt}'
      )
      sample = self.sample_text(
          question,
          max_tokens=256,
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

    raise language_model.InvalidResponseError((
        f'Too many multiple choice attempts.\nLast attempt: {sample}, '
        + f'extracted: {answer}'
    ))

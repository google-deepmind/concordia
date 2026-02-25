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

"""OpenAI language model with vision input and image generation support.

Designed for multimodal use as both the main agent model and image model:
  - Parses inline markdown images from prompts and sends them as multimodal
    content parts to chat completions, allowing the model to "see" images.
  - Detects image-generation requests (prefixed with _IMAGE_GENERATION_TRIGGER)
    and calls the OpenAI images API, returning the result as an inline markdown
    image tag so it is compatible with ImageTextActComponent.

Usage:
    model = GptVisionModel(
        model_name='gpt-5-mini',
        image_model_name='gpt-image-1',
        api_key='YOUR_KEY',
    )
    # As main model:
    text = model.sample_text('What is alchemy?')
    # As image model (called by ImageTextActComponent):
    result = model.sample_text(
        'Generate an image based on the following description. ...'
    )
    # result contains ![image](data:image/png;base64,...) if successful
"""

from collections.abc import Collection, Sequence
import os
import re
from typing import Any, override

from absl import logging
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import text as text_utils
import openai


_IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(data:(image/[a-zA-Z0-9.\-]+);base64,([^)]+)\)'
)

# Prefix injected by ImageTextActComponent when requesting image generation.
_IMAGE_GENERATION_TRIGGER = (
    'Generate an image based on the following description.'
)

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_DEFAULT_VERBOSITY = 'low'


def _parse_prompt_to_content(prompt: str) -> list[dict[str, Any]]:
  """Parse a prompt, converting inline markdown images to content parts."""
  parts = []
  last_end = 0

  for match in _IMAGE_PATTERN.finditer(prompt):
    start, end = match.span()

    text_before = prompt[last_end:start]
    if text_before:
      parts.append({'type': 'text', 'text': text_before})

    mime_type = match.group(2)
    b64_data = match.group(3)
    parts.append({
        'type': 'image_url',
        'image_url': {'url': f'data:{mime_type};base64,{b64_data}'},
    })

    last_end = end

  remaining = prompt[last_end:]
  if remaining:
    parts.append({'type': 'text', 'text': remaining})

  return parts


class GptVisionModel(language_model.LanguageModel):
  """OpenAI model with vision input and image generation output.

  When used as the main language model, parses inline markdown images from
  prompts and sends them as multimodal vision content to chat completions.

  When used as the image model (by ImageTextActComponent), detects the image
  generation trigger prefix and calls the OpenAI images API, returning the
  result as an inline markdown image tag.
  """

  def __init__(
      self,
      model_name: str = 'gpt-5-mini',
      *,
      image_model_name: str = 'gpt-image-1',
      api_key: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ) -> None:
    """Initializes the instance.

    Args:
      model_name: The chat completions model for text (and vision) generation.
      image_model_name: The model used for image generation via the images API.
      api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
      measurements: Optional measurements object for logging statistics.
      channel: The channel to write statistics to.
    """
    if api_key is None:
      api_key = os.getenv('OPENAI_API_KEY')
      if not api_key:
        raise ValueError(
            'OPENAI_API_KEY not found. Please provide it via the api_key '
            'parameter or set the OPENAI_API_KEY environment variable.'
        )
    self._client = openai.OpenAI(api_key=api_key)
    self._model_name = model_name
    self._image_model_name = image_model_name
    self._measurements = measurements
    self._channel = channel

    self._verbosity = _DEFAULT_VERBOSITY
    if 'gpt-4o' in self._model_name:
      self._verbosity = 'medium'  # GPT-4o only supports verbosity 'medium'

  def _generate_image(self, prompt: str) -> str:
    """Generate an image and return it as an inline markdown image tag."""
    try:
      response = self._client.images.generate(
          model=self._image_model_name,
          prompt=prompt,
          n=1,
      )
      b64_data = response.data[0].b64_json
      return f'![image](data:image/png;base64,{b64_data})'
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Image generation failed: %s', e)
      return ''

  def _chat_sample_text(
      self,
      content_parts: list[dict[str, Any]],
      reasoning_effort: str,
      verbosity: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = 1.0,
      top_p: float = language_model.DEFAULT_TOP_P,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    """Call chat completions with pre-parsed multimodal content parts."""
    del top_p, terminators  # Unused for OpenAI models.

    # If all parts are plain text, send a simple string for efficiency.
    if len(content_parts) == 1 and content_parts[0]['type'] == 'text':
      user_content = content_parts[0]['text']
    else:
      user_content = content_parts

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
        {'role': 'user', 'content': user_content},
    ]

    response = self._client.chat.completions.create(
        model=self._model_name,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=timeout,
        seed=seed,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    sample = response.choices[0].message.content or ''
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {'raw_text_length': len(sample)}
      )
    return sample

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
    del top_k  # Unused.

    # Parse inline images from the prompt first, for all code paths.
    content_parts = _parse_prompt_to_content(prompt)

    # Detect image generation requests (from ImageTextActComponent).
    if prompt.lstrip().startswith(_IMAGE_GENERATION_TRIGGER):
      # Extract text-only content as the image generation prompt.
      text_prompt = ' '.join(
          p['text'] for p in content_parts if p['type'] == 'text'
      )
      sample = self._generate_image(text_prompt)
      if self._measurements is not None:
        self._measurements.publish_datum(
            self._channel, {'raw_text_length': len(sample)}
        )
      return sample

    sample = self._chat_sample_text(
        content_parts=content_parts,
        reasoning_effort='minimal',
        verbosity=self._verbosity,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        seed=seed,
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
    prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings:\n'
        + '\n'.join(responses)
        + '.'
    )

    content_parts = _parse_prompt_to_content(prompt)

    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      answer = self._chat_sample_text(
          content_parts,
          reasoning_effort='medium',
          verbosity=self._verbosity,
          temperature=1.0,
          seed=seed,
      )
      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        return idx, responses[idx], {}

    raise language_model.InvalidResponseError(
        f'Too many multiple choice attempts.\nLast attempt: {sample}, '
        + f'extracted: {answer}'
    )

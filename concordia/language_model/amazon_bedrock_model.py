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

"""Language Model that uses Amazon Bedrock models."""

from collections.abc import Collection, Sequence

import boto3
from concordia.language_model import language_model
from concordia.utils import sampling
from concordia.utils.deprecated import measurements as measurements_lib
from typing_extensions import override


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

MODEL_MAX_OUTPUT_TOKENS_LIMITS = {
    'ai21.jamba-instruct-v1:0': 4096,
    'ai21.j2-mid-v1': 8191,
    'ai21.j2-ultra-v1': 8191,
    'amazon.titan-text-express-v1': 8192,
    'amazon.titan-text-lite-v1': 4096,
    'amazon.titan-text-premier-v1:0': 3072,
    'anthropic.claude-v2': 4096,
    'anthropic.claude-v2:1': 4096,
    'anthropic.claude-3-sonnet': 4096,
    'anthropic.claude-3-5-sonnet': 8192,
    'anthropic.claude-3-haiku': 4096,
    'anthropic.claude-3-opus': 4096,
    'cohere.command-text-v14': 4096,
    'cohere.command-light-text-v14': 4096,
    'cohere.command-r-v1:0': 4000,
    'cohere.command-r-plus-v1:0': 4000,
    'meta.llama2-13b-chat-v1': 2048,
    'meta.llama2-70b-chat-v1': 2048,
    'meta.llama3-8b-instruct': 2048,
    'meta.llama3-70b-instruct': 2048,
    'mistral.mistral-7b-instruct-v0:2': 8192,
    'mistral.mixtral-8x7b-instruct-v0:1': 4096,
    'mistral.mistral-large': 8192,
    'mistral.mistral-small': 8192,
}


class AmazonBedrockLanguageModel(language_model.LanguageModel):
  """Language Model that uses Amazon Bedrock models."""

  def __init__(
      self,
      model_name: str,
      *,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The language model to use. For more details, see
        https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._max_tokens_limit = self._get_max_tokens_limit(model_name)

    # AWS credentials are passed via environment variables, see:
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    self._client = boto3.client('bedrock-runtime')

  def _get_max_tokens_limit(self, model_name: str) -> int:
    """Get the max tokens limit for the given model ID."""
    for pattern, value in MODEL_MAX_OUTPUT_TOKENS_LIMITS.items():
      if model_name.startswith(pattern):
        return value
    raise ValueError(f'Unknown model ID: {model_name}')

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
    del timeout, seed  # Unused

    # Use the minimum of max_tokens_limit and the provided max_tokens
    max_tokens = min(self._max_tokens_limit, max_tokens)

    system = [
        {
            'text': (
                'You always continue sentences provided '
                + 'by the user and you never repeat what '
                + 'the user already said.'
            ),
        },
    ]

    messages = [
        {
            'role': 'user',
            'content': [
                {'text': 'Question: Is Jake a turtle?\nAnswer: Jake is '}
            ],
        },
        {'role': 'assistant', 'content': [{'text': 'not a turtle.'}]},
        {
            'role': 'user',
            'content': [{
                'text': (
                    'What is Priya doing right now?\n'
                    + 'Answer: Priya is currently '
                )
            }],
        },
        {'role': 'assistant', 'content': [{'text': 'sleeping.'}]},
        {'role': 'user', 'content': [{'text': prompt}]},
    ]

    # Remove blank stop sequences
    terminators = list(
        filter(lambda terminator: not terminator.isspace(), terminators)
    )

    inference_config = {
        'maxTokens': max_tokens,
        'temperature': temperature,
        'stopSequences': terminators,
    }

    if not terminators:
      del inference_config['stopSequences']

    response = self._client.converse(
        modelId=self._model_name,
        system=system,
        messages=messages,
        inferenceConfig=inference_config,
    )

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {
              'raw_text_length': len(
                  response['output']['message']['content'][0]['text']
              )
          },
      )
    return response['output']['message']['content'][0]['text']

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del seed  # Unused

    prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings:\n'
        + '\n'.join(responses)
        + '.'
    )

    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS
      )

      sample = self.sample_text(
          prompt,
          temperature=temperature,
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

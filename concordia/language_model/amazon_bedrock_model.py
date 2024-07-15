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
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from typing_extensions import override

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class AmazonBedrockLanguageModel(language_model.LanguageModel):
  """Language Model that uses Amazon Bedrock models."""

  def __init__(
      self,
      model_id: str,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_id: The language model to use. For more details, see
        https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._model_id = model_id
    self._measurements = measurements
    self._channel = channel

    # AWS credentials are passed via environment variables, see:
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    self._client = boto3.client('bedrock-runtime')

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

    system = [
        {
            'text': ('You always continue sentences provided ' +
                     'by the user and you never repeat what ' +
                     'the user already said.'),
        },
    ]

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'text': 'Question: Is Jake a turtle?\nAnswer: Jake is '
                }
            ]
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'not a turtle.'
                }
            ]
        },
        {
            'role': 'user',
            'content': [
                {
                    'text': ('What is Priya doing right now?\n' +
                             'Answer: Priya is currently ')
                }
            ]
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'text': 'sleeping.'
                }
            ]
        },
        {
            'role': 'user',
            'content': [
                {
                    'text': prompt
                }
            ]
        }
    ]

    # Remove blank stop sequences
    terminators = list(
        filter(lambda terminator: not terminator.isspace(), terminators))

    inference_config = {
        'maxTokens': max_tokens,
        'temperature': temperature,
        'stopSequences': terminators
    }

    if not terminators:
      del inference_config['stopSequences']

    response = self._client.converse(
        modelId=self._model_id,
        system=system,
        messages=messages,
        inferenceConfig=inference_config
    )

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(
              response['output']['message']['content'][0]['text'])},
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
        + '\n'.join(responses) + '.'
    )

    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

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

    raise language_model.InvalidResponseError(
        (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
         f'extracted: {answer}')
    )

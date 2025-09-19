# Copyright 2024 DeepMind Technologies Limited.
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

"""vLLM Language Model, a wrapper for models running with vLLM inference engine."""

from collections.abc import Collection, Sequence
import json
import os

from concordia.language_model import language_model
from concordia.utils import sampling
from concordia.utils.deprecated import measurements as measurements_lib
from vllm import LLM, SamplingParams
from typing_extensions import override


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TERMINATORS = ()
_DEFAULT_SYSTEM_MESSAGE = (
    'Continue the user\'s sentences. Never repeat their starts. For example, '
    'when you see \'Bob is\', you should continue the sentence after '
    'the word \'is\'. Here are some more examples: \'Question: Is Jake a '
    'turtle?\nAnswer: Jake is \' should be completed as \'not a turtle.\' and '
    '\'Question: What is Priya doing right now?\nAnswer: Priya is currently \' '
    'should be completed as \'working on repairing the sink.\'. Notice that '
    'it is OK to be creative with how you finish the user\'s sentences. The '
    'most important thing is to always continue in the same style as the user.'
)


class VllmLanguageModel(language_model.LanguageModel):
  """Language Model that uses vLLM inference engine."""

  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = _DEFAULT_SYSTEM_MESSAGE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      tensor_parallel_size: int = 1,
      gpu_memory_utilization: float = 0.9,
      max_model_len: int | None = None,
      trust_remote_code: bool = True,
      api_key: str | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The language model to use. Can be a Hugging Face model name
          or local path to a model.
        system_message: System message to prefix to requests when prompting the
          model.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length for the model.
        trust_remote_code: Whether to trust remote code when loading the model.
        api_key: API key for the model.
    """
    self._model_name = model_name
    self._system_message = system_message
    self._terminators = []

    self._measurements = measurements
    self._channel = channel
    self.api_key = api_key

    # Initialize vLLM model
    self._llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = _DEFAULT_TERMINATORS,
      temperature: float = _DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = -1,
      seed: int | None = None,
  ) -> str:
    del timeout, seed  # Unused.

    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'

    terminators = self._terminators + list(terminators)

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=terminators if terminators else None,
    )

    # Generate response
    outputs = self._llm.generate([prompt_with_system_message], sampling_params)
    result = outputs[0].outputs[0].text

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)})

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del seed  # Unused.
    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'
    template = {'choice': '', 'single sentence explanation': ''}
    sample = ''
    answer = ''
    
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

      # Create sampling parameters for JSON response
      sampling_params = SamplingParams(
          temperature=temperature,
          max_tokens=200,  # Reasonable limit for choice responses
          stop=None,
      )

      # Generate response with JSON format instruction
      full_prompt = (f'{prompt_with_system_message}.\n'
                    f'Use the following json template: {json.dumps(template)}.')
      
      outputs = self._llm.generate([full_prompt], sampling_params)
      response_text = outputs[0].outputs[0].text
      
      try:
        json_data_response = json.loads(response_text)
      except json.JSONDecodeError:
        continue
      sample_or_none = json_data_response.get('choice', None)
      if sample_or_none is None:
        if isinstance(json_data_response, dict) and json_data_response:
          sample = next(iter(json_data_response.values()))
        elif isinstance(json_data_response, str) and json_data_response:
          sample = sample_or_none.strip()
        else:
          continue
      else:
        sample = sample_or_none
        if isinstance(sample, str) and sample:
          sample = sample.strip()

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

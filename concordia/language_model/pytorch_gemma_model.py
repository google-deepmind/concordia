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

"""Pytorch Gemma Language Model, for models running on the local machine."""

from collections.abc import Collection, Sequence
import os

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
import numpy as np
import transformers

from typing_extensions import override


class PyTorchGemmaLanguageModel(language_model.LanguageModel):
  """Pytorch Language Model API, for models running on the local machine."""

  def __init__(
      self,
      model_name: str = 'google/gemma-2b-it',
      *,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The local language model to use. For more details,
          see transformers.AutoModelForCausalLM at huggingface.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._tokenizer_name = model_name

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    self._model = transformers.GemmaForCausalLM.from_pretrained(
        self._model_name)
    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
        self._tokenizer_name)

    self._measurements = measurements
    self._channel = channel

    self._text_system_message = (
        'You always continue sentences provided by the user and you never ' +
        'repeat what the user already said.')

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
    del temperature, timeout, seed  # Unused.

    prompt_with_system_message = f'{self._text_system_message}\n\n{prompt}'
    prompt_length = len(prompt_with_system_message)

    inputs = self._tokenizer(prompt_with_system_message, return_tensors='pt')

    generated_tokens = self._model.generate(
        inputs.input_ids,
        max_new_tokens=max_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )

    response = self._tokenizer.decode(
        np.int64(generated_tokens.sequences[0]),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = response[prompt_length:]

    # It would be better to implement terminators in the model generation, but
    # this is a quick way to implement our API for now.
    for terminator in terminators:
      response = response[:response.find(terminator)]

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {'raw_text_length': len(response)}
      )
    return response

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del seed  # Unused.

    inputs = self._tokenizer(prompt, return_tensors='pt')
    generated_tokens = self._model.generate(
        inputs.input_ids,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sample = self._tokenizer.batch_decode(
        [np.argmax(generated_tokens.scores[0][0])],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    answer = sampling.extract_choice_response(sample)
    try:
      idx = responses.index(answer)
      print(f'sample: {sample}, response: {idx}')
    except ValueError:
      raise language_model.InvalidResponseError(
          f'Invalid response: {answer}. '
          f'LLM Input: {prompt}\nLLM Output: {sample}'
      ) from None

    if self._measurements is not None:
      self._measurements.publish_datum(self._channel, {'choices_calls': 1})
    debug = {}
    return idx, responses[idx], debug

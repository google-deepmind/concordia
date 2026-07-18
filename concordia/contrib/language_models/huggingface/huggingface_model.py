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

"""Language Model that uses HuggingFace's transformers library.

To use this model, install concordia with the huggingface extra:
  pip install gdm-concordia[huggingface]
"""

from collections.abc import Collection
from collections.abc import Sequence
import re
from typing import Any, override

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

_DEFAULT_SYSTEM_MESSAGE = (
    'You always continue inputs provided by the user and you never repeat '
    'what the user already said.'
)


class HuggingFaceLanguageModel(language_model.LanguageModel):
  """Language Model that uses HuggingFace's transformer models."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      *,
      trust_remote_code: bool = True,
      dtype: torch.dtype = torch.float16,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The name of the HuggingFace model to use.
      api_key: The API key to use when accessing the HuggingFace API.
      trust_remote_code: Whether to trust remote code.
      dtype: The data type to use for the model.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel

    self._tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=api_key, trust_remote_code=trust_remote_code
    )
    if self._tokenizer.pad_token is None:
      self._tokenizer.pad_token = self._tokenizer.eos_token
    if self._tokenizer.pad_token_id is None:
      self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    self._model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=api_key,
        trust_remote_code=trust_remote_code,
        device_map='auto',
        dtype=dtype,
    )

  @property
  def tokenizer(self) -> AutoTokenizer:
    """Returns the tokenizer instance."""
    return self._tokenizer

  @property
  def device(self) -> torch.device:
    """Returns the device the model is on."""
    return next(iter(self._model.parameters())).device

  def _strip_markdown(self, text_to_strip: str) -> str:
    """Remove markdown code blocks from the text."""
    return re.sub(r'```(?:\w+)?\n?', '', text_to_strip).strip()

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
    if temperature <= 0:
      temperature = 0.1

    messages = [
        {'role': 'system', 'content': _DEFAULT_SYSTEM_MESSAGE},
        {'role': 'user', 'content': prompt},
    ]

    if hasattr(self._tokenizer, 'apply_chat_template'):
      formatted_prompt = self._tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
    else:
      formatted_prompt = f'{_DEFAULT_SYSTEM_MESSAGE}\n\n{prompt}'

    inputs = self._tokenizer(formatted_prompt, return_tensors='pt').to(
        self.device
    )

    generation_args = {
        'max_new_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'do_sample': True,
        'pad_token_id': self._tokenizer.pad_token_id,
        'eos_token_id': self._tokenizer.eos_token_id,
    }
    if seed is not None:
      generation_args['seed'] = seed

    outputs = self._model.generate(**inputs, **generation_args)

    result = self._tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1] :], skip_special_tokens=True
    )
    result = self._strip_markdown(result)

    for term in terminators:
      if term in result:
        result = result.split(term)[0]

    result = result.strip()

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)},
      )

    return result

  def _compute_log_probability(self, prompt: str, response: str) -> float:
    """Computes the log probability of generating the response given the prompt."""
    full_text = prompt + response
    inputs = self._tokenizer(full_text, return_tensors='pt').to(self.device)

    with torch.no_grad():
      outputs = self._model(**inputs)
      logits = outputs.logits

    prompt_inputs = self._tokenizer(prompt, return_tensors='pt').to(self.device)
    prompt_length = prompt_inputs['input_ids'].shape[1]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(prompt_length, inputs['input_ids'].shape[1]):
      token_id = inputs['input_ids'][0, i]
      token_log_prob = log_probs[0, i - 1, token_id].item()
      total_log_prob += token_log_prob

    return total_log_prob

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, Any]]:
    del seed

    log_probs = []
    for response in responses:
      log_prob = self._compute_log_probability(prompt, response)
      log_probs.append(log_prob)

    max_idx = log_probs.index(max(log_probs))

    if self._measurements is not None:
      self._measurements.publish_datum(self._channel, {'choices_calls': 1})

    debug = {response: log_probs[i] for i, response in enumerate(responses)}
    return max_idx, responses[max_idx], debug

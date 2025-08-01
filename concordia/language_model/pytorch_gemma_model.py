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
from concordia.utils import sampling
from concordia.utils.deprecated import measurements as measurements_lib
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
      device: str = 'cpu'
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The local language model to use. For more details,
          see transformers.AutoModelForCausalLM at huggingface.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
        device: Specifies whether to use cpu or cuda for model processing.
    """
    self._model_name = model_name
    self._tokenizer_name = model_name
    self._device = device

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Enhanced loading for Gemma 2 and Gemma 3 compatibility
    self._is_gemma3 = 'gemma-3' in self._model_name.lower()
    
    # Load model with proper configuration for both architectures
    if self._is_gemma3:
      # Gemma 3 models require AutoModelForCausalLM with specific settings
      import torch
      self._model = transformers.AutoModelForCausalLM.from_pretrained(
          self._model_name,
          torch_dtype=torch.float16,
          device_map='auto' if self._device == 'mps' or 'cuda' in self._device else None,
          low_cpu_mem_usage=True,
          trust_remote_code=True
      )
      # Manual device placement for CPU or when device_map not used
      if self._device == 'cpu' or ('cuda' not in self._device and self._device != 'mps'):
        self._model = self._model.to(self._device)
    else:
      # Original Gemma models - backward compatibility
      try:
        self._model = transformers.GemmaForCausalLM.from_pretrained(
            self._model_name).to(self._device)
      except Exception:
        # Fallback to AutoModel if GemmaForCausalLM fails
        import torch
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device != 'cpu' else torch.float32
        ).to(self._device)
    
    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
        self._tokenizer_name)
    
    # Ensure tokenizer has proper padding token for Gemma models
    if self._tokenizer.pad_token is None:
      self._tokenizer.pad_token = self._tokenizer.eos_token

    self._measurements = measurements
    self._channel = channel

    # Different system messages for different model types
    if self._is_gemma3 or '-it' in self._model_name:
      # Instruction-tuned models work better with clear instructions
      self._text_system_message = (
          'You are a helpful assistant. Please provide clear, concise responses ' +
          'in English. Answer questions directly and follow instructions precisely.'
      )
    else:
      # Original completion-style system message for base models
      self._text_system_message = (
          'You always continue sentences provided by the user and you never ' +
          'repeat what the user already said.'
      )

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

    inputs = self._tokenizer(prompt_with_system_message, return_tensors='pt', padding=True)

    # Enhanced generation for Gemma 3 compatibility
    if self._is_gemma3:
      # For Gemma 3 models, use device-aware generation with strict limits
      if hasattr(self._model, 'device'):
        device = self._model.device
      else:
        device = next(self._model.parameters()).device
      
      inputs = {k: v.to(device) for k, v in inputs.items()}
      
      # Conservative but adequate settings for Gemma 3
      generated_tokens = self._model.generate(
          **inputs,
          max_new_tokens=min(max_tokens, 50),  # Allow up to 50 tokens for complete thoughts
          return_dict_in_generate=True,
          output_scores=True,
          do_sample=False,  # Deterministic generation
          pad_token_id=self._tokenizer.eos_token_id,
          eos_token_id=self._tokenizer.eos_token_id,
          use_cache=False,  # Disable caching to avoid layer inconsistencies
          num_beams=1,  # No beam search
          early_stopping=True,  # Stop early if possible
      )
    else:
      # Original generation method for backward compatibility
      generated_tokens = self._model.generate(
          inputs.input_ids.to(self._device),
          max_new_tokens=max_tokens,
          return_dict_in_generate=True,
          output_scores=True,
      )

    response = self._tokenizer.decode(
        np.int64(generated_tokens.sequences[0].cpu()),
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

    # Simplified choice generation for Gemma 3 compatibility

    inputs = self._tokenizer(prompt, return_tensors='pt', padding=True)
    
    # Enhanced choice generation for Gemma 3 compatibility
    if self._is_gemma3:
      # For Gemma 3 models, use device-aware generation
      if hasattr(self._model, 'device'):
        device = self._model.device
      else:
        device = next(self._model.parameters()).device
      
      inputs = {k: v.to(device) for k, v in inputs.items()}
      
      # Simplified single strategy for Gemma 3
      try:
          generated_tokens = self._model.generate(
              **inputs,
              max_new_tokens=1,
              return_dict_in_generate=True,
              output_scores=True,
              do_sample=False,
              pad_token_id=self._tokenizer.eos_token_id,
              eos_token_id=self._tokenizer.eos_token_id,
              use_cache=False,
          )
          
          # Decode the generated tokens
          input_length = inputs['input_ids'].shape[1]
          sample = self._tokenizer.decode(
              generated_tokens.sequences[0][input_length:],
              skip_special_tokens=True,
              clean_up_tokenization_spaces=False
          ).strip()
          
          # If empty, use score-based approach
          if not sample or sample == "":
            if generated_tokens.scores and len(generated_tokens.scores) > 0:
              most_likely_token_id = np.argmax(generated_tokens.scores[0][0].cpu())
              sample = self._tokenizer.decode([most_likely_token_id], skip_special_tokens=True).strip()
            
            # Ultimate fallback
            if not sample:
              sample = "a"
              
      except Exception as e:
          # Emergency fallback
          sample = "a"
          
    else:
      # Original generation method for backward compatibility
      generated_tokens = self._model.generate(
          inputs.input_ids.to(self._device),
          max_new_tokens=1,
          return_dict_in_generate=True,
          output_scores=True,
      )
      # Original decoding method for backward compatibility
      sample = self._tokenizer.batch_decode(
          [np.argmax(generated_tokens.scores[0][0].cpu())],
          skip_special_tokens=True,
          clean_up_tokenization_spaces=False)[0]
    
    answer = sampling.extract_choice_response(sample)
    
    try:
      idx = responses.index(answer)
    except ValueError:
      # Try to find partial matches
      for i, resp in enumerate(responses):
        if resp.lower() in sample.lower() or sample.lower().startswith(resp.lower()):
          idx = i
          answer = resp
          break
      else:
        # If no match found, default to first option
        idx = 0
        answer = responses[0]
        
    if self._measurements is not None:
      self._measurements.publish_datum(self._channel, {'choices_calls': 1})
    debug = {}
    return idx, responses[idx], debug

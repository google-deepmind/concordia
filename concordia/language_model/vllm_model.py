"""Language Model that uses vLLM for local inference.

vLLM is a fast and memory-efficient inference engine for large language models.
This wrapper allows Concordia to use locally hosted models through vLLM.

Example usage:
  model = vllm_model.VLLMLanguageModel(
      model_name="microsoft/DialoGPT-medium",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.8,
  )
"""

from collections.abc import Collection, Sequence
from typing import Any, Mapping

from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib
from typing_extensions import override

try:
  from vllm import LLM, SamplingParams
  from vllm.lora.request import LoRARequest
  VLLM_AVAILABLE = True
except ImportError:
  LLM = None
  SamplingParams = None
  LoRARequest = None
  VLLM_AVAILABLE = False

_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
_DEFAULT_TENSOR_PARALLEL_SIZE = 1
_DEFAULT_ENABLE_PREFIX_CACHING = False
_DEFAULT_MAX_LORA_RANK = 16

class VLLMLanguageModel(language_model.LanguageModel):
  """Language model wrapper for vLLM local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      tensor_parallel_size: int = _DEFAULT_TENSOR_PARALLEL_SIZE,
      gpu_memory_utilization: float = _DEFAULT_GPU_MEMORY_UTILIZATION,
      enable_lora: bool = False,
      lora_path: str | None = None,
      max_model_len: int | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      enable_prefix_caching: bool = _DEFAULT_ENABLE_PREFIX_CACHING,
      max_lora_rank: int = _DEFAULT_MAX_LORA_RANK,
      **kwargs: Any,
  ):
    """Initialize the vLLM language model.

    Args:
      model_name: The name or path of the model to load.
      tensor_parallel_size: Number of GPUs to use for tensor parallelism.
      gpu_memory_utilization: Fraction of GPU memory to use.
      enable_lora: Whether to enable LoRA adapters.
      lora_path: Path to LoRA adapter weights (if enable_lora is True).
      max_model_len: Maximum model context length.
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
      **kwargs: Additional arguments passed to vLLM LLM constructor.

    Raises:
      ImportError: If vLLM is not installed.
      ValueError: If LoRA is enabled but no path is provided.
    """
    if not VLLM_AVAILABLE:
      raise ImportError(
          "vLLM is required but not installed. "
          "Install it with: pip install vllm"
      )

    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._enable_lora = enable_lora
    self._lora_path = lora_path

    # Validate LoRA configuration
    if enable_lora and not lora_path:
      raise ValueError("lora_path must be provided when enable_lora is True")

    # Initialize vLLM model
    llm_kwargs = {
        'model': model_name,
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
        'enable_lora': enable_lora,
        'enable_prefix_caching': enable_prefix_caching,
        'max_lora_rank': max_lora_rank,
        **kwargs
    }

    if max_model_len is not None:
      llm_kwargs['max_model_len'] = max_model_len

    self._llm = LLM(**llm_kwargs)

    # Setup LoRA request if enabled
    self._lora_request = None
    if enable_lora and lora_path:
      self._lora_request = LoRARequest("lora_adapter", 1, lora_path)

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
    """Sample text from the vLLM model."""
    del timeout  # vLLM doesn't support timeout in SamplingParams
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=list(terminators) if terminators else None,
    )

    # Generate response
    outputs = self._llm.generate(
        [prompt],
        sampling_params=sampling_params,
        lora_request=self._lora_request,
    )

    # Extract generated text
    generated_text = outputs[0].outputs[0].text

    # Log statistics
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(generated_text)},
      )

    return generated_text

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Sample a choice from the available responses using log probabilities."""
    
    if not responses:
      raise ValueError("No responses provided to choose from.")

    # Use sampling params that request logprobs
    sampling_params = SamplingParams(
        max_tokens=1,  # We only need logprobs, not generation
        temperature=0.0,
        prompt_logprobs=0,  # Only get logprob of tokens in the prompt
    )
    
    prompts = []
    for response in responses:
      prompts.append(prompt + response)
    
    # Generate to get logprobs (we'll use prompt_logprobs)
    outputs = self._llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=self._lora_request,
    )
    
    logprobs = []
    for i, output in enumerate(outputs):
      if not output.prompt_logprobs:
        raise ValueError("No prompt logprobs returned by vLLM.")
      
      # Find tokens corresponding to the response
      tokenizer = self._llm.get_tokenizer()
      prompt_tokens = tokenizer.encode(prompt)
      full_tokens = tokenizer.encode(prompts[i])
      
      # Response tokens are the difference
      response_start_idx = len(prompt_tokens)
      
      # Sum logprobs for response tokens
      total_logprob = 0.0
      prompt_logprobs = output.prompt_logprobs
      
      for j in range(response_start_idx, len(full_tokens)):
        if j < len(prompt_logprobs) and prompt_logprobs[j]:
          # Get the token ID at this position
          token_id = full_tokens[j]
          if token_id in prompt_logprobs[j]:
            total_logprob += prompt_logprobs[j][token_id].logprob
      
      logprobs.append(total_logprob)
    
    # Find the response with highest log probability
    best_idx = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))
    
    # Create debug info with all scores
    debug_info = {
        'logprobs': {response: logprobs[i]
                     for i, response in enumerate(responses)},
        'method': 'logprobs'
    }
    
    if self._measurements is not None:
        self._measurements.publish_datum(
            self._channel,
            {'choice_method': 'logprobs', 'num_choices': len(responses)},
        )
    
    return best_idx, responses[best_idx], debug_info
  
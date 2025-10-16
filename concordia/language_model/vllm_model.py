"""Language Model that uses vLLM for local inference.

vLLM is a fast and memory-efficient inference engine for large language models.
This wrapper allows Concordia to use locally hosted models through vLLM.

Example usage:
  model = vllm_model.VLLMLanguageModel(
      model_name="microsoft/DialoGPT-medium",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.8,
  )
  
  # Only load the base model once, then create multiple LoRA adapters.
  lora1 = vllm_model.VLLMLora(
      model_name="microsoft/DialoGPT-medium",
      lora_path="/path/to/lora1",
      vllm_language_model=model,
  )
  lora2 = vllm_model.VLLMLora(
      model_name="microsoft/DialoGPT-medium",
      lora_path="/path/to/lora2",
      vllm_language_model=model,
  )
  
  # Load another model with its own LoRA adapter.
  lora3 = vllm_model.VLLMLora(
      model_name="Qwen/Qwen2.5-7B-Instruct",
      lora_path="/path/to/lora3",
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
    self._nbr_lora_adapters = 0 # Number of LoRA adapters used with this model.
    # Each adapter needs a unique ID which is why we keep count.

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
  
  def increment_lora_adapters(self) -> int:
    """Increment the count of LoRA adapters used."""
    if not self._enable_lora:
      raise ValueError("LoRA is not enabled for this model.")
    self._nbr_lora_adapters += 1
    return self._nbr_lora_adapters

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
      lora_request: LoRARequest | None = None,
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
        lora_request=lora_request,
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
      lora_request: LoRARequest | None = None,
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
        lora_request=lora_request,
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

class VLLMLora(language_model.LanguageModel):
  """Language model wrapper for vLLM local inference with LoRA."""

  def __init__(
      self,
      model_name: str,
      *,
      lora_path: str | None = None,
      vllm_language_model: VLLMLanguageModel | None = None,
      tensor_parallel_size: int = _DEFAULT_TENSOR_PARALLEL_SIZE,
      gpu_memory_utilization: float = _DEFAULT_GPU_MEMORY_UTILIZATION,
      enable_lora: bool = False,
      max_model_len: int | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      enable_prefix_caching: bool = _DEFAULT_ENABLE_PREFIX_CACHING,
      max_lora_rank: int = _DEFAULT_MAX_LORA_RANK,
      **kwargs: Any,
  ):
    
    """Initialize the vLLM language model with LoRA.
    
    This is a wrapper around VLLMLanguageModel that passes the LoRA request
    along each sampling call.

    Args:
      model_name: The name or path of the model to load.
      lora_path: Path to LoRA adapter weights (must be provided).
      vllm_language_model: An existing VLLMLanguageModel instance to use.
        If None, a new one is created. If provided, other vLLM parameters are
        ignored.
      tensor_parallel_size: Number of GPUs to use for tensor parallelism.
      gpu_memory_utilization: Fraction of GPU memory to use.
      enable_lora: Whether to enable LoRA adapters.
      max_model_len: Maximum model context length.
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
      enable_prefix_caching: Whether to enable prefix caching in vLLM.
      max_lora_rank: Maximum rank for LoRA adapters.
      **kwargs: Additional arguments passed to vLLM LLM constructor.
    """

    if vllm_language_model is not None:
      self._vllm_model = vllm_language_model
      id = self._vllm_model.increment_lora_adapters()
    else:
      self._vllm_model = VLLMLanguageModel(
          model_name=model_name,
          tensor_parallel_size=tensor_parallel_size,
          gpu_memory_utilization=gpu_memory_utilization,
          enable_lora=enable_lora,
          max_model_len=max_model_len,
          measurements=measurements,
          channel=channel,
          enable_prefix_caching=enable_prefix_caching,
          max_lora_rank=max_lora_rank,
          **kwargs
      )
      id = self._vllm_model.increment_lora_adapters()

    if lora_path is None:
      raise ValueError("lora_path must be provided to initialize VLLMLora.")
    
    # Setup LoRA request
    self._lora_request = LoRARequest(f"lora_adapter_{id}", id, lora_path)
  
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
    """Sample text from the vLLM model with LoRA."""
    return self._vllm_model.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
        lora_request=self._lora_request,
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Sample a choice from the available responses using log probabilities."""
    return self._vllm_model.sample_choice(
        prompt,
        responses,
        seed=seed,
        lora_request=self._lora_request,
    )
  
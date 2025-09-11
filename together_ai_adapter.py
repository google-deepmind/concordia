#!/usr/bin/env python3
"""
Adapter to use Together AI API with Concordia's language model interface.

This allows using open-source models from Together AI (Llama, Mistral, etc.)
instead of OpenAI models.
"""

import os
from typing import Optional, Tuple
import requests
import json
import time

from concordia.language_model import language_model


class TogetherAIModel(language_model.LanguageModel):
    """Together AI adapter for Concordia."""
    
    # Popular models on Together AI
    RECOMMENDED_MODELS = {
        'llama-70b': 'meta-llama/Llama-3.1-70B-Instruct-Turbo',
        'llama-8b': 'meta-llama/Llama-3.1-8B-Instruct-Turbo',
        'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
        'qwen': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
        'deepseek': 'deepseek-ai/deepseek-llm-67b-chat',
    }
    
    def __init__(
        self,
        api_key: str,
        model_name: str = 'llama-70b',
        base_url: str = 'https://api.together.xyz/v1',
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize Together AI model.
        
        Args:
            api_key: Together AI API key
            model_name: Either a shorthand ('llama-70b') or full model ID
            base_url: Together AI API endpoint
            temperature: Sampling temperature
            max_retries: Number of retries for failed requests
            retry_delay: Delay between retries
        """
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Map shorthand to full model name if needed
        if model_name in self.RECOMMENDED_MODELS:
            self.model_name = self.RECOMMENDED_MODELS[model_name]
            print(f"Using model: {self.model_name}")
        else:
            self.model_name = model_name
        
        # Set up headers for API calls
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
    
    def sample_text(
        self,
        prompt: str,
        *,
        max_length: int = 200,
        terminators: Tuple[str, ...] = (),
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text using Together AI API.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            terminators: Stop sequences (not all models support this)
            temperature: Override default temperature
            seed: Random seed for reproducibility
            
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = self.temperature
        
        # Prepare the request
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': max_length,
            'temperature': temperature,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'stop': list(terminators) if terminators else None,
        }
        
        if seed is not None:
            data['seed'] = seed
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f'{self.base_url}/completions',
                    headers=self.headers,
                    json=data,
                    timeout=30,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract generated text
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0].get('text', '').strip()
                    else:
                        return ''
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    if attempt < self.max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {error_msg}")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise Exception(error_msg)
                        
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"Request timed out, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        return ''  # Fallback if all retries failed
    
    def sample_choice(
        self,
        prompt: str,
        responses: Tuple[str, ...],
        *,
        seed: Optional[int] = None,
    ) -> Tuple[int, str]:
        """Sample a choice from given options.
        
        This is a simplified implementation that asks the model to choose.
        """
        # Format prompt with choices
        choice_prompt = f"{prompt}\n\nChoose one of the following options:\n"
        for i, response in enumerate(responses):
            choice_prompt += f"{i+1}. {response}\n"
        choice_prompt += "\nRespond with just the number of your choice:"
        
        # Get response
        response = self.sample_text(
            choice_prompt,
            max_length=10,
            temperature=0.1,  # Low temperature for more deterministic choice
            seed=seed,
        )
        
        # Parse the response to get choice index
        try:
            # Look for a number in the response
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                choice_idx = int(numbers[0]) - 1
                if 0 <= choice_idx < len(responses):
                    return choice_idx, responses[choice_idx]
        except:
            pass
        
        # Default to first option if parsing fails
        return 0, responses[0]


class TogetherAIChat(language_model.LanguageModel):
    """Together AI chat model adapter using the chat completions endpoint."""
    
    RECOMMENDED_MODELS = {
        'llama-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        'llama-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 
        'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
        'qwen': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
    }
    
    def __init__(
        self,
        api_key: str,
        model_name: str = 'llama-8b',
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize Together AI chat model."""
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Map shorthand to full model name
        if model_name in self.RECOMMENDED_MODELS:
            self.model_name = self.RECOMMENDED_MODELS[model_name]
            print(f"Using model: {self.model_name}")
        else:
            self.model_name = model_name
    
    def sample_text(
        self,
        prompt: str,
        *,
        max_length: int = 200,
        terminators: Tuple[str, ...] = (),
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text using Together AI chat API."""
        if temperature is None:
            temperature = self.temperature
        
        # Use together library if available, otherwise use requests
        try:
            import together
            together.api_key = self.api_key
            
            response = together.Complete.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                stop=list(terminators) if terminators else [],
            )
            
            if 'output' in response:
                return response['output']['choices'][0]['text'].strip()
            elif 'choices' in response:
                return response['choices'][0]['text'].strip()
            else:
                return ''
                
        except ImportError:
            # Fallback to requests-based implementation
            url = 'https://api.together.xyz/v1/completions'
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            
            data = {
                'model': self.model_name,
                'prompt': prompt,
                'max_tokens': max_length,
                'temperature': temperature,
                'top_p': 0.9,
                'stop': list(terminators) if terminators else None,
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            return result['choices'][0].get('text', '').strip()
                    elif response.status_code == 429:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        print(f"Error: {e}")
                        return ''
            
            return ''
    
    def sample_choice(
        self,
        prompt: str,
        responses: Tuple[str, ...],
        *,
        seed: Optional[int] = None,
    ) -> Tuple[int, str]:
        """Sample a choice from given options."""
        choice_prompt = f"{prompt}\n\nOptions:\n"
        for i, response in enumerate(responses):
            choice_prompt += f"{i+1}. {response}\n"
        choice_prompt += "\nNumber only:"
        
        response = self.sample_text(
            choice_prompt,
            max_length=10,
            temperature=0.1,
            seed=seed,
        )
        
        try:
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                idx = int(numbers[0]) - 1
                if 0 <= idx < len(responses):
                    return idx, responses[idx]
        except:
            pass
        
        return 0, responses[0]


def create_together_model(
    api_key: Optional[str] = None,
    model: str = 'llama-8b',
    use_chat: bool = False,
) -> language_model.LanguageModel:
    """Convenience function to create a Together AI model.
    
    Args:
        api_key: Together AI API key (uses TOGETHER_API_KEY env var if not provided)
        model: Model shorthand or full model ID
        use_chat: Whether to use chat completions endpoint
        
    Returns:
        Together AI language model instance
    """
    if api_key is None:
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("Together AI API key required. Set TOGETHER_API_KEY environment variable.")
    
    if use_chat:
        return TogetherAIChat(api_key=api_key, model_name=model)
    else:
        return TogetherAIModel(api_key=api_key, model_name=model)


if __name__ == "__main__":
    # Test the adapter
    api_key = os.getenv('TOGETHER_API_KEY')
    if api_key:
        print("Testing Together AI adapter...")
        model = create_together_model(api_key, model='llama-8b')
        
        response = model.sample_text(
            "What is the capital of France? Answer in one word:",
            max_length=10,
            temperature=0.1
        )
        print(f"Response: {response}")
        
        # Test choice sampling
        idx, choice = model.sample_choice(
            "What color is the sky?",
            ("Blue", "Red", "Green"),
        )
        print(f"Choice: {choice} (index {idx})")
    else:
        print("Set TOGETHER_API_KEY to test the adapter")
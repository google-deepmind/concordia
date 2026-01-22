# Language Model

The `language_model` module defines the interface for interacting with Large
Language Models (LLMs) within Concordia. It provides a unified abstraction for
text generation and choice selection, along with utility wrappers for robustness
and debugging.

## Core Interface

### `LanguageModel` (`language_model.py`)
The abstract base class that all model adapters must implement.

*   **`sample_text(prompt, ...)`**: Generates a text response based on a prompt.
*   **`sample_choice(prompt, responses, ...)`**: Selects one option from a list
    of provided responses.

## Wrappers and Utilities

### `RetryLanguageModel` (`retry_wrapper.py`)
Wraps another `LanguageModel` to automatically retry failed calls.

*   Configurable retry attempts, delays, and exponential backoff.
*   Useful for handling transient API errors.

### `CallLimitLanguageModel` (`call_limit_wrapper.py`)
Wraps a model to enforce a maximum number of API calls.

*   Prevents accidental overuse of resources during testing or infinite loops.
*   Returns empty strings or default choices once the limit is reached.

### `ProfiledLanguageModel` (`profiled_language_model.py`)
Wraps a model to collect internal statistics.

*   Tracks token usage or other implementation-specific metrics.
*   Useful for performance monitoring and cost estimation.

### Debugging Models (`no_language_model.py`)

*   **`NoLanguageModel`**: Always returns empty strings and the first choice.
    Useful for unit testing without an actual LLM.
*   **`RandomChoiceLanguageModel`**: Returns random choices.
*   **`BiasedMedianChoiceLanguageModel`**: Biases choices towards the median
    index (useful for testing Likert scale type questions).

## How to Write a Wrapper

You can create custom language model wrappers to integrate with external
providers or to add specialized behavior (e.g., caching, toxic content
filtering).

The `contrib/language_models` directory contains examples of how to wrap common
LLM APIs. The general pattern is:

1.  **Inherit from `LanguageModel`**: Your class must implement the
    `sample_text` and `sample_choice` methods.
2.  **Implement `sample_text`**:

    *   Accept a `prompt` and optional parameters (`max_tokens`, `temperature`,
        etc.).
    *   Call your backend API.
    *   Handle any API-specific errors.
    *   Return the generated text string.
3.  **Implement `sample_choice`**:

    *   Technically optional if you only need text, but required for full
        compatibility.
    *   Many implementations optimize this by scoring the `responses` against
        the `prompt` (e.g., using log-probabilities) rather than asking the
        model to "pick one".
    *   Return a tuple: `(best_index, best_response_text, debug_info_dict)`.

### Example Stubs

```python
from concordia.language_model import language_model

class MyCustomModel(language_model.LanguageModel):

  def sample_text(self, prompt, **kwargs):
    # Call your API here
    result = my_api_client.generate(prompt)
    return result.text

  def sample_choice(self, prompt, responses, **kwargs):
    # Simple inefficient implementation: ask the model to pick
    question = f"{prompt}\nOptions:\n"
    for i, r in enumerate(responses):
        question += f"{i}. {r}\n"
    question += "Answer with the index number only."
    response = self.sample_text(question)
    # Parse response to find index...
    return index, responses[index], {}
```

See
`/concordia/contrib/language_models`
for production-ready examples using OpenAI, Anthropic, and other providers.

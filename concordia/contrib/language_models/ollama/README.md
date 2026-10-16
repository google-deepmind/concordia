# Ollama adapter: thinking models, timeouts and reply length

Use `language_models.language_model_setup(api_type='ollama', model_name=...)`
or construct `ollama_model.OllamaLanguageModel` directly when you need the
options below. With no options the adapter behaves as it always has.

## Thinking models

A thinking model (for example `qwen3`) opens its reply with its thinking.
Ollama applies stop sequences to the thinking as well as the answer, so a
newline terminator, which `InteractiveDocument.open_question` uses, ends the
reply before any answer: every open question comes back empty.

*   `think=True` asks Ollama to keep the thinking apart. The adapter then
    sends no stop sequences to the server and applies the terminators to the
    visible answer itself (`concordia.utils.text.truncate`, as the Gemini
    adapter does). The thinking is never returned as the answer.
*   `think=False` turns thinking off. Replies are much faster, but a small
    thinking model may follow instructions less well without it.
*   `think=None` (the default) sends nothing and leaves the model's own
    setting.

## Timeouts

`timeout` (seconds) is passed to the Ollama client, so a request that takes
too long raises instead of waiting forever, and a retry wrapper such as
`concordia.language_model.retry_wrapper.RetryLanguageModel` can try again.
The default (`None`) waits indefinitely. A thinking reply has no length
bound, and on a host without a GPU one reply can take minutes, so set a
generous timeout together with `think=True`.

`max_tokens` is sent as Ollama's `num_predict`, except with `think=True`:
Ollama counts the kept-apart thinking towards that limit too, so a small
limit would end the thinking before any answer. With `think=None` and a
thinking model the thinking is part of the reply and counts as well.

```python
from concordia.contrib.language_models.ollama import ollama_model
from concordia.language_model import retry_wrapper

model = retry_wrapper.RetryLanguageModel(
    ollama_model.OllamaLanguageModel('qwen3:8b', think=True, timeout=600.0),
    retry_tries=2,
)
```

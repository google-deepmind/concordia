# Together adapter model behavior

Use the existing `language_models.language_model_setup(api_type='together_ai',
model_name='deepseek-ai/DeepSeek-V4.1-Flash', ...)` factory or
`together_ai_model.Base`. For this **exact model name**, the standard DeepSeek
adapter sends the typed Together SDK option `reasoning={'enabled': False}` on
every text request, including requests made by `sample_choice`. This reserves
short completion budgets for visible text. It does not increase the requested
token budget or remove stop sequences. Other model names use their
model-specific request options; the adapter does not infer toggle support from
the DeepSeek family prefix.

The adapter uses the Together SDK's typed `reasoning` parameter. Model
availability depends on the account and endpoint. The SDK controls endpoint
selection, including `TOGETHER_BASE_URL`.

All Together implementations reject missing, non-string, empty or
whitespace-only visible content with `language_model.InvalidResponseError`
immediately. A Gemma reply reduced to empty by a stop sequence also fails. These
responses are not retried by text or choice sampling, and internal reasoning is
never substituted for visible output. Errors do not include prompt, response
body, URL, headers or raw provider exception text.

Permanent failures (including HTTP 401 authentication, 403 permissions and
400/404/422 request/model errors) fail after one adapter request. Only known
transport/timeout failures, HTTP 408/429 and 5xx are retried, at most three
total requests per text sample with short backoff. Exhaustion raises a sanitized
error, never an empty fallback. Choice sampling permits at most three
nonmatching visible answers; provider failures propagate immediately rather than
starting another choice cycle. Request errors propagate without speculative
prompt-trimming retries. The standard SDK client has `max_retries=0`, leaving
the adapter in charge of this retry bound. Injected clients must configure their
own transport retry policy. Callers' timeouts still apply; this is not an
overall monetary or elapsed-time cap.

Model-specific request options include the exact Flash reasoning toggle above.
Error messages identify the provider and numeric status with safe guidance. They
do not assert that an environment variable is misnamed merely because a
credential was rejected.

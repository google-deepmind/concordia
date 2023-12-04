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


"""Language Model that uses Saxml server.

https://github.com/google/saxml
"""

from collections.abc import Collection, Sequence
import concurrent.futures

from concordia.language_model import language_model
from concordia.utils import text
import numpy as np
from saxml.client.python import sax
from scipy import special
from typing_extensions import override

DEFAULT_NUM_CONNECTIONS = 3


class SAXLanguageModel(language_model.LanguageModel):
  """Language Model that uses Saxml server."""

  def __init__(
      self,
      path: str,
      num_conn: int = DEFAULT_NUM_CONNECTIONS,
      deterministic_multiple_choice=False,
  ) -> None:
    """Initializes the instance.

    Args:
      path: sax path of model.
      num_conn: preferred number of connections to sax backend.
      deterministic_multiple_choice: if True, sample_response returns the
        response with max probability instead of sampling.
    """
    options = sax.Options()
    options.num_conn = num_conn
    self._model = sax.Model(path, options).LM()
    self._deterministic_multiple_choice = deterministic_multiple_choice

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    if seed is not None:
      raise NotImplementedError('Unclear how to set seed for sax models.')
    max_tokens = min(max_tokens, max_characters)
    options = sax.ModelOptions()
    options.SetTimeout(timeout)
    options.SetExtraInput('per_example_max_decode_steps', max_tokens)
    options.SetExtraInput('temperature', temperature)
    (sample, _), *_ = self._model.Generate(prompt, options)
    return text.truncate(
        sample, max_length=max_characters, delimiters=terminators
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    scores = self._score_responses(prompt, responses)
    probs = special.softmax(scores)
    entropy = probs @ np.log(probs)
    if self._deterministic_multiple_choice:
      idx = np.argmax(probs, axis=0)
    else:
      idx = np.random.default_rng(seed).choice(len(probs), p=probs)
    debug = {'probs': probs, 'entropy': entropy}
    return idx, responses[idx], debug

  def _score_responses(
      self,
      prompt: str,
      responses: Sequence[str],
  ) -> np.ndarray:
    """Returns the relative log_likelihood of the provided responses.

    Args:
      prompt: the prompt preceding the response.
      responses: the responses to score.

    Returns:
      log Pr(response|prompt)
    """
    if isinstance(responses, str):
      raise TypeError('responses must be a Sequence')

    def get_score(response, model):
      return model.Score(prompt, [response])[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [
          executor.submit(get_score, response, self._model)
          for response in responses
      ]
      scores = [future.result() for future in futures]

    return np.array(list(scores))

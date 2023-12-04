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
"""Google Cloud Language Model."""

from collections.abc import Collection, Sequence

from concordia.language_model import language_model
from concordia.utils import text
from google import auth
from typing_extensions import override
import vertexai
from vertexai.preview import language_models as vertex_models

MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class CloudLanguageModel(language_model.LanguageModel):
  """Language model via a google cloud API."""

  def __init__(
      self,
      project_id: str,
      model_name: str = 'text-bison@001',
      location: str = 'us-central1',
      credentials: auth.credentials.Credentials | None = None,
  ) -> None:
    """Initializes a model instance using the Google Cloud language model API.

    Args:
      project_id: Google Cloud project id in API calls.
      model_name: which language model to use
      location: The location to use when making API calls.
      credentials: Custom credentials to use when making API calls. If not
        provided credentials will be ascertained from the environment.
    """
    if credentials is None:
      credentials, _ = auth.default()
    vertexai.init(
        project=project_id, location=location, credentials=credentials
    )
    self._model = vertex_models.TextGenerationModel.from_pretrained(model_name)

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
      raise NotImplementedError('Unclear how to set seed for cloud models.')

    max_tokens = min(max_tokens, max_characters)
    sample = self._model.predict(
        prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return text.truncate(
        sample.text, max_length=max_characters, delimiters=terminators
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    max_characters = max([len(response) for response in responses])

    for _ in range(MAX_MULTIPLE_CHOICE_ATTEMPTS):
      sample = self.sample_text(
          prompt,
          max_tokens=1,
          max_characters=max_characters,
          temperature=0.0,
          seed=seed,
      )
      try:
        idx = responses.index(sample)
      except ValueError:
        continue
      else:
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError(
        'Too many multiple choice attempts.'
    )

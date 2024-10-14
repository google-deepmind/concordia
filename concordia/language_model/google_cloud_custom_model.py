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

"""Language Model that uses Google Cloud Vertex AI API.

Recommended model names are:
  'gemma-2-9b-it'
  'gemma-2-27b-it'
"""

# Before running:
# 1. Create a Google Cloud project and enable the Vertex AI API.
# 2. Authenticate with Google Cloud credentials
# 3. Find the model you want to use and upload it to the project.
# 4. Deploy the model to an endpoint in the region.
# 5. Replace model_name, project, and location with your actual values.

from collections.abc import Collection, Sequence
import random
import time
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from google.cloud import aiplatform
from typing_extensions import override

_MAX_ATTEMPTS = 20
_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_NUM_SILENT_ATTEMPTS = 3
_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED = 2
_JITTER_SECONDS = 0.25
_DEFAULT_MAX_TOKENS = 5000  # Adjust as needed for the specific model


class VertexAI(language_model.LanguageModel):
  """Language Model that uses Google Cloud Vertex AI models."""

  def __init__(
      self,
      model_name: str,
      *,
      project: str,
      location: str,  # e.g., "us-central1"
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The language model to use.  E.g., "gemma2-9b-it"
      project: Your Google Cloud project ID.
      location: The region where the model is deployed.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._project = project
    self._location = location
    self._measurements = measurements
    self._channel = channel
    aiplatform.init(project=project, location=location)

  @override
  # sample_text:
  # Uses aiplatform.Client().predict to make prediction requests.
  # Might need to adjust response[0]["content"] based on the actual structure).
  # Includes top_p and top_k parameters as examples
  # Add max_output_tokens which corresponds to Together's max_tokens.
  # Applies terminators after generation. Because Vertex AI doesn't support
  # terminators during generation as post-processing step.
  # It will find the last instance of a terminator and truncate there.
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,  # Vertex doesn't directly support seed.
  ) -> str:

    max_tokens = min(max_tokens, _DEFAULT_MAX_TOKENS)

    result = ""
    for attempts in range(_MAX_ATTEMPTS):
      if attempts > 0:
        seconds_to_sleep = _SECONDS_TO_SLEEP_WHEN_RATE_LIMITED + random.uniform(
            -_JITTER_SECONDS, _JITTER_SECONDS
        )
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(
              f"Sleeping for {seconds_to_sleep} seconds... attempt:"
              f" {attempts} / {_MAX_ATTEMPTS}"
          )
        time.sleep(seconds_to_sleep)

      try:
        response = (
            aiplatform.PredictionServiceClient()
            .predict(
                [{"content": prompt}],
                parameters={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,  # Example: Add other parameters as needed
                    "top_k": 40,
                },
            )
            .predictions[0]
        )
        result = response[0][
            "content"
        ]  # Adjust based on API response structure

        # Apply terminators
        for terminator in terminators:
          if terminator in result:
            result = result[: result.index(terminator)] + terminator
            break

        break  # Success, exit the retry loop
      except Exception as err:  # pylint: disable=broad-exception-caught
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(f"  Exception: {err}")
        continue

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {"raw_text_length": len(result)},
      )

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,  # Seed is not supported by Vertex AI.
  ) -> tuple[int, str, dict[str, float]]:
    prompt = (
        prompt
        + "\nRespond EXACTLY with one of the following strings:\n"
        + "\n".join(responses)
        + "."
    )

    sample = ""
    answer = ""
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS
      )

      sample = self.sample_text(
          prompt,
          temperature=temperature,
          seed=seed,
      )
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {"choices_calls": attempts}
          )
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError((
        f"Too many multiple choice attempts.\nLast attempt: {sample}, "
        + f"extracted: {answer}"
    ))

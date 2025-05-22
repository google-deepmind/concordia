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

"""Language Model that uses Google Cloud Vertex AI API."""

from collections.abc import Collection, Sequence
import random
import time

from concordia.language_model import language_model
from concordia.utils import sampling
from concordia.utils.deprecated import measurements as measurements_lib
from google.cloud import aiplatform
from typing_extensions import override


_MAX_ATTEMPTS = 20
_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_NUM_SILENT_ATTEMPTS = 3
_SECONDS_TO_SLEEP_WHEN_RATE_LIMITED = 2
_JITTER_SECONDS = 0.25
_DEFAULT_MAX_TOKENS = 5000  # Adjust as needed for the specific model


def _wrap_prompt(prompt: str) -> str:
  """Wraps a prompt with the default conditioning.

  Args:
    prompt: the prompt to wrap

  Returns:
    the prompt wrapped with the default conditioning
  """
  turns = []
  turns.append(
      "<start_of_turn>system You always continue sentences provided by the "
      "user and you never repeat what the user already said.<end_of_turn>"
  )
  turns.append(
      "<start_of_turn>user Question: Is Jake a turtle?\n"
      "Answer: Jake is <end_of_turn>"
  )
  turns.append(
      "<start_of_turn>model not a turtle.<end_of_turn>"
  )
  turns.append(
      "<start_of_turn>user Question: What is Priya doing right now?\n"
      "Answer: Priya is currently <end_of_turn>"
  )
  turns.append(
      "<start_of_turn>model sleeping.<end_of_turn>"
  )
  turns.append(
      "<start_of_turn>user Question:\n"
      + prompt
      + "<end_of_turn>"
  )
  turns.append(
      "<start_of_turn>model "
  )
  return "\n".join(turns)


class VertexAI(language_model.LanguageModel):
  """Language Model that uses Google Cloud Vertex AI models.

  you need endpoint_id, project_id, region (location) info.
  you can find the project_id in your Google Cloud's main console.
  you can find the endpoint_id, and region in the Vertex AI model registry page.

  the quickest way to find these info at once is to go to the Vertex AI model
  registry page in Google Cloud, and click on the model you want to use.
  Then, click on the Sample Request link, it'll open a panel on the right,
  click on the PYTHON tab, under instruction number 3, you'll see all three
  project_id, endpoint_id and region info.
  """

  def __init__(
      self,
      endpoint_id: str,  # all numbers
      *,
      project_id: str,  # all numbers
      location: str = "us-central1",
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance."""
    self._endpoint_id = endpoint_id
    self._project_id = project_id
    self._location = location
    self._measurements = measurements
    self._channel = channel

    # Initialize the client and endpoint name *once*
    self._endpoint_name = f"projects/{self._project_id}/locations/{self._location}/endpoints/{self._endpoint_id}"
    self._api_endpoint = f"{self._location}-aiplatform.googleapis.com"
    self._client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": self._api_endpoint}
    )
    self._parameters = {
        "top_p": 0.95,
        "top_k": 40,
    }

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
    max_tokens = min(max_tokens, _DEFAULT_MAX_TOKENS)
    self._parameters["temperature"] = temperature
    self._parameters["max_output_tokens"] = max_tokens

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
        response = self._client.predict(
            endpoint=self._endpoint_name,
            instances=[{
                "inputs": _wrap_prompt(prompt)
            }],
            parameters=self._parameters,
        ).predictions[0]

        result = response

        # Apply terminators
        for terminator in terminators:
          if terminator in result:
            result = result[: result.index(terminator)] + terminator
            break

        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel,
              {"raw_text_length": len(result)},
          )

        return result

      except Exception as err:  # pylint: disable=broad-exception-caught
        if attempts >= _NUM_SILENT_ATTEMPTS:
          print(f"  Exception: {err}")

    raise RuntimeError(
        f"Failed to get a response after {_MAX_ATTEMPTS} attempts."
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
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

      sample = self.sample_text(prompt, temperature=temperature, seed=seed)

      # clean up the sample from newlines and spaces
      sample = sample.replace("\n", "").replace(" ", "")
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {"choices_calls": attempts}
          )
        return idx, responses[idx], {}
      except ValueError:
        pass

    raise language_model.InvalidResponseError((
        f"Too many multiple choice attempts.\nLast attempt: {sample}, "
        f"extracted: {answer}"
    ))

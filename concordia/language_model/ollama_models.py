# Copyright 2024 DeepMind Technologies Limited.
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

"""Ollama Language Model."""

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import langchain.llms as llms
import langchain.callbacks.manager as callback_manager
import langchain.callbacks.streaming_stdout as streaming_stdout
from collections.abc import Collection, Sequence
from string import Template
from typing_extensions import override
import re

_PROMPT_TEMPLATE = Template("$system_message\n\n$message")

_TEXT_SYSTEM_MESSAGE = ""

_MULTIPLE_CHOICE_SYSTEM_MESSAGE = """The following is a multiple choice question.
You must always respond with one of the possible choices.
You may not respond with anything else.
"""


def extract_choices(text):
    match = re.search(r"\(?(\w)\)", text)
    if match:
        return match.group(1)
    return None


class OllamaLanguageModel(language_model.LanguageModel):
    """Language Model that uses Ollama LLM models."""

    def __init__(
        self,
        model_name: str,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
        streaming: bool = False,
    ):
        """Initializes the instance.

        Args:
          model_name: The language model to use.
          measurements: The measurements object to log usage statistics to.
          channel: The channel to write the statistics to.
        """
        self._model_name = model_name
        self._measurements = measurements
        self._channel = channel
        if streaming:
            self._client = llms.Ollama(
                model=model_name,
                callback_manager=callback_manager.CallbackManager(
                    [streaming_stdout.StreamingStdOutCallbackHandler()]
                ),
            )
        else:
            self._client = llms.Ollama(model=model_name)

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
        system_message=_TEXT_SYSTEM_MESSAGE,
    ) -> str:

        message = prompt
        prompt = _PROMPT_TEMPLATE.substitute(
            system_message=system_message, message=message
        )

        response = self._client(
            prompt,
            stop=terminators,
            temperature=temperature,
        )

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {"raw_text_length": len(response)},
            )
        return response

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
        system_message=_MULTIPLE_CHOICE_SYSTEM_MESSAGE,
    ) -> tuple[int, str, dict[str, float]]:
        max_characters = len(max(responses, key=len))
        sample = self.sample_text(
            prompt,
            max_characters=max_characters,
            temperature=0.0,
            seed=seed,
            system_message=system_message,
        )
        answer = extract_choices(sample)
        try:
            idx = responses.index(answer)
        except ValueError:
            raise language_model.InvalidResponseError(
                f"Invalid response: {answer}. LLM Input: {prompt}\nLLM Output: {sample}"
            )
        else:
            if self._measurements is not None:
                self._measurements.publish_datum(self._channel, {"choices_calls": 1})
            debug = {}
            return idx, responses[idx], debug

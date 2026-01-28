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


"""Groq language model wrapper following BaseGPTModel style."""

try:
    from groq import Groq
except ImportError:
    Groq = None  

from collections.abc import Collection, Sequence
from typing import Any, override

from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class GroqModel(language_model.LanguageModel):

    def __init__(
        self,
        model_name: str,
        api_key: str,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
    ):
        if Groq is None:
            raise ImportError(
                "Groq dependency not installed. Install with: pip install gdm-concordia[groq]"
            )

        self._model_name = model_name
        self._client = Groq(api_key=api_key)
        self._measurements = measurements
        self._channel = channel


    def _sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:

        del terminators

        messages = [
            {
                "role": "system",
                "content": (
                    "You always continue sentences provided "
                    "by the user and you never repeat what "
                    "the user already said."
                ),
            },
            {
                "role": "user",
                "content": "Question: Is Jake a turtle?\nAnswer: Jake is ",
            },
            {"role": "assistant", "content": "not a turtle."},
            {
                "role": "user",
                "content": (
                    "Question: What is Priya doing right now?\nAnswer: "
                    "Priya is currently "
                ),
            },
            {"role": "assistant", "content": "sleeping."},
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            seed=seed,
        )

        text = response.choices[0].message.content

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel, {"raw_text_length": len(text)}
            )

        return text

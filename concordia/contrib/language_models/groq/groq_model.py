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

from collections.abc import Collection, Sequence
from typing import Any
from typing import override


from groq import Groq

from concordia.language_model import language_model


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class GroqModel(language_model.LanguageModel):
    """Groq wrapper with same structure as BaseGPTModel."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
    ):
        self._model_name = model_name
        self._client = Groq(api_key=api_key)


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
        del terminators, timeout, seed  # unused by Groq

        messages = [
            {
                "role": "system",
                "content": (
                    "You always continue sentences provided "
                    "by the user and never refuse to answer."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        return response.choices[0].message.content

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        del top_k  # unused

        return self._sample_text(
            prompt=prompt,
            max_tokens=max_tokens,
            terminators=terminators,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            seed=seed,
        )

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, Any]]:

        prompt = (
            prompt
            + "\nRespond EXACTLY with one of the following strings:\n"
            + "\n".join(responses)
            + "."
        )

        answer = ""
        for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
            answer = self._sample_text(
                prompt,
                temperature=0,
                seed=seed,
            ).strip()

            try:
                idx = responses.index(answer)
            except ValueError:
                continue
            else:
                return idx, responses[idx], {}

        raise language_model.InvalidResponseError(
            f"Too many multiple choice attempts. Last answer: {answer}"
        )
# Copyright 2026 DeepMind Technologies Limited.
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

"""A language model wrapper that answers multiple choice with a reason first.

`InteractiveDocument.multiple_choice_question` lists lettered options and
asks the model to `sample_choice` a letter. Some providers implement
`sample_choice` by asking for the letter at once (for example inside a JSON
object whose first field is the choice), which leaves a small local model no
room to reason: it can explain the right answer in free text yet pick a
wrong letter when made to answer bare.

`ReasonedChoice` answers `sample_choice` through `sample_text` instead: it
asks for one sentence of reasoning and then the answer as "(letter)" with
the option's words, takes the last option letter in the reply, and falls
back to the wrapped model's own `sample_choice` when no option letter comes
back. If the words after that letter are another option's words, the words
win: a model can reason its way to an answer and then slip on the letter.
`sample_text` passes through unchanged.

Example:

```python
from concordia.contrib import language_models
from concordia.contrib.language_models import reasoned_choice

model = reasoned_choice.ReasonedChoice(
    language_models.language_model_setup(
        api_type='ollama', model_name='qwen3:8b'
    )
)
```

Measured on one game's golden set of rule-following questions (21 cases, 10
of them strict rules), qwen3:8b through Ollama kept 4 of the 10 strict rules
when answering bare and 10 of 10 with `ReasonedChoice`.
"""

from collections.abc import Collection, Mapping, Sequence
import re
from typing import Any, override

from concordia.language_model import language_model

_LETTER = re.compile(r'\(([a-z])\)')
_OPTION = re.compile(r'^\s*\(([a-z])\)\s*(.+?)\s*$', re.MULTILINE)
DEFAULT_INSTRUCTION = (
    'In one sentence, say why; then give the answer as (letter) and the'
    " option's words."
)


def _options(prompt: str) -> dict[str, str]:
  """The lettered options listed in a prompt: letter -> words, lowercased."""
  return {k: v.lower().rstrip('.') for k, v in _OPTION.findall(prompt)}


class ReasonedChoice(language_model.LanguageModel):
  """Wraps a model so that multiple choice goes through a short reason."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      *,
      instruction: str = DEFAULT_INSTRUCTION,
      max_tokens: int = 160,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
  ):
    """Initializes the wrapper.

    Args:
      model: The model to wrap.
      instruction: Appended to a multiple-choice prompt; it must ask for the
        answer as a parenthesized option letter.
      max_tokens: The most tokens the reason and answer may take.
      temperature: The sampling temperature of the reason and answer. A low
        temperature makes a judgement repeatable.
    """
    self._model = model
    self._instruction = instruction
    self._max_tokens = max_tokens
    self._temperature = temperature

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
    return self._model.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
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
  ) -> tuple[int, str, Mapping[str, Any]]:
    question = prompt.rstrip()
    # InteractiveDocument ends a multiple-choice prompt with 'Answer: ('.
    if question.endswith('('):
      question = question[:-1].rstrip()
    if question.endswith('Answer:'):
      question = question[: -len('Answer:')].rstrip()
    reply = self._model.sample_text(
        f'{question}\n{self._instruction}',
        max_tokens=self._max_tokens,
        terminators=(),
        temperature=self._temperature,
        seed=seed,
    )
    found = [m for m in _LETTER.finditer(reply) if m.group(1) in responses]
    if found:
      letter = found[-1].group(1)
      after = reply[found[-1].end() :].lower()
      options = _options(prompt)
      if options.get(letter) and options[letter] not in after:
        # The words name another option: the letter slipped.
        named = [k for k, v in options.items() if v and v in after]
        if len(named) == 1 and named[0] in responses:
          letter = named[0]
      idx = list(responses).index(letter)
      return idx, responses[idx], {'reasoning': reply}
    return self._model.sample_choice(prompt, responses, seed=seed)

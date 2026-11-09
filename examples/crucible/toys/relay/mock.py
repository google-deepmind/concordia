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

"""The mock model's answers for the relay toy, read from prompts alone.

It hears tone by counting warm and cold words (data/relay.yaml), and plays a
party by opening low and conceding a unit a message, in words that follow
its warmth. This is the lookup-table floor of the toy's semantics.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.crucible.toys.relay import toy as toy_lib

_MESSAGE = re.compile(r'has just reached [^:]+:\n"(.*)"\n', re.DOTALL)
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_SPEAKER = re.compile(re.escape(toy_lib.SPEAK_MARKER) + r' (\w+), (\w+):')
_WARMTH = re.compile(r'\(warmth (\d\.\d+)\)')
_LAST = re.compile(r'Your last offer gave \w+ (\d+) of')
_WORD = re.compile(r'[a-z]+')


class RelayMock:
  """Answers the relay toy's prompts deterministically."""

  def __init__(self, data: Mapping[str, Any]):
    self._data = data
    self._warm = frozenset(data['lexicon']['warm'])
    self._cold = frozenset(data['lexicon']['cold'])

  def tone_score(self, text: str) -> int:
    words = _WORD.findall(text.lower())
    return sum(w in self._warm for w in words) - sum(
        w in self._cold for w in words
    )

  def reading(self, text: str) -> int:
    """Index into TONE_OPTIONS."""
    score = self.tone_score(text)
    if score >= 2:
      return 0
    if score <= -3:
      return 3
    if score <= -1:
      return 2
    return 1

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    if 'after reading it?' not in prompt:
      return None
    match = _MESSAGE.search(prompt)
    if not match:
      return None
    wanted = toy_lib.TONE_OPTIONS[self.reading(match.group(1))]
    letters = {text.strip(): key for key, text in _OPTION.findall(prompt)}
    letter = letters.get(wanted)
    return list(responses).index(letter) if letter in responses else None

  def text(self, prompt: str) -> str | None:
    if toy_lib.RELAY_MARKER in prompt and toy_lib.SPEAK_MARKER not in prompt:
      return 'pass'
    found = _SPEAKER.findall(prompt)
    if not found:
      return None
    other, me = found[-1]
    party = self._data['parties'].get(me, {})
    pot = int(self._data['pot'])
    warmth_found = _WARMTH.findall(prompt)
    warmth = float(warmth_found[-1]) if warmth_found else 0.5
    last_found = _LAST.findall(
        prompt[prompt.rfind(toy_lib.SPEAK_MARKER) - 300 :]
    )
    if last_found:
      offer = min(pot, int(last_found[-1]) + int(party.get('concede', 1)))
    else:
      offer = int(party.get('opening', 3))
    if warmth >= 0.6:
      return (
          f'Thank you, {other}. I offer you {offer} of {pot}; I think that is'
          ' fair.'
      )
    if warmth <= 0.35:
      return f'I insist: {offer} of {pot} for you, and that is final.'
    return f'I offer you {offer} of {pot}.'

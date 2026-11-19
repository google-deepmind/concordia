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

"""A lookup-table reading of claims: the mock's stand-in for a language model.

Two claims are related when they share a content word that carries no
polarity. Each claim has a polarity: the product of the poles of its polar
words (safe +1, collapse -1, ...) times -1 for each negation. Related claims
of the same polarity agree; of opposite polarity, they conflict.

This is deliberately the floor of the toy's semantics: it is what a lookup
table can do, and the reason the game needs a language model. The words live
in `data/claims.yaml` (`lexicon`).
"""

from collections.abc import Mapping, Sequence
import re

from examples.crucible.toys.claims import world as world_lib

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


class Lexicon:
  """Content words and polarity, read from data."""

  def __init__(
      self,
      stopwords: Sequence[str],
      negations: Sequence[str],
      poles: Mapping[str, Sequence[str]],
  ):
    """Initializes the lexicon.

    Args:
      stopwords: Words that carry no content.
      negations: Words that flip a claim's polarity.
      poles: Concept name -> words; a word prefixed with '-' is the concept's
        negative pole, e.g. {'safety': ['safe', '-dangerous']}.
    """
    self._stop = frozenset(stopwords)
    self._negations = frozenset(negations)
    self._pole: dict[str, int] = {}
    for words in poles.values():
      for word in words:
        if word.startswith('-'):
          self._pole[word[1:]] = -1
        else:
          self._pole[word] = 1

  def words(self, text: str) -> list[str]:
    words = _WORD.findall(text.lower().replace('’', "'"))
    return [w[:-2] if w.endswith("'s") else w for w in words]

  def stem(self, word: str) -> str:
    for suffix in ('ing', 'ed', 'es', 's'):
      if len(word) > len(suffix) + 2 and word.endswith(suffix):
        return word[: -len(suffix)]
    return word

  def content(self, text: str) -> set[str]:
    """Stems of the words that carry content but no polarity."""
    out = set()
    for word in self.words(text):
      if word in self._stop or self._is_negation(word):
        continue
      if word in self._pole or self.stem(word) in self._pole:
        continue
      out.add(self.stem(word))
    return out

  def _is_negation(self, word: str) -> bool:
    return word in self._negations or word.endswith("n't")

  def polarity(self, text: str) -> int:
    sign = 1
    for word in self.words(text):
      if self._is_negation(word):
        sign = -sign
      elif word in self._pole:
        sign *= self._pole[word]
      elif self.stem(word) in self._pole:
        sign *= self._pole[self.stem(word)]
    return sign

  def relation(self, a: str, b: str) -> str:
    """AGREE, CONFLICT, RESTATES or UNRELATED, by words and polarity."""
    shared = self.content(a) & self.content(b)
    if not shared:
      return world_lib.UNRELATED
    if self.polarity(a) != self.polarity(b):
      return world_lib.CONFLICT
    if self.content(a) == self.content(b):
      return world_lib.RESTATES
    return world_lib.AGREE

  def negate(self, text: str) -> str:
    """A claim that the lexicon reads as conflicting with `text`."""
    body = text.strip().rstrip('.')
    return f'It is false that {body[:1].lower()}{body[1:]}.'

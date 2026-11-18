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

"""Helper functions for language model sampling."""

from collections.abc import Collection, Iterator
import itertools
import re
import string


def choice_labels() -> Iterator[str]:
  """Yields a, ..., z, aa, ..., az, ba, ... without a choice-count limit."""
  for length in itertools.count(1):
    for letters in itertools.product(string.ascii_lowercase, repeat=length):
      yield ''.join(letters)


def extract_choice_response(
    sample: str, responses: Collection[str] | None = None
) -> str | None:
  """Extracts one unambiguous choice, without matching word fragments.

  Accepts an exact response, a response followed by a period or closing
  parenthesis, or a parenthesized response in text. Repeated occurrences of
  the same choice are allowed; distinct choices are ambiguous. Matching is
  case sensitive. Bare choices embedded in prose are not extracted.

  Args:
    sample: The model's output.
    responses: Valid response strings, including multi-character labels. When
      omitted, only single-character choices are recognized, as in the legacy
      interface. An empty collection accepts nothing.
  """
  sample = sample.strip()
  choices = set(responses) if responses is not None else None

  def is_choice(value: str) -> bool:
    if choices is None:
      return len(value) == 1
    return bool(value) and value in choices

  if is_choice(sample):
    return sample
  if sample.endswith('.') and is_choice(sample[:-1]):
    return sample[:-1]

  # Consume complete parenthesized strings first. Bare labels must start at
  # whitespace or the beginning, so contractions and hyphenated words do not
  # yield suffix choices. Both punctuation forms count towards ambiguity.
  matches = {
      (parenthesized or unopened).strip()
      for parenthesized, unopened in re.findall(
          r'\(([^()]*)\)|(?<!\S)(\w+)(?:\)|\.(?=\s|$))', sample
      )
      if is_choice((parenthesized or unopened).strip())
  }
  return matches.pop() if len(matches) == 1 else None


def dynamically_adjust_temperature(
    attempts: int,
    max_attempts: int,
) -> float:
  """Adjusts choice sampling temperature based on number of attempts so far."""
  # Increase temperature after the first failed attempt.
  temperature = 0.0
  if attempts > 1 and attempts < (max_attempts / 2.0):
    temperature = 0.5
  elif attempts >= (max_attempts / 2.0):
    temperature = 0.75
  return temperature

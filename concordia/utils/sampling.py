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

"""Helper functions for language model sampling.
"""

import re


def _extract_parenthesized_choice(sample: str):
  """Given text formatted as 'lorum(a)ipsum', return 'a'."""
  match = re.search(r'\(?(\w)\)', sample)
  if match:
    return match.group(1)
  else:
    return None


def extract_choice_response(sample: str) -> str | None:
  """Given a sample such as "a", "a)", or "foo(a)bar, return the choice."""
  if len(sample) == 1:
    # i.e. this would be a sample such as "a"
    return sample
  elif len(sample) == 2:
    # i.e. this would be a sample such as "a)"
    return sample[0]
  else:
    # extract a substring like "(a)" wherever it may be in a longer string
    return _extract_parenthesized_choice(sample)


def dynamically_adjust_temperature(
    attempts: int,
    max_attempts: int,
) -> float:
  """Adjusts choice sampling temperature based on number of attempts so far."""
  # Increase temperature after the first failed attempt.
  temperature = 0.0
  if attempts > 1 and attempts < (max_attempts / 2.0):
    temperature = 0.5
  elif attempts > (max_attempts / 2.0):
    temperature = 0.75
  return temperature

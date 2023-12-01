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


"""String formatting utilities."""

from collections.abc import Collection
import sys
import textwrap


def wrap(string: str, width: int = 70) -> str:
  """Returns the string wrapped to the specified width."""
  lines = string.split('\n')
  wrapped_lines = (textwrap.fill(line, width=width) for line in lines)
  return '\n'.join(wrapped_lines)


def truncate(
    string: str,
    *,
    max_length: int = sys.maxsize,
    delimiters: Collection[str] = (),
) -> str:
  """Truncates a string.

  Args:
    string: string to truncate
    max_length: maximum length of the string.
    delimiters: delimiters that must not be present in the truncated string.

  Returns:
    The longest prefix of string that does not exceed max_length and does not
    contain any delimiter.
  """
  truncated = string[:max_length]
  for delimiter in delimiters:
    truncated = truncated.split(delimiter, 1)[0]
  return truncated

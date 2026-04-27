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

"""Markdown stripping utilities for language model responses."""

import re


def strip_markdown(text: str) -> str:
  """Strips markdown code blocks and formatting from text.

  This function removes:
  - Triple backtick code blocks (```...```)
  - Single backtick inline code (`...`)
  - Leading and trailing whitespace

  Args:
    text: The text to strip markdown from.

  Returns:
    The text with markdown formatting removed.
  """
  result = text

  result = re.sub(r'```[\w]*\n', '\n', result)
  result = re.sub(r'```$', '', result, flags=re.MULTILINE)
  result = re.sub(r'\n\n+', '\n', result)

  result = re.sub(r'`([^`]+)`', r'\1', result)

  result = result.strip()

  return result
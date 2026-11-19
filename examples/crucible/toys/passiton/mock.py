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

"""Pass it on's mock: the orders mock already answers every prompt it asks.

A hop is the orders toy's relay prompt (the mock reads the law it states)
and the last in line read by the orders toy's reading prompts, so this
handler recognises nothing of its own.
"""

from collections.abc import Mapping, Sequence
from typing import Any


class PassItOnMock:
  """A handler that leaves every prompt to the orders mock."""

  def __init__(self, data: Mapping[str, Any]):
    self._data = data

  def text(self, prompt: str) -> str | None:
    del prompt
    return None

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    del prompt, responses
    return None

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

"""News that knows where it happened.

A `Located` is a plain string (so everything that reads news reads it as
before) that also carries the tile it happened on. The planet publishes such
news on its region's local topic too, and the peoples hear it later the
farther away they are.
"""


class Located(str):
  """News text with the tile it happened on (row from the south, column)."""

  where: tuple[int, int] | None = None


def located(text: str, where: tuple[int, int]) -> Located:
  """Text that happened at a tile."""
  out = Located(text)
  out.where = (int(where[0]), int(where[1]))
  return out


def place_of(text: str) -> tuple[int, int] | None:
  """Where a piece of news happened, if it says."""
  return getattr(text, 'where', None)

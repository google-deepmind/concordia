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

"""Mapping language back to rules: options with a key and a label.

Behaviours and adaptations (for daisies and for grazers alike) are offered to
models by their labels. These helpers find which one a chosen option or a
free-text answer refers to.
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar


class Labelled(Protocol):

  @property
  def key(self) -> str:
    ...

  @property
  def label(self) -> str:
    ...


T = TypeVar('T', bound=Labelled)


def by_key(items: Sequence[T], key: str) -> T:
  """The item with this key (raises StopIteration if there is none)."""
  return next(item for item in items if item.key == key)


def by_label_prefix(items: Sequence[T], text: str) -> T | None:
  """The item whose label starts a chosen option's text, if any."""
  text = text.strip().lower()
  return next(
      (item for item in items if text.startswith(item.label.lower())), None
  )


def earliest_named(items: Sequence[T], text: str) -> T | None:
  """The item whose label appears first in a free-text answer, if any.

  Ties (labels found at the same position) go to the earlier item.

  Args:
    items: Candidate items.
    text: The answer.

  Returns:
    The item, or None if no label appears.
  """
  lowered = text.lower()
  found = [
      (lowered.find(item.label.lower()), i, item)
      for i, item in enumerate(items)
      if item.label.lower() in lowered
  ]
  if not found:
    return None
  return min(found, key=lambda x: (x[0], x[1]))[2]

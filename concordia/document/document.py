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


"""A document that is built from a chain of text."""

from collections.abc import Collection, Iterable, Iterator, Set
import contextlib
import dataclasses
from typing import TypeVar

T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class Content:
  """Content appended to a document.

  Attributes:
    text: the text of the content
    hidden: if True the content should be hidden from the reader
    tags: tags provided at time this was written to the document
  """
  text: str
  _: dataclasses.KW_ONLY
  tags: Set[str] = frozenset()

  # TODO: b/311191278 - implement _repr_pretty_, _repr_html_, _repr_markdown_

  def __post_init__(self):
    object.__setattr__(self, 'tags', frozenset(self.tags))

  def __str__(self):
    return self.text


class Document:
  """A document of text and associated metadata."""

  def __init__(self, contents: Iterable[Content] = ()) -> None:
    """Initializes the document.

    Args:
      contents: Initial contents of the document.
    """
    # TODO: b/311191572 - be more efficient if contents is a tupel iter.
    self._contents = tuple(contents)

  # TODO: b/311191905 - implement __iadd__, __add__?
  # TODO: b/311191278 - implement _repr_pretty_, _repr_html_, _repr_markdown_

  def __iter__(self) -> Iterator[Content]:
    """Yields the contents in the document."""
    yield from self._contents

  def __eq__(self, other):
    """Returns True if other is a Document with identical contents."""
    if not isinstance(other, type(self)):
      return NotImplemented
    else:
      return self._contents == other._contents

  def __ne__(self, other):
    """Returns True if other is not a Document or has different contents."""
    return not self.__eq__(other)

  def contents(self) -> tuple[Content, ...]:
    """Returns the contents in the document."""
    return self._contents

  def text(self) -> str:
    """Returns all the text in the document."""
    return ''.join(content.text for content in self)

  def view(
      self,
      include_tags: Iterable[str] = (),
      exclude_tags: Iterable[str] = (),
  ) -> 'View':
    """Returns a view of the document.

    Args:
      include_tags: specifies which tags to include in the view.
      exclude_tags: specifies which tags to exclude from the view.
    """
    return View(self, include_tags=include_tags, exclude_tags=exclude_tags)

  def clear(self):
    """Clears the document."""
    self._contents = ()

  def append(
      self,
      text: str,
      *,
      tags: Collection[str] = (),
  ) -> None:
    """Appends text to the document."""
    text = Content(text=text, tags=frozenset(tags))
    self._contents += (text,)

  def extend(self, contents: Iterable[Content]) -> None:
    """Extends the document with the provided contents."""
    self._contents += tuple(contents)

  def copy(self) -> 'Document':
    """Returns a copy of the document."""
    return Document(self.contents())

  def new(self: T) -> T:
    """Returns an empty copy of this document."""
    document = self.copy()
    document.clear()
    return document

  @contextlib.contextmanager
  def edit(self: T) -> Iterator[T]:
    """Edits the current document.

    Creates a edit based on the current document. Once the context is completed,
    the edit will be committed to the document. If you wish not to commit the
    edit call edit.clear() before leavign the context.

    Yields:
      The document being edited.
    """
    edit = self.new()
    yield edit
    self.extend(edit.contents())


class View:
  """A view of a document."""

  def __init__(
      self,
      document: Document,
      include_tags: Iterable[str] = (),
      exclude_tags: Iterable[str] = (),
  ) -> None:
    """Initializes the instance.

    Args:
      document: the base document on which to add edits.
      include_tags: specifies which tags to include in the view.
      exclude_tags: specifies which tags to exclude from the view.
    """
    self._include_tags = frozenset(include_tags)
    self._exclude_tags = frozenset(exclude_tags)
    common_tags = self._include_tags & self._exclude_tags
    if common_tags:
      raise ValueError(f'Cannot both include and exclude tags {common_tags!r}')
    self._document = document

  def __iter__(self) -> Iterator[Content]:
    """Yields the contents in the view."""
    for content in self._document:
      if self._exclude_tags and content.tags & self._exclude_tags:
        continue
      elif self._include_tags and not content.tags & self._include_tags:
        continue
      else:
        yield content

  def contents(self) -> tuple[Content, ...]:
    """Yields the contents in the view."""
    return tuple(self)

  def text(self) -> str:
    """Returns the contents of the document as a single string."""
    return ''.join(content.text for content in self)

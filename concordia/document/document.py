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

  def __post_init__(self):
    object.__setattr__(self, 'tags', frozenset(self.tags))

  def __str__(self):
    return self.text

  def _repr_pretty_(self, p, cycle):
    """Pretty print representation for IPython and rich terminals."""
    if cycle:
      p.text('Content(...)')
    else:
      with p.group(2, 'Content(', ')'):
        p.text(f'text={self.text!r}')
        if self.tags:
          p.text(', ')
          p.breakable()
          p.text(f'tags={set(self.tags)!r}')

  def _repr_html_(self):
    """HTML representation for Jupyter notebooks."""
    import html
    text_html = html.escape(self.text).replace('\n', '<br>')
    if self.tags:
      tags_html = ', '.join(html.escape(tag) for tag in sorted(self.tags))
      return (
          f'<div style="border-left: 3px solid #3498db; padding-left: 10px; '
          f'margin: 5px 0;">'
          f'<div style="font-family: monospace;">{text_html}</div>'
          f'<div style="color: #7f8c8d; font-size: 0.9em; margin-top: 5px;">'
          f'Tags: {tags_html}</div>'
          f'</div>'
      )
    else:
      return (
          f'<div style="border-left: 3px solid #3498db; padding-left: 10px; '
          f'margin: 5px 0;">'
          f'<div style="font-family: monospace;">{text_html}</div>'
          f'</div>'
      )

  def _repr_markdown_(self):
    """Markdown representation for Jupyter notebooks and compatible viewers."""
    if self.tags:
      tags_str = ', '.join(f'`{tag}`' for tag in sorted(self.tags))
      return f'{self.text}\n\n*Tags: {tags_str}*'
    else:
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

  def __iter__(self) -> Iterator[Content]:
    """Yields the contents in the document."""
    yield from self._contents

  def _repr_pretty_(self, p, cycle):
    """Pretty print representation for IPython and rich terminals."""
    if cycle:
      p.text('Document(...)')
    else:
      num_contents = len(self._contents)
      text_preview = self.text()[:100]
      if len(self.text()) > 100:
        text_preview += '...'
      with p.group(2, 'Document(', ')'):
        p.text(f'{num_contents} content(s), ')
        p.text(f'{len(self.text())} chars')
        if text_preview:
          p.breakable()
          p.text(f'Preview: {text_preview!r}')

  def _repr_html_(self):
    """HTML representation for Jupyter notebooks."""
    import html
    if not self._contents:
      return '<div style="color: #95a5a6;"><em>Empty document</em></div>'
    
    parts = [
        '<div style="border: 1px solid #bdc3c7; border-radius: 5px; '
        'padding: 10px; margin: 10px 0; background-color: #ecf0f1;">',
        f'<div style="font-weight: bold; margin-bottom: 10px; '
        f'color: #2c3e50;">'
        f'Document ({len(self._contents)} content(s), '
        f'{len(self.text())} chars)</div>',
        '<div style="background-color: white; padding: 10px; '
        'border-radius: 3px; max-height: 400px; overflow-y: auto;">',
    ]
    
    for i, content in enumerate(self._contents[:20]):  # Limit to first 20
      content_html = html.escape(content.text).replace('\n', '<br>')
      if content.tags:
        tags_html = ', '.join(html.escape(tag) for tag in sorted(content.tags))
        parts.append(
            f'<div style="margin-bottom: 10px; padding: 5px; '
            f'border-left: 3px solid #3498db;">'
            f'{content_html}'
            f'<div style="color: #7f8c8d; font-size: 0.85em; '
            f'margin-top: 3px;">'
            f'Tags: {tags_html}</div>'
            f'</div>'
        )
      else:
        parts.append(
            f'<div style="margin-bottom: 10px; padding: 5px; '
            f'border-left: 3px solid #95a5a6;">'
            f'{content_html}'
            f'</div>'
        )
    
    if len(self._contents) > 20:
      remaining = len(self._contents) - 20
      parts.append(
          f'<div style="color: #7f8c8d; font-style: italic; '
          f'margin-top: 10px;">'
          f'... and {remaining} more content(s)</div>'
      )
    
    parts.append('</div></div>')
    return ''.join(parts)

  def _repr_markdown_(self):
    """Markdown representation for Jupyter notebooks and compatible viewers."""
    if not self._contents:
      return '*Empty document*'
    
    parts = [
        f'# Document\n\n',
        f'**{len(self._contents)} content(s), {len(self.text())} characters**\n\n',
        '---\n\n',
    ]
    
    for i, content in enumerate(self._contents[:20], 1):  # Limit to first 20
      parts.append(f'### Content {i}\n\n')
      parts.append(content.text)
      if content.tags:
        tags_str = ', '.join(f'`{tag}`' for tag in sorted(content.tags))
        parts.append(f'\n\n*Tags: {tags_str}*')
      parts.append('\n\n---\n\n')
    
    if len(self._contents) > 20:
      remaining = len(self._contents) - 20
      parts.append(f'\n\n*... and {remaining} more content(s)*\n')
    
    return ''.join(parts)

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

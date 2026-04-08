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

"""Tests for document repr methods."""

import unittest
from unittest import mock

from concordia.document import document


class ContentReprTest(unittest.TestCase):
  """Tests for Content representation methods."""

  def test_repr_pretty_simple(self):
    """Tests _repr_pretty_ with simple content."""
    content = document.Content(text='Hello, world!')
    
    # Mock the pretty printer
    p = mock.Mock()
    p.group = mock.MagicMock()
    
    content._repr_pretty_(p, cycle=False)
    
    # Verify that text was called with content info
    p.text.assert_called()
    call_args = [call[0][0] for call in p.text.call_args_list]
    self.assertTrue(any('Hello, world!' in str(arg) for arg in call_args))

  def test_repr_pretty_with_tags(self):
    """Tests _repr_pretty_ with tags."""
    content = document.Content(
        text='Tagged content', 
        tags={'tag1', 'tag2'}
    )
    
    p = mock.Mock()
    p.group = mock.MagicMock()
    p.breakable = mock.Mock()
    
    content._repr_pretty_(p, cycle=False)
    
    # Verify tags were included
    call_args = [call[0][0] for call in p.text.call_args_list]
    self.assertTrue(any('tags=' in str(arg) for arg in call_args))

  def test_repr_pretty_cycle(self):
    """Tests _repr_pretty_ with cycle detection."""
    content = document.Content(text='Test')
    
    p = mock.Mock()
    content._repr_pretty_(p, cycle=True)
    
    p.text.assert_called_once_with('Content(...)')

  def test_repr_html_simple(self):
    """Tests _repr_html_ with simple content."""
    content = document.Content(text='Test content')
    
    html = content._repr_html_()
    
    self.assertIn('Test content', html)
    self.assertIn('<div', html)
    self.assertIn('</div>', html)

  def test_repr_html_with_newlines(self):
    """Tests _repr_html_ handles newlines correctly."""
    content = document.Content(text='Line 1\nLine 2\nLine 3')
    
    html = content._repr_html_()
    
    self.assertIn('<br>', html)
    self.assertIn('Line 1', html)
    self.assertIn('Line 2', html)

  def test_repr_html_with_tags(self):
    """Tests _repr_html_ includes tags."""
    content = document.Content(
        text='Content with tags',
        tags={'important', 'review'}
    )
    
    html = content._repr_html_()
    
    self.assertIn('Content with tags', html)
    self.assertIn('Tags:', html)
    # Both tags should appear (order may vary)
    self.assertIn('important', html)
    self.assertIn('review', html)

  def test_repr_html_escapes_special_chars(self):
    """Tests _repr_html_ escapes HTML special characters."""
    content = document.Content(text='<script>alert("test")</script>')
    
    html = content._repr_html_()
    
    # Should be escaped
    self.assertNotIn('<script>', html)
    self.assertIn('&lt;script&gt;', html)

  def test_repr_markdown_simple(self):
    """Tests _repr_markdown_ with simple content."""
    content = document.Content(text='# Heading\n\nSome text.')
    
    markdown = content._repr_markdown_()
    
    self.assertIn('# Heading', markdown)
    self.assertIn('Some text.', markdown)

  def test_repr_markdown_with_tags(self):
    """Tests _repr_markdown_ includes tags."""
    content = document.Content(
        text='Important note',
        tags={'urgent', 'action'}
    )
    
    markdown = content._repr_markdown_()
    
    self.assertIn('Important note', markdown)
    self.assertIn('Tags:', markdown)
    self.assertIn('`urgent`', markdown)
    self.assertIn('`action`', markdown)


class DocumentReprTest(unittest.TestCase):
  """Tests for Document representation methods."""

  def test_repr_pretty_empty(self):
    """Tests _repr_pretty_ with empty document."""
    doc = document.Document()
    
    p = mock.Mock()
    p.group = mock.MagicMock()
    p.breakable = mock.Mock()
    
    doc._repr_pretty_(p, cycle=False)
    
    # Should indicate 0 contents
    call_args = [call[0][0] for call in p.text.call_args_list]
    self.assertTrue(any('0 content' in str(arg) for arg in call_args))

  def test_repr_pretty_with_content(self):
    """Tests _repr_pretty_ with document contents."""
    doc = document.Document()
    doc.append('First line')
    doc.append('Second line')
    
    p = mock.Mock()
    p.group = mock.MagicMock()
    p.breakable = mock.Mock()
    
    doc._repr_pretty_(p, cycle=False)
    
    # Should show number of contents and char count
    call_args = [call[0][0] for call in p.text.call_args_list]
    text_output = ' '.join(str(arg) for arg in call_args)
    self.assertIn('2 content', text_output)

  def test_repr_pretty_cycle(self):
    """Tests _repr_pretty_ with cycle detection."""
    doc = document.Document()
    
    p = mock.Mock()
    doc._repr_pretty_(p, cycle=True)
    
    p.text.assert_called_once_with('Document(...)')

  def test_repr_html_empty(self):
    """Tests _repr_html_ with empty document."""
    doc = document.Document()
    
    html = doc._repr_html_()
    
    self.assertIn('Empty document', html)

  def test_repr_html_with_content(self):
    """Tests _repr_html_ with document contents."""
    doc = document.Document()
    doc.append('Content 1')
    doc.append('Content 2', tags={'important'})
    doc.append('Content 3')
    
    html = doc._repr_html_()
    
    self.assertIn('Document', html)
    self.assertIn('3 content(s)', html)
    self.assertIn('Content 1', html)
    self.assertIn('Content 2', html)
    self.assertIn('Content 3', html)
    self.assertIn('important', html)

  def test_repr_html_limits_content(self):
    """Tests _repr_html_ limits display to first 20 contents."""
    doc = document.Document()
    for i in range(25):
      doc.append(f'Content {i}')
    
    html = doc._repr_html_()
    
    # Should show truncation message
    self.assertIn('... and 5 more content', html)
    # Should have first few items
    self.assertIn('Content 0', html)
    self.assertIn('Content 19', html)
    # Should not show the last items
    self.assertNotIn('Content 24', html)

  def test_repr_html_handles_newlines(self):
    """Tests _repr_html_ converts newlines to <br> tags."""
    doc = document.Document()
    doc.append('Line 1\nLine 2\nLine 3')
    
    html = doc._repr_html_()
    
    self.assertIn('<br>', html)

  def test_repr_markdown_empty(self):
    """Tests _repr_markdown_ with empty document."""
    doc = document.Document()
    
    markdown = doc._repr_markdown_()
    
    self.assertIn('Empty document', markdown)

  def test_repr_markdown_with_content(self):
    """Tests _repr_markdown_ with document contents."""
    doc = document.Document()
    doc.append('First paragraph')
    doc.append('Second paragraph', tags={'note'})
    
    markdown = doc._repr_markdown_()
    
    self.assertIn('# Document', markdown)
    self.assertIn('First paragraph', markdown)
    self.assertIn('Second paragraph', markdown)
    self.assertIn('### Content 1', markdown)
    self.assertIn('### Content 2', markdown)
    self.assertIn('`note`', markdown)

  def test_repr_markdown_limits_content(self):
    """Tests _repr_markdown_ limits display to first 20 contents."""
    doc = document.Document()
    for i in range(30):
      doc.append(f'Item {i}')
    
    markdown = doc._repr_markdown_()
    
    # Should indicate truncation
    self.assertIn('... and 10 more content', markdown)
    # Should have early items
    self.assertIn('Item 0', markdown)
    # Should not have late items
    self.assertNotIn('Item 29', markdown)


if __name__ == '__main__':
  unittest.main()

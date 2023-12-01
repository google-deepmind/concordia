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


from absl.testing import absltest
from absl.testing import parameterized
from concordia.document import document


class DocumentTest(parameterized.TestCase):

  def test_init(self):
    doc = document.Document()
    with self.subTest('text'):
      self.assertEmpty(doc.text())
    with self.subTest('contents'):
      self.assertEmpty(doc.contents())

  def test_append(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])

    with self.subTest('text'):
      self.assertEqual(doc.text(), 'onetwo')

    with self.subTest('contents'):
      expected = [
          document.Content(text='one', tags=frozenset({'a', 'b'})),
          document.Content(text='two', tags=frozenset({'b', 'c'})),
      ]
      self.assertSequenceEqual(doc.contents(), expected)

    with self.subTest('document'):
      self.assertNotEmpty(doc.text())

  def test_clear(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.clear()

    with self.subTest('text'):
      self.assertEmpty(doc.text())
    with self.subTest('contents'):
      self.assertEmpty(doc.contents())
    with self.subTest('document'):
      self.assertEmpty(doc.text())

  def test_view(self):
    doc = document.Document()
    view = doc.view()
    initial_text = view.text()
    initial_contents = view.contents()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.append('three', tags=['c', 'd'])
    final_text = view.text()
    final_contents = view.contents()

    with self.subTest('initial_text'):
      self.assertEqual(initial_text, '')
    with self.subTest('initial_contents'):
      self.assertEmpty(initial_contents)
    with self.subTest('final_text'):
      self.assertEqual(final_text, 'onetwothree')
    with self.subTest('final_contents'):
      self.assertSequenceEqual(
          final_contents,
          [
              document.Content(text='one', tags=frozenset({'a', 'b'})),
              document.Content(text='two', tags=frozenset({'b', 'c'})),
              document.Content(text='three', tags=frozenset({'c', 'd'})),
          ],
      )

  def test_filtered_view(self):
    doc = document.Document()
    view = doc.view(include_tags={'b'}, exclude_tags={'a'})
    initial_text = view.text()
    initial_contents = view.contents()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.append('three', tags=['c', 'd'])
    final_text = view.text()
    final_contents = view.contents()

    with self.subTest('initial_text'):
      self.assertEmpty(initial_text)
    with self.subTest('initial_contents'):
      self.assertEmpty(initial_contents)
    with self.subTest('final_text'):
      self.assertEqual(final_text, 'two')
    with self.subTest('final_contents'):
      self.assertSequenceEqual(
          final_contents,
          [
              document.Content(text='two', tags=frozenset({'b', 'c'})),
          ],
      )

  def test_edit(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.append('three', tags=['c', 'd'])
    doc_before_edit = doc.contents()
    with doc.edit() as edit:
      edit.append('four', tags=['d', 'e'])
      doc_during_edit = doc.contents()
    edit_contents = edit.contents()
    doc_after_edit = doc.contents()

    with self.subTest('doc_during_edit'):
      self.assertEqual(doc_during_edit, doc_before_edit)
    with self.subTest('doc_after_edit'):
      self.assertEqual(doc_after_edit, doc_before_edit + edit_contents)
    with self.subTest('edit_contents'):
      self.assertSequenceEqual(
          edit_contents,
          [
              document.Content(text='four', tags=frozenset({'d', 'e'})),
          ],
      )

  def test_edit_rollback(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.append('three', tags=['c', 'd'])
    doc_before_edit = doc.contents()
    with doc.edit() as edit:
      edit.append('four', tags=['d', 'e'])
      edit.clear()
    doc_after_edit = doc.contents()

    with self.subTest('doc_after_no_edit'):
      self.assertEqual(doc_after_edit, doc_before_edit)
    with self.subTest('empty_edit'):
      self.assertEmpty(edit.contents())

  def test_eq(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.append('three', tags=['c', 'd'])
    copy = doc.copy()
    self.assertEqual(doc, copy)

  def test_ne(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    copy = doc.copy()
    doc.append('three', tags=['c', 'd'])
    self.assertNotEqual(doc, copy)

  def test_new(self):
    doc = document.Document()
    doc.append('one', tags=['a', 'b'])
    doc.append('two', tags=['b', 'c'])
    doc.append('three', tags=['c', 'd'])
    new_doc = doc.new()
    self.assertEqual(new_doc, document.Document())


if __name__ == '__main__':
  absltest.main()

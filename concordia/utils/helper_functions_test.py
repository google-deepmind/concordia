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

import collections
import copy
import types
import unittest

from absl.testing import absltest
from concordia.prefabs.entity import basic
from concordia.utils import helper_functions

EXPECTED_OUTPUT_STANDARD_CASE = """
---
**`basic__Entity`**:
```python
Entity(
    description='An entity.',
    params={'name': 'Logan', 'goal': ''}
)
```
---
""".strip()


class TestPrettyPrintFunction(unittest.TestCase):

  def test_empty_dictionary(self):
    """Tests that an empty dictionary returns the correct placeholder string."""
    self.assertEqual(
        first=helper_functions.print_pretty_prefabs({}),
        second='(The dictionary is empty)',
    )

  def test_standard_case_with_filtering(self):
    """Tests a standard case with two objects, ensuring 'entities=None' and 'entities=()' are correctly filtered out."""
    test_dict = {
        'basic__Entity': basic.Entity(
            description='An entity.',
            params={'name': 'Logan', 'goal': ''},
            entities=None,
        )
    }
    self.assertEqual(
        helper_functions.print_pretty_prefabs(test_dict),
        EXPECTED_OUTPUT_STANDARD_CASE,
    )


class FindNestedDataTest(absltest.TestCase):

  def test_preserves_nested_duplicates_when_disabled(self):
    value = {'name': 'Alice'}
    data = {'outer': [{'event': value}, {'event': value}]}
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(
            data, 'event', remove_duplicates=False
        ),
        [value, value],
    )

  def test_preserves_nested_scalar_values_when_disabled(self):
    data = {'outer': [{'event': 'Alice'}, {'event': 'Alice'}]}
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(
            data, 'event', remove_duplicates=False
        ),
        ['Alice', 'Alice'],
    )

  def test_default_removes_duplicates_across_nested_branches(self):
    value = {'name': 'Alice'}
    data = {'left': {'event': value}, 'right': [{'event': value}]}
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'), [value]
    )

  def test_default_handles_top_level_scalar_value(self):
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(
            {'event': 'Alice'}, 'event'
        ),
        ['Alice'],
    )

  def test_default_removes_duplicate_scalar_values(self):
    data = {'outer': [{'event': 'Alice'}, {'event': 'Alice'}]}
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'),
        ['Alice'],
    )

  def test_default_removes_duplicates_across_mixed_value_types(self):
    data = {
        'a': {'event': {'name': 'Alice'}},
        'b': {'event': 'Alice'},
        'c': [{'event': {'name': 'Alice'}}, {'event': ['x', 'y']}],
        'd': {'event': ['x', 'y']},
    }
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'),
        [{'name': 'Alice'}, 'Alice', ['x', 'y']],
    )

  def test_default_removes_duplicates_of_dicts_with_unhashable_values(self):
    value = {'name': 'Alice', 'tags': ['x']}
    data = {'left': {'event': value}, 'right': {'event': dict(value)}}
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'), [value]
    )


class DeepCompareComponentsTest(absltest.TestCase):

  def test_compares_nested_components_without_skip_keys(self):
    first = types.SimpleNamespace(child=types.SimpleNamespace(value=1))
    second = types.SimpleNamespace(child=types.SimpleNamespace(value=1))
    helper_functions.deep_compare_components(first, second, self)

  def test_reports_value_mismatch_without_skip_keys(self):
    with self.assertRaises(AssertionError):
      helper_functions.deep_compare_components(
          types.SimpleNamespace(value=1), types.SimpleNamespace(value=2), self
      )

  def test_still_skips_requested_keys(self):
    helper_functions.deep_compare_components(
        types.SimpleNamespace(value=1),
        types.SimpleNamespace(value=2),
        self,
        skip_keys={'value'},
    )


class FindNestedSequenceDataTest(absltest.TestCase):

  def test_top_level_tuple_is_searched(self):
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(('a', {'a': 1}), 'a'),
        [1],
    )

  def test_mixed_sequences_keep_depth_first_order_and_input(self):
    data = (
        {'event': 1, 'nested': [{'event': 2}]},
        [{'branch': ({'event': 3},)}],
    )
    before = copy.deepcopy(data)
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'), [1, 2, 3]
    )
    self.assertEqual(data, before)

  def test_duplicates_across_list_and_tuple_branches(self):
    value = {'name': 'Alice', 'tags': ['x']}
    data = ([{'event': value}], {'outer': ({'event': copy.deepcopy(value)},)})
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'), [value]
    )
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(
            data, 'event', remove_duplicates=False
        ),
        [value, value],
    )

  def test_other_sequences_follow_the_same_traversal(self):
    pair = collections.namedtuple('Pair', ['first', 'second'])
    for data in (
        collections.UserList([{'event': 1}, {'event': 2}]),
        pair({'event': 1}, {'event': 2}),
    ):
      with self.subTest(sequence_type=type(data)):
        self.assertEqual(
            helper_functions.find_data_in_nested_structure(data, 'event'),
            [1, 2],
        )

  def test_text_binary_and_empty_sequences_remain_leaves(self):
    data = ('event', b'event', bytearray(b'event'), (), [], {'event': 'kept'})
    self.assertEqual(
        helper_functions.find_data_in_nested_structure(data, 'event'), ['kept']
    )
    self.assertEmpty(
        helper_functions.find_data_in_nested_structure((), 'event')
    )

  def test_matching_tuple_value_is_preserved_and_searched(self):
    value = ({'event': 'nested'},)
    actual = helper_functions.find_data_in_nested_structure(
        {'event': value}, 'event'
    )
    self.assertEqual(actual, [value, 'nested'])
    self.assertIs(actual[0], value)


if __name__ == '__main__':
  absltest.main()

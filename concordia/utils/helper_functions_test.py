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


class TestDeepCompareComponents(unittest.TestCase):

  def test_no_skip_keys_does_not_crash(self):
    """Regression: calling with skip_keys=None (old default) raised TypeError."""

    class Foo:
      pass

    a, b = Foo(), Foo()
    a.x = 1
    b.x = 1
    # Should not raise TypeError: argument of type 'NoneType' is not iterable
    helper_functions.deep_compare_components(a, b, self)

  def test_skip_keys_excludes_field(self):

    class Bar:
      pass

    a, b = Bar(), Bar()
    a.x = 1
    b.x = 99  # would fail if compared
    helper_functions.deep_compare_components(a, b, self, skip_keys=('x',))


if __name__ == '__main__':
  absltest.main()

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

import datetime
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


class TestTimedeltaToReadableStr(unittest.TestCase):

  def test_hours_and_minutes(self):
    """Regression: hours+minutes used to produce 'X hours a n d Y minutes'."""
    td = datetime.timedelta(hours=2, minutes=30)
    result = helper_functions.timedelta_to_readable_str(td)
    self.assertEqual(result, '2 hours and 30 minutes')

  def test_hours_minutes_and_seconds(self):
    td = datetime.timedelta(hours=1, minutes=5, seconds=10)
    result = helper_functions.timedelta_to_readable_str(td)
    self.assertEqual(result, '1 hour and 5 minutes and 10 seconds')

  def test_minutes_and_seconds_only(self):
    td = datetime.timedelta(minutes=3, seconds=45)
    result = helper_functions.timedelta_to_readable_str(td)
    self.assertEqual(result, '3 minutes and 45 seconds')

  def test_seconds_only(self):
    td = datetime.timedelta(seconds=42)
    result = helper_functions.timedelta_to_readable_str(td)
    self.assertEqual(result, '42 seconds')

  def test_singular_units(self):
    td = datetime.timedelta(hours=1, minutes=1, seconds=1)
    result = helper_functions.timedelta_to_readable_str(td)
    self.assertEqual(result, '1 hour and 1 minute and 1 second')


if __name__ == '__main__':
  absltest.main()

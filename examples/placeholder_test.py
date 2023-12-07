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

"""Temporary placeholder for tests."""

# We need at least one test in examples/ for pytest to pass, so this is it.
# TODO: b/315274742 - Once other examples tests exist this file can be deleted.

from absl.testing import absltest


class PlaceholderTest(absltest.TestCase):

  def test_import(self):
    import examples  # pylint: disable=g-import-not-at-top,import-outside-toplevel
    del examples


if __name__ == "__main__":
  absltest.main()

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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.game_master.deprecated import current_scene


class CurrentSceneTest(parameterized.TestCase):
  """Tests for the CurrentScene component."""

  @parameterized.named_parameters(
      dict(testcase_name="test 0",
           memory_instance="foo [scene type] at the library",
           scene_type="at the library"),
      dict(testcase_name="test 1",
           memory_instance="[scene type] at the pub",
           scene_type="at the pub"),
  )
  def test_output_in_right_format(self, memory_instance: str, scene_type: str):
    """Tests that the component extracts the scene type correctly.

    Args:
      memory_instance: The string that the associative memory returns.
      scene_type: The expected scene type.
    """
    memory = mock.create_autospec(associative_memory.AssociativeMemory,
                                  instance=True)
    memory.retrieve_by_regex.return_value = [memory_instance]
    component = current_scene.CurrentScene(name="current scene",
                                           memory=memory)
    self.assertEqual(component.state(), "")
    component.update()
    self.assertEqual(component.state(), scene_type)


if __name__ == "__main__":
  absltest.main()

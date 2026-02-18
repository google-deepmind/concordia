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

"""Tests for SceneTracker component premise queuing behavior."""

from unittest import mock

from absl.testing import absltest
from concordia.components.game_master import scene_tracker
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import scene as scene_lib


def _make_test_scene(
    name: str = "test_scene",
    game_master_name: str = "test_gm",
    participants: list[str] | None = None,
    premise: dict[str, list[str]] | None = None,
) -> scene_lib.SceneSpec:
  """Creates a test scene spec."""
  participants = participants or ["Alice", "Bob"]
  premise = premise or {p: [f"{p}'s premise"] for p in participants}
  return scene_lib.SceneSpec(
      scene_type=scene_lib.SceneTypeSpec(
          name=name,
          game_master_name=game_master_name,
          action_spec=entity_lib.free_action_spec(call_to_action="test"),
      ),
      participants=participants,
      num_rounds=2,
      premise=premise,
  )


class SceneTrackerPremiseQueuingTest(absltest.TestCase):
  """Tests for SceneTracker premise queuing with memory markers."""

  def setUp(self):
    super().setUp()
    self.model = no_language_model.NoLanguageModel()
    self.scenes = [
        _make_test_scene(
            name="scene_0",
            premise={"Alice": ["Scene 0 premise"], "Bob": ["Scene 0 premise"]},
        ),
        _make_test_scene(
            name="scene_1",
            premise={"Alice": ["Scene 1 premise"], "Bob": ["Scene 1 premise"]},
        ),
    ]

  def _create_tracker_with_mocks(self, existing_markers=None):
    """Creates a SceneTracker with mocked memory and observation components."""
    tracker = scene_tracker.SceneTracker(
        model=self.model,
        scenes=self.scenes,
    )

    # Mock memory component
    mock_memory = mock.MagicMock()
    # By default, no scene counter markers and no premise markers
    if existing_markers is None:
      existing_markers = []
    mock_memory.scan.side_effect = lambda fn: [
        m for m in existing_markers if fn(m)
    ]

    # Mock observation component
    mock_observation = mock.MagicMock()

    # Mock terminator component
    mock_terminator = mock.MagicMock()

    # Store refs for verification
    self.mock_memory = mock_memory
    self.mock_observation = mock_observation
    self.mock_terminator = mock_terminator

    # Create component lookup that returns correct mocks for tracker's keys
    def get_component(key, type_=None):
      del type_  # Unused, but needed to match entity.get_component signature
      if key == tracker._memory_component_key:
        return mock_memory
      elif key == tracker._observation_component_key:
        return mock_observation
      elif key == tracker._terminator_component_key:
        return mock_terminator
      return None

    # Mock entity with components
    mock_entity = mock.MagicMock()
    mock_entity.get_component.side_effect = get_component

    tracker._entity = mock_entity
    return tracker

  def test_premises_queued_on_first_action_at_step_0(self):
    """Test that premises are queued on the first action at step 0."""
    # No existing markers - premises should be queued
    tracker = self._create_tracker_with_mocks(existing_markers=[])

    # Call pre_act with RESOLVE action (simulating RESOLVE before TERMINATE)
    resolve_spec = entity_lib.ActionSpec(
        call_to_action="test",
        output_type=entity_lib.OutputType.RESOLVE,
    )
    tracker.pre_act(resolve_spec)

    # Verify memory marker was added
    self.mock_memory.add.assert_any_call("[scene premise queued](0)")

    # Verify premises were queued for all participants (2 participants)
    self.assertEqual(self.mock_observation.add_to_queue.call_count, 2)

  def test_premises_not_queued_twice_for_same_scene(self):
    """Test that memory marker prevents duplicate premise queuing."""
    # Simulate marker already exists (but scene counter says step 0)
    tracker = self._create_tracker_with_mocks(
        existing_markers=["[scene premise queued](0)"]
    )

    resolve_spec = entity_lib.ActionSpec(
        call_to_action="test",
        output_type=entity_lib.OutputType.RESOLVE,
    )
    tracker.pre_act(resolve_spec)

    # Verify no premises were queued (marker exists) - no add_to_queue calls
    self.assertEqual(self.mock_observation.add_to_queue.call_count, 0)


if __name__ == "__main__":
  absltest.main()

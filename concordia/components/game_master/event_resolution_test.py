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

"""Tests for event_resolution component."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution
from concordia.components.game_master import next_acting
from concordia.typing import entity as entity_lib


PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG


class EventResolutionFilteringTest(parameterized.TestCase):
  """Tests that event resolution correctly filters putative events by player."""

  @parameterized.named_parameters(
      dict(
          testcase_name='alice_after_bob_acted',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Alice: Action A',
              f'{PUTATIVE_EVENT_TAG} Bob: Action B',
              f'{PUTATIVE_EVENT_TAG} Alice: Action A2',
          ],
          active_player='Alice',
          expected_action='Action A2',
      ),
      dict(
          testcase_name='bob_after_alice_acted',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Alice: Action A',
              f'{PUTATIVE_EVENT_TAG} Bob: Action B',
          ],
          active_player='Bob',
          expected_action='Action B',
      ),
      dict(
          testcase_name='single_player_action',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Charlie: Action C',
          ],
          active_player='Charlie',
          expected_action='Action C',
      ),
      dict(
          testcase_name='out_of_order_memory',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Bob: Action B',
              f'{PUTATIVE_EVENT_TAG} Alice: Action A',
              f'{PUTATIVE_EVENT_TAG} Charlie: Action C',
          ],
          active_player='Alice',
          expected_action='Action A',
      ),
  )
  def test_filters_by_active_player(
      self, memory_contents, active_player, expected_action
  ):
    """Test that the correct action is retrieved for each active player."""
    mock_model = mock.MagicMock()
    mock_memory = mock.MagicMock(spec=memory_component.Memory)
    mock_memory.scan.return_value = memory_contents

    mock_next_acting = mock.MagicMock(spec=next_acting.NextActing)
    mock_next_acting.get_currently_active_player.return_value = active_player

    mock_entity = mock.MagicMock()
    mock_entity.get_component.side_effect = lambda key, type_: {
        memory_component.DEFAULT_MEMORY_COMPONENT_KEY: mock_memory,
        next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: mock_next_acting,
    }.get(key)

    component = event_resolution.EventResolution(
        model=mock_model,
        event_resolution_steps=[],
        components=[],
    )
    component.set_entity(mock_entity)

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    component.pre_act(action_spec)

    self.assertEqual(component.get_putative_action(), expected_action)
    self.assertEqual(component.get_active_entity_name(), active_player)


if __name__ == '__main__':
  absltest.main()

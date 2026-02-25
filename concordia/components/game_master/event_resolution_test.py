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
      # Tests for conversation action format (with dashes)
      dict(
          testcase_name='conversation_action_with_dashes',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Alice -- "Hello there!"',
          ],
          active_player='Alice',
          expected_action='"Hello there!"',
      ),
      dict(
          testcase_name='conversation_action_mixed_with_decision',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Bob: 3 coins',
              f'{PUTATIVE_EVENT_TAG} Alice -- "That seems fair."',
          ],
          active_player='Alice',
          expected_action='"That seems fair."',
      ),
      dict(
          testcase_name='decision_action_after_conversation',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Alice -- "I propose 3 coins."',
              f'{PUTATIVE_EVENT_TAG} Bob: accept',
          ],
          active_player='Bob',
          expected_action='accept',
      ),
      dict(
          testcase_name='multiple_conversation_actions',
          memory_contents=[
              f'{PUTATIVE_EVENT_TAG} Alice -- "First message"',
              f'{PUTATIVE_EVENT_TAG} Bob -- "Reply to Alice"',
              f'{PUTATIVE_EVENT_TAG} Alice -- "Second message"',
          ],
          active_player='Alice',
          expected_action='"Second message"',
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


class SendEventToRelevantPlayersTest(absltest.TestCase):
  """Tests for SendEventToRelevantPlayers component."""

  def test_player_filter_limits_recipients(self):
    """Test that player_filter callback correctly limits event recipients."""
    mock_model = mock.MagicMock()
    all_players = ['Alice', 'Bob', 'Charlie']
    filtered_players = ['Alice', 'Charlie']

    mock_make_obs = mock.MagicMock()

    mock_entity = mock.MagicMock()
    mock_entity.get_component.return_value = mock_make_obs

    component = event_resolution.SendEventToRelevantPlayers(
        model=mock_model,
        player_names=all_players,
        make_observation_component_key='make_observation',
        player_filter=lambda: filtered_players,
    )
    component.set_entity(mock_entity)

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    component.pre_act(action_spec)
    component.post_act('Event: Something happened')

    called_players = [
        call.args[0] for call in mock_make_obs.add_to_queue.call_args_list
    ]
    self.assertIn('Alice', called_players)
    self.assertIn('Charlie', called_players)
    self.assertNotIn('Bob', called_players)

  def test_no_player_filter_sends_to_all(self):
    """Test that without player_filter, all players receive events."""
    mock_model = mock.MagicMock()
    all_players = ['Alice', 'Bob', 'Charlie']

    mock_make_obs = mock.MagicMock()

    mock_entity = mock.MagicMock()
    mock_entity.get_component.return_value = mock_make_obs

    component = event_resolution.SendEventToRelevantPlayers(
        model=mock_model,
        player_names=all_players,
        make_observation_component_key='make_observation',
    )
    component.set_entity(mock_entity)

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    component.pre_act(action_spec)
    component.post_act('Event: Something happened')

    called_players = [
        call.args[0] for call in mock_make_obs.add_to_queue.call_args_list
    ]
    self.assertIn('Alice', called_players)
    self.assertIn('Bob', called_players)
    self.assertIn('Charlie', called_players)


EVENT_TAG = event_resolution.EVENT_TAG


class DisplayEventsFilterTest(absltest.TestCase):
  """Tests for DisplayEvents event_filter_fn parameter."""

  def _create_component(self, memory_contents, event_filter_fn=None):
    mock_model = mock.MagicMock()
    mock_memory = mock.MagicMock(spec=memory_component.Memory)
    mock_memory.scan.return_value = memory_contents

    mock_entity = mock.MagicMock()
    mock_entity.get_component.return_value = mock_memory

    component = event_resolution.DisplayEvents(
        model=mock_model,
        event_filter_fn=event_filter_fn,
    )
    component.set_entity(mock_entity)
    component._logging_channel = mock.MagicMock()
    return component

  def test_filter_excludes_events(self):
    events = [
        f'{EVENT_TAG} Alice arrived at town_square.',
        f'{EVENT_TAG} Bob is at the market.',
        f'{EVENT_TAG} Charlie is at town_square.',
    ]
    component = self._create_component(
        events,
        event_filter_fn=lambda e: 'town_square' in e,
    )
    result = component._make_pre_act_value()
    self.assertIn('Alice arrived at town_square', result)
    self.assertIn('Charlie is at town_square', result)
    self.assertNotIn('Bob', result)

  def test_no_filter_passes_all_events(self):
    events = [
        f'{EVENT_TAG} Alice arrived at town_square.',
        f'{EVENT_TAG} Bob is at the market.',
    ]
    component = self._create_component(events, event_filter_fn=None)
    result = component._make_pre_act_value()
    self.assertIn('Alice', result)
    self.assertIn('Bob', result)

  def test_filter_applied_before_recency_limit(self):
    events = [
        f'{EVENT_TAG} Old event at location_a.',
        f'{EVENT_TAG} Event at location_b.',
        f'{EVENT_TAG} Recent event at location_a.',
    ]
    component = self._create_component(
        events,
        event_filter_fn=lambda e: 'location_a' in e,
    )
    component._num_events_to_retrieve = 1
    result = component._make_pre_act_value()
    self.assertIn('Recent event at location_a', result)
    self.assertNotIn('Old event', result)
    self.assertNotIn('location_b', result)


if __name__ == '__main__':
  absltest.main()

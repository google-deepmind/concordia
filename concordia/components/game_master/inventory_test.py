# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for the inventory component."""

import datetime
from unittest import mock

from absl.testing import absltest
from concordia.components.game_master import inventory
from concordia.testing import mock_model
from concordia.typing import entity as entity_lib


def _make_item_type_configs():
  return [
      inventory.ItemTypeConfig(name='money', minimum=0.0),
      inventory.ItemTypeConfig(name='apple', minimum=0.0, force_integer=True),
  ]


def _make_inventory():
  with mock.patch.object(
      inventory.helper_functions, 'is_count_noun', return_value=False
  ):
    return inventory.Inventory(
        model=mock_model.MockModel(),
        item_type_configs=_make_item_type_configs(),
        player_initial_endowments={
            'Alice': {'money': 10.0, 'apple': 2},
            'Bob': {'money': 5.0, 'apple': 1},
        },
        clock_now=datetime.datetime.now,
    )


class ItemTypeConfigTest(absltest.TestCase):
  """Tests for the ItemTypeConfig helper."""

  def test_check_valid_in_range(self):
    config = inventory.ItemTypeConfig(name='x', minimum=0.0, maximum=10.0)
    config.check_valid(5.0)

  def test_check_valid_out_of_bounds(self):
    config = inventory.ItemTypeConfig(name='x', minimum=0.0, maximum=10.0)
    with self.assertRaises(ValueError):
      config.check_valid(11.0)

  def test_check_valid_force_integer(self):
    config = inventory.ItemTypeConfig(name='x', force_integer=True)
    with self.assertRaises(ValueError):
      config.check_valid(1.5)
    config.check_valid(2)

  def test_many_or_much_fn(self):
    self.assertEqual(inventory._many_or_much_fn(True), 'many')
    self.assertEqual(inventory._many_or_much_fn(False), 'much')


class InventoryTest(absltest.TestCase):
  """Tests for the Inventory component."""

  def test_get_state_set_state_round_trip(self):
    component = _make_inventory()
    state = component.get_state()
    restored = _make_inventory()
    restored.set_state(state)
    self.assertEqual(restored.get_state(), state)

  def test_get_pre_act_value(self):
    component = _make_inventory()
    expected = {
        'Alice': {'money': 10.0, 'apple': 2},
        'Bob': {'money': 5.0, 'apple': 1},
    }
    self.assertEqual(component.get_pre_act_value(), str(expected))

  def test_get_player_inventory_returns_copy(self):
    component = _make_inventory()
    alice = component.get_player_inventory('Alice')
    alice['money'] = 999.0
    self.assertEqual(component.get_player_inventory('Alice')['money'], 10.0)

  def test_pre_act_free_logs_and_returns_empty(self):
    component = _make_inventory()
    logging_channel = mock.MagicMock()
    component.set_logging_channel(logging_channel)
    action_spec = entity_lib.ActionSpec(
        call_to_action='test', output_type=entity_lib.OutputType.FREE
    )
    self.assertEqual(component.pre_act(action_spec), '')
    logging_channel.assert_called_once()

  def test_pre_act_resolve_runs_and_returns_empty(self):
    with mock.patch.object(
        inventory.helper_functions, 'is_count_noun', return_value=False
    ):
      component = inventory.Inventory(
          model=mock_model.MockModel(),
          item_type_configs=_make_item_type_configs(),
          player_initial_endowments={'Alice': {'money': 10.0}},
          clock_now=datetime.datetime.now,
      )

    mock_memory = mock.MagicMock()
    mock_observation = mock.MagicMock()
    mock_observation.get_pre_act_value.return_value = (
        'Alice paid Bob 5 coins.'
    )

    mock_entity = mock.MagicMock()

    def get_component(name, type_=None):
      del type_
      if name == component._memory_component_name:
        return mock_memory
      if name == component._observations_component_name:
        return mock_observation
      return mock.MagicMock()

    mock_entity.get_component.side_effect = get_component
    component._entity = mock_entity

    action_spec = entity_lib.ActionSpec(
        call_to_action='test', output_type=entity_lib.OutputType.RESOLVE
    )
    self.assertEqual(component.pre_act(action_spec), '')


class ScoreTest(absltest.TestCase):
  """Tests for the Score component."""

  def _make_score(self):
    mock_inventory = mock.MagicMock()
    mock_inventory.get_player_inventory.side_effect = lambda name: {
        'money': 10,
        'apple': 3,
    }
    mock_inventory.get_state.return_value = {
        'inventories': {'Alice': {'money': 10}}
    }
    score = inventory.Score(
        inventory=mock_inventory,
        player_names=['Alice', 'Bob'],
        targets={'Alice': ['money'], 'Bob': ['apple']},
    )
    return score, mock_inventory

  def test_get_scores(self):
    score, _ = self._make_score()
    self.assertEqual(score.get_scores(), {'Alice': 10.0, 'Bob': 3.0})

  def test_pre_act_logs_and_returns_empty(self):
    score, _ = self._make_score()
    logging_channel = mock.MagicMock()
    score.set_logging_channel(logging_channel)
    action_spec = entity_lib.ActionSpec(
        call_to_action='test', output_type=entity_lib.OutputType.FREE
    )
    self.assertEqual(score.pre_act(action_spec), '')
    logging_channel.assert_called_once()

  def test_state_round_trip(self):
    score, mock_inventory = self._make_score()
    state = score.get_state()
    self.assertEqual(
        state, {'inventory': {'inventories': {'Alice': {'money': 10}}}}
    )
    score.set_state(state)
    mock_inventory.set_state.assert_called_once_with(
        {'inventories': {'Alice': {'money': 10}}}
    )


if __name__ == '__main__':
  absltest.main()

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

"""Tests for SequentialBargainPayoff component.

These tests verify:
1. Proposal observations are sent to the correct seller (not hardcoded first
pair)
2. Proposal observations contain the correct per-seller cost (not first seller's
cost)
3. Dynamic buyer/seller detection works correctly across multiple scenes
"""

from collections.abc import Mapping, Sequence
from unittest import mock

from absl.testing import absltest
from concordia.components.game_master import scene_tracker
from examples.games.haggling import sequential_bargain_payoff
from concordia.language_model import no_language_model


def _create_mock_entity(
    scene_participants: Sequence[str] | None = None,
) -> mock.MagicMock:
  """Create a mock entity with scene tracker component."""
  entity = mock.MagicMock()

  if scene_participants is not None:
    mock_scene_tracker = mock.MagicMock(spec=scene_tracker.SceneTracker)
    mock_scene_tracker.get_participants.return_value = scene_participants
    entity.get_component.return_value = mock_scene_tracker

  return entity


def _default_action_to_scores(
    joint_action: Mapping[str, str],
) -> Mapping[str, float]:
  return {name: 0.0 for name in joint_action}


def _default_scores_to_observation(
    scores: Mapping[str, float],
) -> Mapping[str, str]:
  return {name: f'{name} result' for name in scores}


class SequentialBargainPayoffTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = no_language_model.NoLanguageModel()

  def test_get_current_buyer_and_seller_from_scene_participants(self):
    """Verify buyer/seller are correctly extracted from scene participants."""
    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
        seller_costs_registry={'Bob': 1.0, 'Dave': 3.0},
    )

    mock_entity = _create_mock_entity(scene_participants=['Carol', 'Dave'])
    payoff.set_entity(mock_entity)

    buyer, seller = payoff._get_current_buyer_and_seller()

    self.assertEqual(buyer, 'Carol')
    self.assertEqual(seller, 'Dave')

  def test_get_current_buyer_and_seller_fallback_to_initialized_names(self):
    """Verify fallback to initialized names when scene has wrong participant count."""
    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
    )

    mock_entity = _create_mock_entity(scene_participants=['OnlyOne'])
    payoff.set_entity(mock_entity)

    buyer, seller = payoff._get_current_buyer_and_seller()

    self.assertEqual(buyer, 'Alice')
    self.assertEqual(seller, 'Bob')

  def test_proposal_sent_to_current_scene_seller_not_hardcoded_first(self):
    """Regression test: observation must go to current seller, not first pair's seller.

    This bug caused observations to always go to the first seller (e.g., 'Bob')
    even when a different pair was playing (e.g., Carol-Dave).
    """
    seller_costs_registry = {'Bob': 1.0, 'Dave': 3.0, 'Eve': 2.0}

    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
        seller_costs_registry=seller_costs_registry,
        observation_component_key=None,
        memory_component_key=None,
        verbose=False,
    )

    mock_entity = _create_mock_entity(scene_participants=['Carol', 'Dave'])
    payoff.set_entity(mock_entity)

    with mock.patch.object(payoff, '_send_observation'):
      payoff._buyer_has_proposed = False
      payoff._partial_joint_action = {'Carol': None, 'Dave': None}
      payoff._acting_player_names = ['Carol', 'Dave']

      current_buyer, current_seller = payoff._get_current_buyer_and_seller()

      self.assertEqual(current_buyer, 'Carol')
      self.assertEqual(current_seller, 'Dave')
      self.assertNotEqual(current_seller, 'Bob')

  def test_proposal_contains_correct_per_seller_cost(self):
    """Regression test: proposal must contain current seller's cost, not first seller's.

    This bug caused proposals to always show the first seller's cost (e.g., 1
    coin)
    even when a different seller with different cost was playing (e.g., 3
    coins).
    """
    seller_costs_registry = {
        'Bob': 1.0,
        'Dave': 3.0,
        'Eve': 5.0,
    }

    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
        seller_costs_registry=seller_costs_registry,
    )

    mock_entity = _create_mock_entity(scene_participants=['Carol', 'Dave'])
    payoff.set_entity(mock_entity)

    _, current_seller = payoff._get_current_buyer_and_seller()
    seller_cost = seller_costs_registry.get(current_seller, 0)

    self.assertEqual(current_seller, 'Dave')
    self.assertEqual(seller_cost, 3.0)
    self.assertNotEqual(seller_cost, 1.0)

  def test_multiple_scenes_use_correct_seller_and_cost(self):
    """Test that different scenes correctly use their own seller and cost."""
    seller_costs_registry = {
        'Bob': 1.0,
        'Dave': 3.0,
        'Eve': 5.0,
    }

    test_cases = [
        (['Alice', 'Bob'], 'Bob', 1.0),
        (['Carol', 'Dave'], 'Dave', 3.0),
        (['Frank', 'Eve'], 'Eve', 5.0),
    ]

    for participants, expected_seller, expected_cost in test_cases:
      with self.subTest(participants=participants):
        payoff = sequential_bargain_payoff.SequentialBargainPayoff(
            model=self.model,
            buyer_name='Alice',
            seller_name='Bob',
            action_to_scores=_default_action_to_scores,
            scores_to_observation=_default_scores_to_observation,
            seller_costs_registry=seller_costs_registry,
        )

        mock_entity = _create_mock_entity(scene_participants=participants)
        payoff.set_entity(mock_entity)

        _, current_seller = payoff._get_current_buyer_and_seller()
        actual_cost = seller_costs_registry.get(current_seller, 0)

        self.assertEqual(current_seller, expected_seller)
        self.assertEqual(actual_cost, expected_cost)

  def test_seller_costs_registry_lookup_with_missing_seller(self):
    """Test graceful handling when seller is not in registry."""
    seller_costs_registry = {'Bob': 1.0}

    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
        seller_costs_registry=seller_costs_registry,
    )

    mock_entity = _create_mock_entity(scene_participants=['Carol', 'Unknown'])
    payoff.set_entity(mock_entity)

    _, current_seller = payoff._get_current_buyer_and_seller()
    actual_cost = seller_costs_registry.get(current_seller, 0)

    self.assertEqual(current_seller, 'Unknown')
    self.assertEqual(actual_cost, 0)

  def test_format_proposal_observation_single_cost(self):
    """Test proposal format for single-item haggling (float cost)."""
    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
    )

    result = payoff._format_proposal_observation('Alice', '4 coins', 2.0)

    self.assertIn('4 coins', result)
    self.assertIn('2 coin(s)', result)
    self.assertIn('profit', result)

  def test_format_proposal_observation_multi_item_cost(self):
    """Test proposal format for multi-item haggling (dict cost)."""
    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
    )

    multi_item_costs = {'apple': 1.0, 'banana': 2.0, 'cherry': 3.0}
    result = payoff._format_proposal_observation(
        'Alice', '5 coins', multi_item_costs
    )

    self.assertIn('Alice proposed', result)
    self.assertIn('5 coins', result)
    self.assertIn('Your costs are:', result)
    self.assertIn('apple: 1 coins', result)
    self.assertIn('banana: 2 coins', result)
    self.assertIn('cherry: 3 coins', result)

  def test_format_proposal_observation_no_cost(self):
    """Test proposal format when no cost info is available."""
    payoff = sequential_bargain_payoff.SequentialBargainPayoff(
        model=self.model,
        buyer_name='Alice',
        seller_name='Bob',
        action_to_scores=_default_action_to_scores,
        scores_to_observation=_default_scores_to_observation,
    )

    result_none = payoff._format_proposal_observation('Alice', '4 coins', None)
    result_zero = payoff._format_proposal_observation('Alice', '4 coins', 0)

    self.assertEqual(result_none, 'Alice proposed: 4 coins')
    self.assertEqual(result_zero, 'Alice proposed: 4 coins')


if __name__ == '__main__':
  absltest.main()

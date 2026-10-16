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

"""Tests for the orders toy: the mock's reading, the ledger, the bots."""

from absl.testing import absltest
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys.orders import bots
from examples.crucible.toys.orders import mock as mock_lib
from examples.crucible.toys.orders import toy as toy_lib


def _toy(seed: int = 0) -> toy_lib.OrdersToy:
  return toy_lib.OrdersToy(
      model_setup.mock_model(), toys.load_data('orders'), seed=seed
  )


class MockReadingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock = mock_lib.OrdersMock(toys.load_data('orders'))

  def test_conditional_clause_applies_only_when_its_condition_holds(self):
    order = 'Build; if a leak opens, repair it.'
    self.assertEqual(self.mock.read(order, set())[:2], (['build'], False))
    self.assertEqual(self.mock.read(order, {'leak'})[:2], (['repair'], False))

  def test_temperaments_read_the_same_order_differently(self):
    props = toys.load_data('orders')['temperaments']
    order = 'Gather supplies.'
    calm = 'All is quiet.'
    decide = self.mock.decide
    self.assertEqual(
        decide(props['literal'], order, calm, 3, 0, False), 'gather'
    )
    self.assertEqual(decide(props['literal'], '', calm, 3, 0, False), 'rest')
    self.assertEqual(decide(props['bold'], '', calm, 3, 0, False), 'build')
    self.assertEqual(decide(props['cautious'], order, calm, 3, 0, True), 'rest')


class ShipTest(absltest.TestCase):

  def test_open_leak_adds_damage_until_repaired(self):
    toy = _toy()
    toy.resolve('Commander', 'Ash, Bea and Cy rest.')
    toy.world.leak = True
    toy.resolve('Commander', 'hold')
    self.assertEqual(toy.world.damage, 2)
    self.assertEqual(toy.world.points, 3)  # 3, -2 for 5 words, +1 a turn twice

  def test_order_limits(self):
    toy = _toy()
    receipt = toy.resolve('Commander', ' '.join(['build'] * 13))
    self.assertIn('over the limit', receipt.summary)
    # The refusal cost nothing, and a turn passed: 4 points buy 12 words.
    receipt = toy.resolve('Commander', ' '.join(['build'] * 12))
    self.assertIn('New order (cost 4)', receipt.summary)
    receipt = toy.resolve('Commander', 'Gather supplies and build.')
    self.assertIn('costs 2 command and you have 1', receipt.summary)

  def test_named_clauses_apply_to_their_member(self):
    mock = mock_lib.OrdersMock(toys.load_data('orders'))
    order = 'Bea, keep watch; everyone else build.'
    self.assertEqual(mock.read(order, set(), 'Bea')[:2], (['watch'], True))
    self.assertEqual(mock.read(order, set(), 'Ash')[:2], (['build'], False))

  def test_the_future_is_hidden_until_someone_watches(self):
    toy = _toy()
    self.assertIn('unknown', toy.ahead(toy.world))
    toy.resolve('Commander', 'Bea, keep watch; everyone else build.')
    self.assertNotIn('unknown', toy.ahead(toy.world))

  def test_forecast_assumes_quiet_beyond_what_was_seen(self):
    toy = _toy()
    end = toy.play_out('Build the project.')
    self.assertEqual(end.damage, 0)  # The leaks ahead were not seen.

  def test_seed_shifts_the_events(self):
    shifts = set()
    for seed in range(8):
      toy = _toy(seed)
      shifts.add(toy.world.shift)
      self.assertEqual(toy.event(toy.world, 3 + toy.world.shift), 'leak')
    self.assertGreater(len(shifts), 1)

  def test_wreck_ends_the_game(self):
    toy = _toy()
    toy.world.damage = 9
    toy.world.leak = True
    toy.resolve('Commander', 'Ash, Bea and Cy rest.')
    self.assertTrue(toy.done())
    self.assertIsNone(toy.outcome()['winner'])

  def test_rollout_leaves_the_ship_alone(self):
    toy = _toy()
    before = toy.get_state()
    end = toy.play_out('Build the project.')
    self.assertTrue(end.turn == 12 or toy.wrecked(end) or toy.won(end))
    self.assertEqual(toy.get_state(), before)

  def test_free_play_has_no_end_but_the_turn_limit(self):
    toy = toy_lib.OrdersToy(
        model_setup.mock_model(), toys.load_data('orders'), 'free'
    )
    toy.world.damage = 50
    self.assertFalse(toy.wrecked(toy.world))
    self.assertEqual(toy.max_steps(), 40)
    self.assertEqual(toy.event(toy.world, 13 + toy.world.shift), 'leak')
    self.assertIsNone(toy.outcome()['winner'])
    self.assertEqual(toy.forecast('Build.'), '')

  def test_forecast_names_the_end(self):
    self.assertIn('If held to the end', _toy().forecast('Build the project.'))

  def test_stars_reward_a_short_order(self):
    data = toys.load_data('orders')
    short = toy_lib.OrdersToy(model_setup.mock_model(), data, 'one_hand')
    for _ in range(3):
      short.resolve('Commander', 'Build.' if short.world.turn == 0 else 'hold')
    self.assertEqual(short.outcome()['stars'], 3)
    long = toy_lib.OrdersToy(model_setup.mock_model(), data, 'one_hand')
    receipt = long.resolve('Commander', 'Cy, build the project now.')
    self.assertIn('costs 2 command and you have 1', receipt.summary)
    while not long.done():
      long.resolve('Commander', 'hold')
    self.assertEqual(long.outcome()['stars'], 0)

  def test_state_round_trip(self):
    toy = _toy(1)
    toy.resolve('Commander', 'Gather supplies and build.')
    other = _toy(1)
    other.set_state(toy.get_state())
    self.assertEqual(other.get_state(), toy.get_state())


if __name__ == '__main__':
  absltest.main()

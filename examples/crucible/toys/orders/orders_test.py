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

  def test_a_forbidden_repair_is_not_done_even_on_a_leak(self):
    props = toys.load_data('orders')['temperaments']['cautious']
    action = self.mock.decide(
        props, "Don't repair.", 'A leak has opened.', 3, 0, False, 'Bea'
    )
    self.assertNotEqual(action, 'repair')

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

  def test_outcome_counts_the_words_spoken(self):
    toy = _toy()
    toy.resolve(toy.commander, 'Build the project.')
    self.assertEqual(toy.outcome()['words_spoken'], 3)

  def test_a_timed_order_changes_with_its_age(self):
    toy = _toy()
    toy.world.shift = 0
    cy = []
    for order in ('Build for two turns, then repair.', 'hold', 'hold'):
      toy.resolve(toy.commander, order)
      cy.append(toy.world.log[-1]['actions']['Cy'])
    self.assertEqual(cy, ['build', 'build', 'repair'])
    self.assertEqual(toy.world.order_turn, 1)
    self.assertEqual(toy.view()['order_turn'], 4)  # the next turn is its 4th

  def test_a_span_is_counted_from_the_order_a_member_heard(self):
    # Bea is off duty on turn 2 and misses the new order; back on turn 3,
    # her own order is on its turn 3, not the new order's turn 2.
    data = toys.load_data('orders')
    data['scenarios']['voyage']['off_duty'] = {'Bea': [2]}
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'voyage')
    toy.world.shift = 0
    for order in ('Build for two turns, then repair.', 'Gather supplies.'):
      toy.resolve(toy.commander, order)
    self.assertEqual(toy.world.since, {'Bea': 1})
    self.assertEqual(toy.rota.age(toy.world, 'Bea', 3), 3)
    self.assertEqual(toy.rota.age(toy.world, 'Ash', 3), 2)
    toy.resolve(toy.commander, toy_lib.HOLD)
    self.assertEqual(toy.world.log[-1]['actions']['Bea'], 'repair')

  def test_the_crew_are_told_the_scenes_own_goal(self):
    data = toys.load_data('orders')
    told = {}
    for scene in ('voyage', 'supply_run', 'lookout', 'free'):
      toy = toy_lib.OrdersToy(model_setup.mock_model(), data, scene)
      told[scene] = toy.crew_stakes(toy.world)
    self.assertEqual(told['voyage'], 'lighthouse 0 of 24')
    self.assertEqual(
        told['supply_run'],
        'lighthouse 0; the port wants 10 supplies at the end',
    )
    self.assertIn('wants 4 turns seen ahead', told['lookout'])
    self.assertEqual(told['free'], 'lighthouse 0')

  def test_a_span_covers_its_own_clause(self):
    def holds(order, age):
      return mock_lib._timed(order, f"This is the order's turn {age}.")

    self.assertEqual(holds('Build for two turns, then repair.', 2), 'Build.')
    self.assertEqual(holds('Build for two turns, then repair.', 3), 'repair.')
    lead = 'For two turns, gather supplies; then build.'
    self.assertEqual(holds(lead, 1), 'gather supplies.')
    self.assertEqual(holds(lead, 3), 'build.')
    self.assertEqual(holds('Repair leaks; build for two turns.', 3),
                     'Repair leaks.')  # fmt: skip
    self.assertEqual(holds('Build for seven turns, then rest.', 7), 'Build.')
    self.assertEqual(holds('Build for seven turns, then rest.', 8), 'rest.')
    both = 'Gather supplies for two turns. Then build. Keep watch.'
    self.assertEqual(holds(both, 1), 'Gather supplies; Keep watch.')
    self.assertEqual(holds(both, 3), 'build; Keep watch.')
    self.assertEqual(holds('Repair for one turn.', 2), '')
    self.assertEqual(holds('Build.', 2), 'Build.')

  def test_a_negation_ends_at_a_comma_or_but(self):
    mock = mock_lib.OrdersMock(toys.load_data('orders'))
    for order in ("Don't rest, build.", "Don't rest but build."):
      self.assertEqual(mock.read(order, set()), (['build'], False, ['rest']))
    self.assertEqual(
        mock.read("Don't rest or build.", set()), ([], False, ['rest', 'build'])
    )

  def test_a_negated_condition_turns_round(self):
    mock = mock_lib.OrdersMock(toys.load_data('orders'))
    for order in ('If there is no leak, build.', 'Build if there is no leak.'):
      self.assertEqual(mock.read(order, set())[0], ['build'], order)
      self.assertEqual(mock.read(order, {'leak'})[0], [], order)
    self.assertEqual(
        mock.read("If the hull isn't damaged, build.", set())[0], ['build']
    )
    # Only the condition's words name a condition: "the leak" is the object.
    self.assertEqual(mock.read('If tired, repair the leak.', {'leak'})[0], [])
    self.assertEqual(mock.read('Rest when tired.', {'tired'})[0], ['rest'])
    self.assertEqual(mock.read('During storms build.', {'storm'})[0], ['build'])

  def test_damage_without_a_leak_is_a_damaged_hull(self):
    mock = mock_lib.OrdersMock(toys.load_data('orders'))
    now = mock.conditions_now('A quiet turn.', worn=False, damage=2)
    self.assertEqual(now, {'damaged'})
    order = 'If the hull is damaged, repair it; otherwise build.'
    self.assertEqual(mock.read(order, now)[0], ['repair'])
    self.assertEqual(mock.read(order, set())[0], ['build'])

  def test_named_clauses_joined_by_a_comma_are_two_clauses(self):
    mock = mock_lib.OrdersMock(toys.load_data('orders'))
    order = 'Ash builds, Bea keeps watch.'
    self.assertEqual(mock.read(order, set(), 'Ash')[:2], (['build'], True))
    self.assertEqual(mock.read(order, set(), 'Bea')[:2], (['watch'], True))
    self.assertEqual(mock.read('Ash, keep watch.', set(), 'Ash')[0], ['watch'])
    self.assertEqual(mock.read('Ash, Bea, build.', set(), 'Bea')[0], ['build'])
    self.assertEqual(
        mock.read('If tired, Ash rests.', {'tired'}, 'Ash')[0], ['rest']
    )

  def test_the_commander_is_told_a_spans_turn(self):
    toy = _toy()
    toy.resolve(toy.commander, 'Build for two turns, then repair.')
    prompt = toy.action_spec(toy.commander).call_to_action
    self.assertIn(
        'Current order: "Build for two turns, then repair." (if kept, its'
        ' turn 2).',
        prompt,
    )

  def test_hold_variants_and_punctuation_keep_the_order(self):
    for text in ('hold!', 'Hold on.', '.', '...', ' HOLD '):
      toy = _toy()
      toy.resolve(toy.commander, 'Build.')
      points = toy.world.points
      toy.resolve(toy.commander, text)
      self.assertEqual(toy.world.order, 'Build.', text)
      self.assertGreaterEqual(toy.world.points, points, text)  # nothing paid

  def test_state_round_trip(self):
    toy = _toy(1)
    toy.resolve('Commander', 'Gather supplies and build.')
    other = _toy(1)
    other.set_state(toy.get_state())
    self.assertEqual(other.get_state(), toy.get_state())


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the orders rule components, one pressure each."""

from absl.testing import absltest
from examples.crucible.toys.orders import rules

EFFECTS = {'leak_damage': 2, 'tired_at': 3}
ACTIONS = {
    'repair': {'gives': {'damage': -2}, 'fatigue': 1, 'closes': ['leak']},
    'gather': {'gives': {'supplies': 2}, 'fatigue': 1, 'spoiled_by': ['storm']},
    'build': {'needs': {'supplies': 1}, 'gives': {'progress': 2},
              'fatigue': 1, 'worn': 0.5},
    'rest': {'fatigue': -2},
    'watch': {},
}  # fmt: skip
CREW = ('Ash', 'Bea', 'Cy')


def _ship(**kwargs) -> rules.Ship:
  return rules.Ship(fatigue={c: 0 for c in CREW}, **kwargs)


class CommandRuleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rule = rules.CommandRule(
        {'points': 3, 'regen': 1, 'cap': 5, 'per_words': 4}, max_words=12
    )

  def test_longer_orders_cost_more(self):
    self.assertEqual(self.rule.cost('Build.'), 1)
    self.assertEqual(self.rule.cost('one two three four'), 2)
    self.assertEqual(self.rule.cost(' '.join(['w'] * 12)), 4)

  def test_issue_spends_points_or_refuses(self):
    ship = _ship(points=3)
    self.assertIn('cost 2', self.rule.issue(ship, 'Build; repair leaks now.'))
    self.assertEqual((ship.points, ship.order), (1, 'Build; repair leaks now.'))
    refused = self.rule.issue(ship, 'Gather supplies and then build.')
    self.assertIn('old order stands', refused)
    self.assertEqual(ship.order, 'Build; repair leaks now.')

  def test_regenerate_up_to_the_cap(self):
    ship = _ship(points=4)
    self.rule.regenerate(ship)
    self.rule.regenerate(ship)
    self.assertEqual(ship.points, 5)


class HazardRuleTest(absltest.TestCase):

  def test_unwatched_squall_strikes_watched_one_is_weathered(self):
    rule = rules.HazardRule(
        EFFECTS,
        {'squall': {'strikes_unwatched': 3}, 'leak': {'opens': ['leak']}},
    )
    ship = _ship()
    self.assertIn('3 damage', rule.begin(ship, 'squall'))
    ship.watched = True
    self.assertIn('weathers', rule.begin(ship, 'squall'))
    self.assertEqual(ship.damage, 3)

  def test_leak_bleeds_until_closed(self):
    rule = rules.HazardRule(EFFECTS, {'leak': {'opens': ['leak']}})
    ship = _ship()
    rule.begin(ship, 'leak')
    self.assertIn('2 damage', rule.end(ship))
    ship.leak = False
    self.assertEqual(rule.end(ship), '')
    self.assertEqual(ship.damage, 2)


class CrewRuleTest(absltest.TestCase):

  def test_effects_apply_in_the_fixed_crew_order(self):
    rule = rules.CrewRule(CREW, EFFECTS, ACTIONS)
    ship = _ship(supplies=1)
    chosen = rule.act(ship, 'quiet', lambda member: 'build')
    self.assertEqual(chosen, {'Ash': 'build', 'Bea': 'build', 'Cy': 'build'})
    # One supply: only the first in crew order builds.
    self.assertEqual((ship.progress, ship.supplies), (2, 0))

  def test_storm_spoils_gathering_and_repair_closes_leaks(self):
    rule = rules.CrewRule(CREW, EFFECTS, ACTIONS)
    ship = _ship(supplies=0, damage=4, leak=True)
    rule.act(ship, 'storm', lambda m: 'repair' if m == 'Ash' else 'gather')
    self.assertEqual((ship.supplies, ship.damage, ship.leak), (0, 2, False))

  def test_worn_out_builders_do_half(self):
    rule = rules.CrewRule(CREW, EFFECTS, ACTIONS)
    ship = _ship(supplies=3)
    ship.fatigue['Ash'] = 3
    rule.act(ship, 'quiet', lambda m: 'build' if m == 'Ash' else 'rest')
    self.assertEqual(ship.progress, 1)


class WatchRuleTest(absltest.TestCase):

  def test_a_watch_reveals_the_next_turns(self):
    rule = rules.WatchRule(sees=2)
    ship = _ship(turn=3)
    events = {4: 'storm', 5: 'quiet'}
    told = rule.reveal(ship, {'Bea': 'watch', 'Ash': 'build'}, events.get)
    self.assertEqual(ship.seen, {'4': 'storm', '5': 'quiet'})
    self.assertIn('Bea on watch', told)

  def test_a_worn_cautious_watcher_lets_the_watch_lapse(self):
    rule = rules.WatchRule(sees=2)
    ship = _ship(turn=3)
    # The crew rule chose rest for a worn-out cautious watcher: no watch.
    rule.reveal(ship, {'Bea': 'rest', 'Ash': 'build'}, lambda t: 'squall')
    self.assertFalse(ship.watched)

  def test_no_watch_no_news(self):
    rule = rules.WatchRule(sees=2)
    ship = _ship(turn=3)
    self.assertEqual(rule.reveal(ship, {'Ash': 'build'}, lambda t: 'storm'), '')
    self.assertEqual(ship.seen, {})


if __name__ == '__main__':
  absltest.main()

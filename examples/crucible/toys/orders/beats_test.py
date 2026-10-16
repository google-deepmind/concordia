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

"""Each scene of the first arc does its beat's job (GUIDE.md, Phase 4)."""

from absl.testing import absltest
from examples.crucible import audits
from examples.crucible import toys

POLICIES = ('skilled', 'random', 'passive', 'habit')


def _rates(scenario: str) -> dict[str, dict[str, float]]:
  return audits.win_rates('orders', scenario, seeds=4, policies=POLICIES)


class BeatsTest(absltest.TestCase):

  def test_arc_has_all_four_beats(self):
    scenarios = toys.load_data('orders')['scenarios']
    beats = [s.get('beat') for s in scenarios.values()]
    for beat in ('isolate', 'develop', 'twist', 'catharsis'):
      self.assertIn(beat, beats)

  def test_isolate_the_rule_is_the_only_way_through(self):
    rates = _rates('one_hand')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)  # Silence never works.

  def test_develop_the_gap_widens(self):
    rates = _rates('two_hands')
    self.assertGreaterEqual(
        rates['skilled']['win_pct'] - rates['random']['win_pct'], 30
    )

  def test_twist_the_habit_loses(self):
    rates = _rates('supply_run')
    self.assertEqual(rates['habit']['win_pct'], 0)
    self.assertEqual(rates['skilled']['win_pct'], 100)

  def test_arc_two_isolate_and_develop(self):
    rates = _rates('lookout')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)
    rates = _rates('squalls')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)

  def test_arc_two_twist_breaks_the_watch_habit(self):
    rates = audits.win_rates('orders', 'night_watch', 8, POLICIES)
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertLessEqual(rates['habit']['win_pct'], 25)

  def test_arc_two_catharsis(self):
    rates = _rates('crossing')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['random']['win_pct'], 0)

  def test_arc_three_contrary(self):
    rates = _rates('contrary')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['habit']['win_pct'], 0)  # "Build." makes Dex rest.
    rates = _rates('mixed_hands')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)

  def test_arc_three_twist_and_catharsis(self):
    rates = _rates('the_mender')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['habit']['win_pct'], 0)  # Dex builds; no one mends.
    rates = _rates('all_hands')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)

  def test_arc_four_isolate_and_develop(self):
    rates = _rates('rota')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)
    self.assertEqual(rates['habit']['win_pct'], 0)  # Bea's watch lapses.
    rates = audits.win_rates('orders', 'watches', 8, POLICIES)
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertGreaterEqual(
        rates['skilled']['win_pct'] - rates['random']['win_pct'], 30
    )

  def test_arc_four_twist_and_catharsis(self):
    rates = _rates('no_volunteers')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['habit']['win_pct'], 0)  # No one takes "someone".
    rates = _rates('long_night')
    self.assertEqual(rates['skilled']['win_pct'], 100)
    self.assertEqual(rates['passive']['win_pct'], 0)

  def test_catharsis_rewards_skill(self):
    rates = _rates('voyage')
    self.assertGreaterEqual(
        rates['skilled']['win_pct'] - rates['random']['win_pct'], 30
    )
    self.assertGreaterEqual(
        rates['skilled']['win_pct'] - rates['passive']['win_pct'], 50
    )


if __name__ == '__main__':
  absltest.main()

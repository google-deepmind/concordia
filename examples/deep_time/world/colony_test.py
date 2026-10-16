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

"""Tests for the colony on a world light-years from home."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.world import colony as colony_lib
from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import publishing


def _colony() -> colony_lib.Colony:
  return colony_lib.Colony(data.load('colony.yaml'))


class ColonyTest(absltest.TestCase):

  def test_shortfalls_kill_and_plenty_grows(self):
    colony = _colony()
    colony.step()
    self.assertAlmostEqual(colony.population, 20.0 * (1 + colony.growth))
    colony.ledger.withdraw(colony_lib.HOLDER, 'air', colony.stock('air'))
    colony.allocation = {'ice_mine': 1.0}
    before = colony.population
    events = colony.step()
    self.assertIn('runs short of air', events[0])
    self.assertEqual(colony.shortfall, {'air': 1.0})
    self.assertAlmostEqual(colony.population, before * (1 - colony.deaths))
    self.assertIn('ran short of air (100%)', colony.sheet())

  def test_terraforming_warms_and_the_sky_cools_back(self):
    colony = _colony()
    colony.ledger.deposit(colony_lib.HOLDER, 'warmth', 40.0, 'test')
    colony.allocation = {'ice_mine': 1.0}
    colony.step()
    warmed = colony.start_temp_c + 40 * colony.warmth_c
    self.assertAlmostEqual(
        colony.temp_c, warmed - colony.cooling * (warmed - colony.start_temp_c)
    )
    colony.step()
    self.assertLess(colony.temp_c, warmed)

  def test_designs_arrive_once_and_unlock_recipes(self):
    colony = _colony()
    self.assertNotIn('fabricator', [r.key for r in colony.known()])
    text = colony.receive_design('fabrication')
    assert text is not None
    self.assertIn('fabricating solar panels', text)
    self.assertIn('fabricator', [r.key for r in colony.known()])
    self.assertIsNone(colony.receive_design('fabrication'))
    self.assertIsNone(colony.receive_design('warp drive'))

  def test_plans_balance_terraforming_with_power(self):
    colony = _colony()
    shares = colony.plan('terraformer', 0.4)
    self.assertAlmostEqual(sum(shares.values()), 0.4)
    self.assertGreater(shares['solar_farm'], 0.0)

  def test_plans_assume_a_calm_sky(self):
    colony = _colony()
    calm = colony.plan('terraformer', 0.4)
    colony.storm = True
    self.assertEqual(colony.plan('terraformer', 0.4), calm)

  def test_the_sheet_says_how_long_a_falling_store_lasts(self):
    colony = _colony()
    colony.allocation = {'terraformer': 1.0}  # No food made at all.
    colony.step()
    sheet = colony.sheet()
    self.assertIn('Last generation: air 0.0 made for', sheet)
    self.assertRegex(
        sheet,
        r'food 0\.0 made for \d+\.\d needed \(the store lasts \d+ more'
        r' generations\)',
    )
    other = _colony()
    other.set_state(colony.get_state())
    self.assertEqual(other.sheet(), sheet)

  def test_state_round_trip(self):
    colony = _colony()
    colony.step()
    colony.receive_design('fabrication')
    other = _colony()
    other.set_state(colony.get_state())
    self.assertEqual(other.payload(1), colony.payload(1))


class LightDelayTest(absltest.TestCase):

  def test_colonists_act_and_the_goal_is_judged(self):
    context = game_lib.new_game('light_delay')
    planet = context.planet
    colony = planet.colony
    assert colony is not None
    actor = actors_lib.ColonyActor('Colonists', planet, 0.15, 0.4)
    self.assertIn(
        'put more labor into: run the terraforming plants', actor.options()
    )
    self.assertStartsWith(
        actor.resolve('put more labor into: mine ice'), 'Colonists put more'
    )
    context.publish_state()
    status = context.bus.latest('colony.status')
    assert status is not None
    self.assertIn(
        'New Ararat (generation 0): 20.0 colonists',
        publishing.render_colony(status),
    )
    colony.temp_c = 5.0  # Warm enough to stay above 0 C after cooling.
    planet.advance()
    self.assertEqual(planet.outcome, planet_lib.VICTORY)

  def test_colonists_can_replan_a_worked_chain(self):
    context = game_lib.new_game('light_delay')
    actor = actors_lib.ColonyActor('Colonists', context.planet, 0.15, 0.4)
    self.assertIn(
        'plan a balanced chain to run the terraforming plants', actor.options()
    )

  def test_a_warm_but_shrunken_colony_still_loses(self):
    context = game_lib.new_game('light_delay')
    planet = context.planet
    colony = planet.colony
    assert colony is not None
    for _ in range(planet.max_ticks - 1):
      planet.tick += 1
    colony.temp_c = 9.0
    colony.population = 15.0
    colony.growth = 0.0
    planet.advance()
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertIn('but only 15 of 20 colonists remain', planet.outcome_reason)

  def test_a_dwindling_colony_is_lost(self):
    context = game_lib.new_game('light_delay')
    colony = context.planet.colony
    assert colony is not None
    colony.population = 4.0
    context.planet.advance()
    self.assertEqual(context.planet.outcome, planet_lib.DEFEAT)
    self.assertIn('was lost', context.planet.outcome_reason)

  def test_mock_colonists_keep_reserves_then_terraform(self):
    colony = _colony()
    options = {
        'a': 'put more labor into: grow food under glass',
        'b': 'put more labor into: run the terraforming plants',
        'c': actors_lib.KEEP_LABOR,
    }
    policy = mock_language_model.colony_policy
    colony.ledger.withdraw(colony_lib.HOLDER, 'food', 5.0, 'test')
    self.assertEqual(policy(colony.sheet(), options), options['a'])
    colony.ledger.deposit(colony_lib.HOLDER, 'food', 50.0, 'test')
    colony.ledger.deposit(colony_lib.HOLDER, 'air', 50.0, 'test')
    self.assertEqual(policy(colony.sheet(), options), options['b'])
    self.assertIsNone(policy('A people sheet.', options))


if __name__ == '__main__':
  absltest.main()

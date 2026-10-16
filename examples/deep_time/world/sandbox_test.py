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

"""Tests for the seed ship and the Sandbox scenario."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib
import numpy as np


def _game():
  context = game_lib.new_game('sandbox')
  planet = context.planet
  civ = planet.civ
  assert civ is not None
  return context, planet, civ


class SeedShipTest(absltest.TestCase):

  def test_the_goal_text_names_the_biodiversity_bar(self):
    config = data.scenario('sandbox')
    bar = f"{float(config['goal']['min_biodiversity']):.2f}"
    self.assertIn(f'at least {bar}', ' '.join(config['steward_goal'].split()))

  def test_seed_vaults_yield_with_biodiversity(self):
    _, _, civ = _game()
    people = civ.polities['amber']
    temps, forest = np.full(48, 15.0), np.zeros(48)
    civ.biodiversity = civ.seed_reference / 2
    half = civ._yield('biodiversity', people, temps, 1.0, forest)  # pylint: disable=protected-access
    civ.biodiversity = civ.seed_reference * 2
    full = civ._yield('biodiversity', people, temps, 1.0, forest)  # pylint: disable=protected-access
    self.assertAlmostEqual(half, 0.5 * full)

  def test_a_seed_ship_from_a_living_world_wins(self):
    _, planet, civ = _game()
    civ.polities['amber'].techs += ['photovoltaics', 'rocketry', 'arks']
    for good, amount in (('rockets', 3.0), ('panels', 2.0), ('seeds', 10.0)):
      civ.ledger.deposit('amber', good, amount, 'test')
    civ.set_allocation('amber', {'farm': 0.5, 'ark_works': 0.5})
    report = planet.advance()
    self.assertIn(
        'Amber folk launch a seed ship toward the stars',
        ' '.join(e.text for e in report.events),
    )
    self.assertLen(planet.launches, 1)
    if planet.launches[0][2] >= 0.15:
      self.assertEqual(planet.outcome, planet_lib.VICTORY)
    else:
      self.assertIsNone(planet.outcome)

  def test_part_built_ships_do_not_count(self):
    _, planet, civ = _game()
    for key in ('amber', 'basalt', 'cedar'):
      civ.ledger.deposit(key, 'seed_ships', 0.6, 'test')
    planet.advance()
    self.assertEqual(planet.launches, [])
    self.assertIsNone(planet.outcome)

  def test_no_ship_by_the_deadline_loses(self):
    _, planet, _ = _game()
    planet.tick = planet.max_ticks - 1
    planet.advance()
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertEqual(planet.outcome_reason, 'no seed ship left for the stars')

  def test_the_ship_is_one_balanced_plan(self):
    _, planet, civ = _game()
    civ.polities['amber'].techs += ['photovoltaics', 'rocketry', 'arks']
    civ.biodiversity = 0.2  # The planet sets it before every step.
    shares, _ = civ.plan('amber', 'ark_works')
    for key in ('ark_works', 'rocket_works', 'panel_factory', 'seed_vault'):
      self.assertGreater(shares[key], 0.0)
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    self.assertIn(
        'plan a balanced chain to build and launch the seed ship',
        amber.options(),
    )

  def test_the_learn_menu_keeps_the_frontier(self):
    _, planet, civ = _game()
    civ.ledger.deposit('amber', 'research', 500.0, 'test')
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    learn = [o for o in amber.options() if o.startswith('learn')]
    self.assertLen(learn, actors_lib.MAX_LEARN)
    self.assertIn('learn photovoltaics (50 research)', learn)

  def test_mock_studies_first_then_plans_the_ship(self):
    goal = 'launch a seed ship to the stars. '
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. Labor: farm the'
        ' land 60%, study and teach 5%. '
    )
    options = {
        'a': 'put more labor into: study and teach',
        'b': 'plan a balanced chain to build and launch the seed ship',
        'c': actors_lib.KEEP_LABOR,
    }
    policy = mock_language_model.polity_policy
    self.assertEqual(policy(goal + sheet, options), options['a'])
    arks = sheet.replace('tools 5.0.', 'tools 5.0, seeds 0.0.')
    self.assertEqual(policy(goal + arks, options), options['b'])


if __name__ == '__main__':
  absltest.main()

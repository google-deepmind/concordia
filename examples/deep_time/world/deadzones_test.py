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

"""Tests for fertiliser runoff and the coastal dead zones it opens."""

import json

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.world import data
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import ocean as ocean_lib
from examples.deep_time.world import publishing
import numpy as np

GRID = grid_lib.RegionGrid(6, 8)
OCEAN = data.load('ocean.yaml')


def _update(ocean: ocean_lib.Ocean, sst_c: float = 15.0) -> list[str]:
  return ocean.update(
      weights=GRID.area_weights(),
      land=np.full(GRID.size, 0.5),
      ice=np.zeros(GRID.size),
      temps_c=np.full(GRID.size, sst_c),
      ocean_carbon=None,
  )


def _fertilising_game(*techs: str):
  """The Great Acceleration, with Amber farming only with fertiliser."""
  context = game_lib.new_game('great_acceleration')
  planet, civ = context.planet, context.planet.civ
  assert civ is not None
  civ.polities['amber'].techs.extend(['chemistry', *techs])
  civ.ledger.deposit('amber', 'fertiliser', 50.0, 'test')
  civ.set_allocation('amber', {'fertilised_farm': 1.0})
  return context, planet, civ


class DeadZonesTest(absltest.TestCase):

  def test_runoff_opens_dead_zones_that_the_seas_flush_away(self):
    ocean = ocean_lib.Ocean(OCEAN, reference_ocean_carbon=1.0)
    alerts = []
    for _ in range(20):
      ocean.feed(2.0)
      alerts += _update(ocean)
    self.assertGreater(ocean.dead_zone, ocean.dead_zone_alert)
    self.assertLessEqual(ocean.dead_zone, ocean.max_dead_zone)
    self.assertLen(alerts, 1)  # Once, as the dead zones pass the alert.
    self.assertIn('Dead zones spread along the coasts', alerts[0])
    self.assertEqual(ocean.payload(0.5)['dead_zone'], round(ocean.dead_zone, 3))
    peak = ocean.dead_zone
    for _ in range(15):
      _update(ocean)
    self.assertLess(ocean.dead_zone, 0.1 * peak)
    restored = ocean_lib.Ocean(OCEAN, 1.0)
    restored.set_state(ocean.get_state())
    self.assertEqual(restored.payload(0.5), ocean.payload(0.5))
    old = ocean.get_state()
    del old['nutrients'], old['dead_zone']  # A checkpoint from before.
    restored.set_state(old)
    self.assertEqual(restored.dead_zone, 0.0)

  def test_warm_seas_hold_less_oxygen(self):
    ocean = ocean_lib.Ocean(OCEAN, reference_ocean_carbon=1.0)
    cold, mild, warm = (ocean.dead_zone_for(4.0, t) for t in (5.0, 15.0, 25.0))
    self.assertLess(cold, mild)
    self.assertLess(mild, warm)
    self.assertAlmostEqual(mild, 0.5 * ocean.max_dead_zone)
    self.assertEqual(ocean.dead_zone_for(0.0, 30.0), 0.0)
    self.assertLessEqual(ocean.dead_zone_for(1e9, 60.0), ocean.max_dead_zone)

  def test_fertilised_fields_wash_runoff_to_sea_and_buffers_catch_it(self):
    _, planet, civ = _fertilising_game()
    planet.advance()
    washed = civ.polities['amber'].emissions['runoff']
    self.assertGreater(washed, 0.0)
    self.assertAlmostEqual(planet.ocean.nutrients, washed)
    self.assertIn('washed down the rivers to the sea', civ.sheet('amber'))
    _, buffered_planet, buffered = _fertilising_game('wetland_buffers')
    buffered_planet.advance()
    self.assertAlmostEqual(
        buffered.polities['amber'].emissions['runoff'], 0.35 * washed
    )
    # Grain is untouched: buffers only catch the runoff.
    self.assertAlmostEqual(
        buffered.ledger.stock('amber', 'grain'),
        civ.ledger.stock('amber', 'grain'),
    )

  def test_dead_zones_cut_the_catch_and_stop_the_fish_breeding(self):
    games = []
    for nutrients in (0.0, 1000.0):
      context = game_lib.new_game('great_acceleration')
      planet, civ = context.planet, context.planet.civ
      assert civ is not None
      civ.fish_stock = 0.5
      planet.ocean.nutrients = nutrients
      planet.advance()
      games.append((planet, civ))
    (_, clean), (planet, choked) = games[0], games[1]
    self.assertGreater(choked.dead_zone, 0.5)
    self.assertEqual(choked.dead_zone, planet.ocean.dead_zone)
    self.assertLess(choked.fish_stock, clean.fish_stock)
    amber = choked.polities['amber']
    temps, forest = np.full(48, 15.0), np.zeros(48)
    choked_catch = choked._yield('fish', amber, temps, 1.0, forest)  # pylint: disable=protected-access
    choked.dead_zone = 0.0
    full_catch = choked._yield('fish', amber, temps, 1.0, forest)  # pylint: disable=protected-access
    self.assertAlmostEqual(
        choked_catch, full_catch * (1.0 - planet.ocean.dead_zone)
    )

  def test_the_sheet_and_the_tide_gauge_report_dead_zones(self):
    context = game_lib.new_game('great_acceleration')
    civ = context.planet.civ
    assert civ is not None
    civ.dead_zone = 0.3
    self.assertIn(
        'Dead zones choke 30% of the coastal seas', civ.sheet('amber')
    )
    payload = context.planet.ocean_payload()
    payload['dead_zone'] = 0.3
    message = topics.Message(
        topic='ocean.global', payload=json.dumps(payload), seq=1
    )
    self.assertIn('Dead zones choke 30%', publishing.render_ocean(message))
    payload['dead_zone'] = 0.0
    clean = topics.Message(
        topic='ocean.global', payload=json.dumps(payload), seq=2
    )
    self.assertNotIn('Dead zones', publishing.render_ocean(clean))

  def test_dead_zones_survive_a_checkpoint(self):
    context, planet, _ = _fertilising_game()
    planet.advance()
    planet.advance()
    state = context.get_state()
    other = game_lib.new_game('great_acceleration')
    other.set_state(state)
    self.assertEqual(other.planet.ocean.nutrients, planet.ocean.nutrients)
    self.assertEqual(other.planet.ocean.dead_zone, planet.ocean.dead_zone)
    # The peoples see the same seas, and their last runoff, after a resume.
    civ, other_civ = planet.civ, other.planet.civ
    assert civ is not None and other_civ is not None
    self.assertEqual(other_civ.dead_zone, civ.dead_zone)
    self.assertEqual(other_civ.sheet('amber'), civ.sheet('amber'))


class MarineLifeTest(absltest.TestCase):

  def test_empty_and_choking_seas_weigh_on_biodiversity(self):
    context = game_lib.new_game('great_acceleration')
    planet, civ = context.planet, context.planet.civ
    assert civ is not None
    full = planet.biodiversity()
    self.assertEqual(planet.sea_health(), 1.0)  # Healthy seas cost nothing.
    civ.fish_stock = 0.5
    planet.ocean.dead_zone = 0.2
    health = planet.sea_health()
    assert health is not None
    self.assertAlmostEqual(health, 0.4)
    weight = planet.ocean.marine_weight
    self.assertAlmostEqual(
        planet.biodiversity(), full * (1 - weight * (1 - 0.4))
    )
    context.publish_state()
    meter = context.bus.latest('gaia.indices')
    assert meter is not None
    self.assertIn(
        'the seas at 40% of their life', publishing.render_indices(meter)
    )

  def test_the_steward_can_restore_the_coastal_marshes(self):
    planet = game_lib.new_game('great_acceleration').planet
    civ = planet.civ
    assert civ is not None
    marshes = planet.intervention_for_option(
        next(i.option for i in planet.interventions() if 'marshes' in i.label)
    )
    assert marshes is not None
    planet.omega, planet.ocean.nutrients, civ.fish_stock = 10.0, 4.0, 0.3
    text = planet.apply_intervention('Gaia', marshes)
    self.assertIn('the fish stocks stand at 40%', text)
    self.assertAlmostEqual(planet.ocean.nutrients, 2.0)
    self.assertAlmostEqual(planet.omega, 10.0 - marshes.cost)
    # Not on worlds without fisheries.
    daisy = game_lib.new_game('daisyworld').planet
    self.assertFalse(any('marshes' in i.label for i in daisy.interventions()))

  def test_the_mock_steward_restores_dying_seas(self):
    options = {
        'a': 'wait and save omega',
        'b': (
            'restore the coastal marshes (the seas breathe, the fish return)'
            ' [4 omega]'
        ),
    }
    sheet = (
        'Global mean temperature 15.0 C (habitable band 5 to 30 C).'
        ' Omega {omega}; biodiversity 0.12 (the seas at {seas}% of their'
        ' life); stability 0.90.'
    )
    policy = mock_language_model.steward_policy
    self.assertEqual(
        policy(sheet.format(omega=8, seas=30), options), options['b']
    )
    self.assertNotEqual(
        policy(sheet.format(omega=8, seas=80), options), options['b']
    )
    self.assertNotEqual(
        policy(sheet.format(omega=4, seas=30), options), options['b']
    )

  def test_recharging_interventions_are_not_offered(self):
    planet = game_lib.new_game('great_acceleration').planet
    steward = actors_lib.StewardActor('Gaia', planet)
    marsh = next(o for o in steward.options() if 'marshes' in o)
    planet.omega = 20.0
    intervention = planet.intervention_for_option(marsh)
    assert intervention is not None
    planet.apply_intervention('Gaia', intervention)
    self.assertNotIn(marsh, steward.options())
    planet.tick += intervention.cooldown
    self.assertIn(marsh, steward.options())

  def test_a_civilization_without_fisheries_keeps_its_biodiversity(self):
    planet = game_lib.new_game('great_acceleration').planet
    civ = planet.civ
    assert civ is not None
    civ.fish_regrowth, civ.fish_stock = 0.0, 0.0
    self.assertIsNone(planet.sea_health())

  def test_worlds_without_fisheries_are_unchanged(self):
    planet = game_lib.new_game('daisyworld').planet
    self.assertIsNone(planet.sea_health())
    self.assertIsNone(planet.indices_payload()['seas'])

  def test_a_too_poor_world_never_reads_as_rich_enough(self):
    planet = game_lib.new_game('sandbox').planet
    planet.launches = [(20, 'amber', 0.1496)]  # Rounds to the 0.15 needed.
    planet.vault = 1.0  # The vault was full: the world itself was poor.
    planet.tick = planet.max_ticks
    planet._judge()  # pylint: disable=protected-access
    self.assertIn('(biodiversity 0.14 at best)', planet.outcome_reason)

  def test_a_shortfall_never_reads_as_the_target(self):
    planet = game_lib.new_game('fire_and_iron').planet
    civ = planet.civ
    assert civ is not None
    civ.totals['iron'] = 109.7  # Rounds to the 110 needed.
    planet.tick = planet.max_ticks
    planet._judge()  # pylint: disable=protected-access
    self.assertEqual(
        planet.outcome_reason, 'only 109 of the 110 iron was forged'
    )


class MockPeoplesTest(absltest.TestCase):

  SHEET = (
      '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
      ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. Labor: farm the'
      ' land {fields}%. Research 60.0; we know: electricity. '
  )

  def test_a_people_bound_to_its_fields_studies_fertiliser(self):
    # A people leaving coal studies the clean path right after electricity,
    # unless its fields take so many hands that fertiliser comes first.
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'learn glassmaking (30 research)',
        'c': (
            'learn industrial chemistry (fertiliser from the air) (44 research)'
        ),
        'd': 'keep the current balance of labor',
    }
    clean = ' Our sky watch: the air holds 1.20 times its starting CO2.'
    policy = mock_language_model.polity_policy
    self.assertEqual(
        policy(self.SHEET.format(fields=45) + clean, options), options['c']
    )
    self.assertEqual(
        policy(self.SHEET.format(fields=20) + clean, options), options['b']
    )

  def test_a_people_whose_runoff_feeds_dead_zones_buffers_it(self):
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'learn glassmaking (30 research)',
        'c': (
            'learn wetlands and buffer strips (catch fertiliser runoff) (30'
            ' research)'
        ),
        'd': 'keep the current balance of labor',
    }
    sheet = self.SHEET.format(fields=20) + (
        ' Dead zones choke 25% of the coastal seas, and fish neither live'
        ' nor breed there. Last generation 1.2 of our fertiliser washed down'
        ' the rivers to the sea.'
    )
    policy = mock_language_model.polity_policy
    self.assertEqual(policy(sheet, options), options['c'])
    # Someone else's runoff: no reason to pay for our own buffers first.
    others = sheet.replace(
        ' Last generation 1.2 of our fertiliser washed'
        ' down the rivers to the sea.',
        '',
    )
    self.assertEqual(policy(others, options), options['b'])

  def test_a_hungry_people_that_knows_fertiliser_plans_the_chain(self):
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'plan a balanced chain to farm with fertiliser',
        'c': 'keep the current balance of labor',
    }
    sheet = self.SHEET.format(fields=50).replace('fed 100%', 'fed 70%')
    self.assertEqual(
        mock_language_model.polity_policy(sheet, options), options['b']
    )
    del options['b']
    self.assertEqual(
        mock_language_model.polity_policy(sheet, options), options['a']
    )


if __name__ == '__main__':
  absltest.main()

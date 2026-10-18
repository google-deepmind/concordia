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

"""Tests for moves aimed at a place on the map."""

import json

from absl.testing import absltest
from examples.deep_time import game as game_lib
import numpy as np


def _planet(scenario: str = 'daisyworld'):
  return game_lib.new_game(scenario).planet


def _intervention(planet, key):
  return next(i for i in planet.interventions() if i.key == key)


class TargetsTest(absltest.TestCase):

  def test_aimed_seeding_plants_on_the_bare_land_nearest_the_tap(self):
    planet = _planet()
    planet.omega = 20.0
    target = (22, 10)
    before = planet.tile_life.covers['white'].copy()
    bare = np.clip(
        planet.tiles.land
        - np.sum(list(planet.tile_life.covers.values()), axis=0),
        0.0,
        None,
    )
    planet.set_target('Gaia', *target)
    text = planet.apply_intervention(
        'Gaia', _intervention(planet, 'seed_white')
    )
    self.assertIn('around', text)
    gained = planet.tile_life.covers['white'] - before
    steps = planet._near_target(target)  # pylint: disable=protected-access
    # The nearest bare land fills first, ring by ring, until it is all planted.
    land = planet.tiles.land
    last = int(steps[gained > 1e-12].max())
    for ring in range(last):  # Inner rings full (to 80% of their bare land).
      inside = (steps == ring) & land
      np.testing.assert_allclose(gained[inside], 0.8 * bare[inside], atol=1e-9)
    self.assertEqual(float(gained[steps > last].sum()), 0.0)
    self.assertIn('0.06 of the planet', text)  # As much as unaimed seeding.

  def test_aimed_seeding_plants_as_much_anywhere_even_from_the_sea(self):
    planted = []
    for target in ((15, 3), (28, 20), (1, 33)):
      planet = _planet('fire_and_iron')
      planet.omega = 20.0
      before = planet.daisies.cover('forests')
      planet.set_target('Gaia', *target)
      planet.apply_intervention('Gaia', _intervention(planet, 'plant_forests'))
      planted.append(planet.daisies.cover('forests') - before)
    sea = _planet('fire_and_iron')
    tile = next(zip(*np.nonzero(~sea.tiles.land)))
    sea.omega = 20.0
    before = sea.daisies.cover('forests')
    sea.set_target('Gaia', *tile)
    sea.apply_intervention('Gaia', _intervention(sea, 'plant_forests'))
    planted.append(sea.daisies.cover('forests') - before)
    np.testing.assert_allclose(planted, planted[0], rtol=0.02)
    self.assertGreater(planted[0], 0.03)
    for species in planet.daisies.species():  # Tiles and regions agree.
      np.testing.assert_allclose(
          planet.tile_life.region_mean(planet.tile_life.covers[species]),
          planet.daisies.cover_map(species),
          atol=1e-9,
      )
    self.assertEqual(planet.targets, {})  # A target is used once.

  def test_aimed_clouds_gather_over_the_tapped_region(self):
    planet = _planet()
    planet.omega = 20.0
    target = (5, 30)
    planet.set_target('Gaia', *target)
    planet.apply_intervention('Gaia', _intervention(planet, 'bright_clouds'))
    local = planet.daisies.local_cloud()
    region = int(planet.tiles.region[target])
    self.assertEqual(int(np.argmax(local)), region)
    far = [r for r in range(planet.grid.size) if local[r] == 0]
    self.assertNotEmpty(far)
    self.assertAlmostEqual(
        planet.daisies.cloud_albedo(),
        float(_intervention(planet, 'bright_clouds').effect['delta']),
        places=6,
    )

  def test_untargeted_moves_spread_as_before(self):
    planet = _planet()
    planet.omega = 20.0
    planet.apply_intervention('Gaia', _intervention(planet, 'bright_clouds'))
    self.assertFalse(planet.daisies.local_cloud().any())

  def test_the_session_passes_a_target_only_for_aimable_moves(self):
    from examples.deep_time.web import session as session_lib  # pylint: disable=g-import-not-at-top

    game = session_lib.GameSession('daisyworld', 'play', role='Gaia')
    aim = game._targetable  # pylint: disable=protected-access
    self.assertTrue(aim('Gaia', 'seed white daisies [2 omega]'))
    self.assertFalse(aim('Gaia', 'wait and save omega'))
    self.assertFalse(
        aim(
            'Gaia',
            'build better instruments (sharper, faster readings) [3 omega]',
        )
    )
    self.assertTrue(aim('White daisies', 'spread aggressively'))
    self.assertFalse(aim('White daisies', 'hold steady'))

  def test_a_species_spreads_where_it_is_aimed_for_one_tick(self):
    planet = _planet()
    target = (22, 10)
    planet.set_target('White daisies', *target)
    text = planet.set_behaviour('white', 'spread', actor='White daisies')
    self.assertIn('around', text)
    factors = planet.daisies.growth_factors('white')
    region = int(planet.tiles.region[target])
    self.assertEqual(int(np.argmax(factors)), region)
    planet.advance()
    np.testing.assert_allclose(  # The push lasts one tick.
        planet.daisies.growth_factors('white'),
        planet.daisies.growth_factors('white')[0],
    )

  def test_aimed_moves_survive_a_checkpoint(self):
    for scenario, aim in (
        ('daisyworld', lambda p: p.set_behaviour('white', 'spread', 'W')),
        ('fire_and_iron', lambda p: p.aim_felling('amber', 'W')),
    ):
      live = game_lib.new_game(scenario, seed=2)
      live.planet.advance()
      live.planet.set_target('W', 20, 12)
      aim(live.planet)
      saved = json.loads(json.dumps(live.get_state()))
      restored = game_lib.new_game(scenario, seed=2)
      restored.set_state(saved)
      for _ in range(3):
        live.planet.advance()
        restored.planet.advance()
      self.assertEqual(
          json.dumps(live.planet.get_state(), sort_keys=True),
          json.dumps(restored.planet.get_state(), sort_keys=True),
          scenario,
      )

  def test_a_people_aims_its_woodcutters(self):
    planet = _planet('fire_and_iron')
    key = 'amber'
    polity = planet.civ.polities[key]
    towns = planet.settlements
    lands = np.isin(planet.tiles.region, polity.regions) & planet.tiles.land
    taken = set()
    for v in towns.places.values():
      taken.update((s.row, s.col) for s in v)
    spot = next(
        (r, c)
        for r, c in zip(*np.nonzero(lands))
        if all(
            towns._steps(r, c, tr, tc) >= 3 for tr, tc in taken  # pylint: disable=protected-access
        )
    )
    planet.set_target('Amber folk', *spot)
    self.assertIn('around', planet.aim_felling(key, 'Amber folk'))
    self.assertIsNotNone(polity.felling)

  def test_woodcutters_may_be_aimed(self):
    from examples.deep_time import actors as actors_lib  # pylint: disable=g-import-not-at-top

    people = actors_lib.PolityActor(
        'Amber folk', 'amber', _planet('fire_and_iron'), 0.1
    )
    self.assertTrue(people.aimable('put more labor into: cut wood'))


if __name__ == '__main__':
  absltest.main()

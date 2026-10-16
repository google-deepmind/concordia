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

"""Tests for sessile life on tiles."""

import json

from absl.testing import absltest
from examples.deep_time import game as game_lib
import numpy as np


def _planet(scenario: str = 'daisyworld'):
  return game_lib.new_game(scenario).planet


class TileLifeTest(absltest.TestCase):

  def test_regions_aggregate_the_tiles(self):
    planet = _planet('great_oxidation')
    for _ in range(3):
      planet.advance()
    life = planet.tile_life
    for species in planet.daisies.species():
      np.testing.assert_allclose(
          life.region_mean(life.covers[species]),
          planet.daisies.cover_map(species),
          atol=1e-9,
      )
    total = np.sum(list(life.covers.values()), axis=0)
    self.assertTrue(np.all(total <= planet.tiles.land + 1e-9))
    self.assertTrue(np.all(total[~planet.tiles.land] == 0))

  def test_region_writers_reconcile_onto_tiles(self):
    planet = _planet()
    life = planet.tile_life
    before = life.covers['white'].sum()
    planted = planet.daisies.seed('white', 0.05)
    self.assertGreater(planted, 0)
    life.reconcile(planet.daisies)
    np.testing.assert_allclose(
        life.region_mean(life.covers['white']),
        planet.daisies.cover_map('white'),
        atol=1e-9,
    )
    self.assertGreater(life.covers['white'].sum(), before)
    planet.daisies.kill('white', 0.5)
    life.reconcile(planet.daisies)
    np.testing.assert_allclose(
        life.region_mean(life.covers['white']),
        planet.daisies.cover_map('white'),
        atol=1e-9,
    )

  def test_species_spread_to_neighbouring_tiles_across_borders(self):
    planet = _planet()
    life = planet.tile_life
    daisies = planet.daisies
    floor = daisies.params.min_cover
    white = np.full_like(life.covers['white'], floor)
    region = planet.grid.region(2, 3)
    block = life.tiles.region == region
    white[block] = 0.6
    life.covers['white'] = white
    life.covers['black'] = np.full_like(white, floor)
    # Keep the daisy model in step with the tiles we crafted.
    daisies._cover['white'] = life.region_mean(white)  # pylint: disable=protected-access
    daisies._cover['black'] = life.region_mean(life.covers['black'])  # pylint: disable=protected-access
    rows, cols = np.nonzero(block)
    edge = (rows.min(), cols.max() + 1)  # Just outside, in the next region.
    self.assertNotEqual(life.tiles.region[edge], region)
    before = life.covers['white'][edge]
    for _ in range(4):
      life.grow(daisies, daisies.params.dt)
    self.assertGreater(life.covers['white'][edge], before)

  def test_dominant_species_and_checkpoint(self):
    context = game_lib.new_game('great_oxidation')
    planet = context.planet
    planet.advance()
    keys, dominant, total = planet.tile_life.dominant()
    self.assertEqual(keys, planet.daisies.species())
    self.assertEqual(dominant.shape, (30, 40))
    np.testing.assert_allclose(total, sum(planet.tile_life.covers.values()))
    other = game_lib.new_game('great_oxidation')
    other.set_state(context.get_state())
    for k in keys:
      np.testing.assert_allclose(
          other.planet.tile_life.covers[k],
          planet.tile_life.covers[k],
          atol=1e-5,
      )


class RestoreTest(absltest.TestCase):

  def test_reconcile_lets_one_species_grow_where_another_shrank(self):
    planet = _planet()
    daisies, life = planet.daisies, planet.tile_life
    first, last = daisies.species()[0], daisies.species()[-1]
    daisies.seed(last, 1.0)  # A full meadow.
    life.reconcile(daisies)
    daisies.seed(first, daisies.kill(last, 0.3))
    life.reconcile(daisies)
    for species in daisies.species():
      np.testing.assert_allclose(
          life.region_mean(life.covers[species]),
          daisies.cover_map(species),
          atol=0.01,
      )

  def test_a_restored_game_plays_on_exactly(self):
    for scenario, seed in (('great_acceleration', 3), ('daisyworld', 2)):
      live = game_lib.new_game(scenario, seed=seed)
      for _ in range(6):
        live.planet.advance()
      saved = json.loads(json.dumps(live.get_state()))
      restored = game_lib.new_game(scenario, seed=seed)
      restored.set_state(saved)
      for _ in range(6):
        live.planet.advance()
        restored.planet.advance()
      self.assertEqual(
          json.dumps(live.planet.get_state(), sort_keys=True),
          json.dumps(restored.planet.get_state(), sort_keys=True),
          scenario,
      )

  def test_a_restored_people_plans_as_the_live_one_does(self):
    live = game_lib.new_game('great_acceleration', seed=0)
    for _ in range(10):
      live.planet.advance()
    saved = json.loads(json.dumps(live.get_state()))
    restored = game_lib.new_game('great_acceleration', seed=0)
    restored.set_state(saved)
    for game in (live, restored):  # A choice made between ticks.
      civ = game.planet.civ
      assert civ is not None
      civ.adopt_plan('amber', 'smithy')
    for _ in range(3):
      live.planet.advance()
      restored.planet.advance()
    self.assertEqual(
        json.dumps(live.planet.get_state(), sort_keys=True),
        json.dumps(restored.planet.get_state(), sort_keys=True),
    )


if __name__ == '__main__':
  absltest.main()

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

"""Tests for herds and packs on the tile map."""

import copy

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time.world import herds as herds_lib
from examples.deep_time.world import tiles as tiles_lib
import numpy as np


def _planet(scenario: str = 'great_acceleration'):
  return game_lib.new_game(scenario).planet


class HerdsTest(absltest.TestCase):

  def test_species_live_where_there_is_food_for_them(self):
    self.assertEqual(
        list(_planet().herds.specs), ['bison', 'wolves', 'fish', 'geese']
    )
    self.assertEqual(list(_planet('light_delay').herds.specs), [])
    self.assertEqual(list(_planet('daisyworld').herds.specs), [])
    herds = _planet().herds
    for key, species in herds.herds.items():
      habitat = herds.habitat(herds.specs[key])  # Land, or sea for fish.
      for h in species:
        self.assertGreater(habitat[h.row, h.col], 0, key)

  def test_herds_graze_their_own_tile(self):
    planet = _planet()
    herds = planet.herds
    herd = herds.herds['bison'][0]
    covers = planet.tile_life.covers
    floor = np.zeros_like(planet.tiles.land, dtype=float)
    burned = np.zeros_like(planet.tiles.land)
    herds.herds = {k: [] for k in herds.herds}
    herds.herds['bison'] = [herd]
    row, col = herd.row, herd.col
    _, eaten, _ = herds.step(1, covers, burned, floor)
    total = np.sum(list(eaten.values()), axis=0)
    self.assertGreater(float(total[row, col]), 0.0)
    total[row, col] = 0.0
    self.assertEqual(float(total.sum()), 0.0)  # Only its own tile.

  def test_packs_hunt_nearby_herds_and_herds_flee_fire(self):
    planet = _planet()
    herds = planet.herds
    prey = herds.herds['bison'][0]
    far = herds_lib.Herd(999, (prey.row + 10) % herds.tiles.rows, prey.col, 1.0)
    pack = herds_lib.Herd(1000, prey.row, prey.col, 1.0)
    herds.herds = {k: [] for k in herds.herds}
    herds.herds.update(bison=[prey, far], wolves=[pack])
    before_prey, before_far = prey.size, far.size
    suit = herds.suitability(herds.specs['wolves'], herds.biomes.temperature)
    crowd = herds._density(['wolves'])  # pylint: disable=protected-access
    herds._hunt(herds.specs['wolves'], [pack], suit, crowd)  # pylint: disable=protected-access
    self.assertLess(prey.size, before_prey)
    self.assertEqual(far.size, before_far)
    burned = np.zeros_like(planet.tiles.land)
    burned[far.row, far.col] = True
    floor = np.zeros_like(planet.tiles.land, dtype=float)
    size = far.size
    herds.step(2, planet.tile_life.covers, burned, floor)
    self.assertLess(far.size, size)

  def test_herds_move_split_and_replay_alike(self):
    one, two = _planet(), _planet()
    for _ in range(12):
      one.advance()
      two.advance()
    self.assertEqual(one.herds.get_state(), two.herds.get_state())
    moved = one.herds.get_state()['herds']['bison']
    start = _planet().herds.get_state()['herds']['bison']
    self.assertNotEqual(
        [(h[1], h[2]) for h in moved[: len(start)]],
        [(h[1], h[2]) for h in start],
    )

  def test_crowded_herds_can_fall_sick(self):
    planet = _planet()
    herds = planet.herds
    a = herds.herds['bison'][0]
    pair = [
        herds_lib.Herd(1, a.row, a.col, 1.8),
        herds_lib.Herd(2, a.row, (a.col + 1) % herds.tiles.cols, 1.8),
        herds_lib.Herd(3, a.row, (a.col + 2) % herds.tiles.cols, 1.8),
    ]
    config = copy.deepcopy(dict(herds.config))
    config['disease'] = dict(config['disease'], chance=1.0)
    herds.config = config
    herds.herds = {k: [] for k in herds.herds}
    herds.herds['bison'] = pair
    news = herds._outbreaks(np.random.default_rng(0))  # pylint: disable=protected-access
    self.assertTrue(all(h.size < 1.8 for h in pair))
    self.assertTrue(
        any('Disease runs through the bison herds' in n for n in news)
    )

  def test_schools_follow_the_fish_stock_and_flee_dead_zones(self):
    planet = _planet()
    herds = planet.herds
    covers = planet.tile_life.covers
    floor = np.zeros_like(planet.tiles.land, dtype=float)
    burned = np.zeros_like(planet.tiles.land)
    herds.step(1, covers, burned, floor, fish_stock=1.0)
    self.assertLen(herds.herds['fish'], 12)
    # Choke the shelf around every school.
    dead = np.zeros_like(planet.tiles.land)
    for h in herds.herds['fish']:
      dead[h.row, h.col] = True
    dead &= planet.tiles.terrain == tiles_lib.SHELF
    herds.step(2, covers, burned, floor, fish_stock=0.5, dead=dead)
    self.assertLen(herds.herds['fish'], 6)
    self.assertAlmostEqual(herds.total('fish'), 6.0)
    sea = ~planet.tiles.land
    self.assertTrue(all(sea[h.row, h.col] for h in herds.herds['fish']))
    for _ in range(2):
      herds.step(3, covers, burned, floor, fish_stock=0.5, dead=dead)
    self.assertFalse(any(dead[h.row, h.col] for h in herds.herds['fish']))

  def test_geese_migrate_with_the_seasons_and_sow_seeds(self):
    planet = _planet()
    herds = planet.herds
    covers = planet.tile_life.covers
    floor = np.zeros_like(planet.tiles.land, dtype=float)
    burned = np.zeros_like(planet.tiles.land)
    lats = []
    sown_any = False
    for tick in range(6):
      _, _, sown = herds.step(tick, covers, burned, floor)
      sown_any = sown_any or any(v.any() for v in sown.values())
      north = [h for h in herds.herds['geese'] if planet.biomes.lat[h.row] > 0]
      lats.append(np.mean([planet.biomes.lat[h.row] for h in north]))
    # Northern flocks sit further north in northern summers (even ticks).
    self.assertGreater(lats[2], lats[3])
    self.assertGreater(lats[4], lats[5])
    self.assertTrue(sown_any)

  def test_an_older_checkpoint_without_new_kinds_plays_on(self):
    planet = _planet()
    state = planet.herds.get_state()
    del state['herds']['fish'], state['herds']['geese']
    planet.herds.set_state(state)
    self.assertEqual(planet.herds.herds['fish'], [])
    planet.advance()  # No KeyError; the schools come back with the stock.
    self.assertNotEmpty(planet.herds.herds['fish'])

  def test_checkpoint_round_trip(self):
    planet = _planet()
    for _ in range(4):
      planet.advance()
    state = planet.herds.get_state()
    other = _planet()
    other.herds.set_state(state)
    self.assertEqual(other.herds.get_state(), state)

  def test_walkers_leave_tracks_that_lead_to_where_they_stand(self):
    planet = _planet()
    for _ in range(10):
      planet.advance()
    herds = planet.herds
    walkers = {}
    for k, v in herds.herds.items():
      if herds.specs[k]['kind'] in herds_lib.WALKERS:
        for h in v:
          walkers[h.id] = h
    self.assertEqual(set(herds.trails), set(walkers))  # Not fish or geese.
    for i, trail in herds.trails.items():
      self.assertLessEqual(len(trail), herds_lib.TRAIL_LENGTH)
      self.assertEqual(trail[-1], (walkers[i].row, walkers[i].col))
    # The map gets each track before the beast's own tile, rows north first.
    payload = herds.payload()['trails']
    some = next(i for i, t in herds.trails.items() if len(t) > 2)
    rows = planet.tiles.rows
    self.assertEqual(
        payload[str(some)],
        [[rows - 1 - r, c] for r, c in herds.trails[some][:-1]],
    )

  def test_a_daughter_herd_inherits_its_mothers_track(self):
    planet = _planet()
    herds = planet.herds
    for _ in range(3):
      planet.advance()
    mother = herds.herds['bison'][0]
    track = list(herds.trails[mother.id])
    mother.size = 2.5  # Big enough to split this tick.
    herds._split_and_die('bison', herds.specs['bison'])  # pylint: disable=protected-access
    daughter = herds.herds['bison'][-1]
    self.assertNotEqual(daughter.id, mother.id)
    self.assertNotIn(daughter.id, herds.trails)  # Not until the next record.
    herds._record_trails(herds_lib.WALKERS)  # pylint: disable=protected-access
    self.assertEqual(
        herds.trails[daughter.id],
        (track + [(daughter.row, daughter.col)])[-herds_lib.TRAIL_LENGTH :],
    )

  def test_older_checkpoints_without_tracks_start_them_afresh(self):
    planet = _planet()
    for _ in range(3):
      planet.advance()
    state = planet.herds.get_state()
    del state['trails']
    planet.herds.set_state(state)
    self.assertEqual(planet.herds.trails, {})
    planet.advance()
    self.assertTrue(all(len(t) == 1 for t in planet.herds.trails.values()))

  def test_region_grazers_show_as_flocks_that_follow_their_biomass(self):
    self.assertNotIn('flocks', _planet().herds.specs)  # No region grazers.
    planet = _planet('herds')
    herds, grazers = planet.herds, planet.grazers
    assert grazers is not None
    spec = herds.specs['flocks']
    land = planet.tiles.land & ~planet.tiles.ice
    for flock in herds.herds['flocks']:
      self.assertTrue(land[flock.row, flock.col])
    mass = float(
        np.dot(
            grazers.density,
            np.bincount(
                planet.tiles.region[land], minlength=len(grazers.density)
            ),
        )
    )
    self.assertLen(
        herds.herds['flocks'], int(mass / float(spec['per_herd']) + 0.5)
    )
    before = {f.id for f in herds.herds['flocks']}
    grazers.density = grazers.density * 1.6
    herds.place_flocks(1, grazers, planet.tile_life.covers)
    after = {f.id for f in herds.herds['flocks']}
    self.assertGreater(len(after), len(before))
    self.assertLessEqual(before, after)  # The same flocks, and newcomers.
    self.assertAlmostEqual(
        sum(f.size for f in herds.herds['flocks']),
        1.6 * mass / float(spec['per_herd']),
    )
    self.assertNotIn('Grazers', herds.census())  # Region census has them.

  def test_flocks_are_coloured_and_named_by_what_they_eat(self):
    planet = _planet('herds')
    grazers = planet.grazers
    assert grazers is not None
    grazers.diet_white = 0.9
    planet.herds.place_flocks(1, grazers, planet.tile_life.covers)
    pale = planet.herds.payload()['species'][0]
    self.assertEqual(pale['name'], 'Grazers (eat white petals)')
    grazers.diet_white = 0.1
    planet.herds.place_flocks(2, grazers, planet.tile_life.covers)
    dark = planet.herds.payload()['species'][0]
    self.assertEqual(dark['name'], 'Grazers (eat black petals)')
    self.assertGreater(sum(pale['light']), sum(dark['light']) + 300)

  def test_flocks_keep_their_tracks_and_replay_from_a_checkpoint(self):
    planet = _planet('herds')
    for _ in range(4):
      planet.advance()
    flocks = planet.herds.herds['flocks']
    self.assertTrue(any(len(planet.herds.trails[f.id]) > 1 for f in flocks))
    state = copy.deepcopy(planet.get_state())
    planet.advance()
    other = _planet('herds')
    other.set_state(state)
    other.advance()
    self.assertEqual(other.herds.get_state(), planet.herds.get_state())

  def test_tracks_are_swapped_whole_for_the_map_thread(self):
    planet = _planet()
    planet.advance()
    seen = planet.herds.trails
    copy_seen = {i: list(t) for i, t in seen.items()}
    for _ in range(3):
      planet.advance()
    self.assertIsNot(planet.herds.trails, seen)
    self.assertEqual(seen, copy_seen)  # What a reader held is untouched.


if __name__ == '__main__':
  absltest.main()

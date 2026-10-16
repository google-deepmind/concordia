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

"""Tests for the Great Migration scenario and thinning the packs."""

import json

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import game as game_lib
from examples.deep_time import sweep
from examples.deep_time.world import publishing


def _thin(planet):
  return next(i for i in planet.interventions() if i.key == 'thin_packs')


class GreatMigrationTest(absltest.TestCase):

  def test_scenario_wolves_are_bolder_than_elsewhere(self):
    planet = game_lib.new_game('great_migration').planet
    self.assertLen(planet.herds.herds['wolves'], 6)
    self.assertEqual(planet.herds.specs['wolves']['attack'], 2.6)
    other = game_lib.new_game('savanna_fire').planet
    self.assertLen(other.herds.herds['wolves'], 4)
    self.assertEqual(other.herds.specs['wolves']['attack'], 1.6)

  def test_aimed_thinning_takes_only_the_packs_near_the_tap(self):
    planet = game_lib.new_game('great_migration').planet
    planet.omega = 20.0
    packs = planet.herds.herds['wolves']
    first = packs[0]
    planet.set_target('Gaia', first.row, first.col)
    near = planet._near_target((first.row, first.col))  # pylint: disable=protected-access
    before = {p.id: (p.size, int(near[p.row, p.col])) for p in packs}
    text = planet.apply_intervention('Gaia', _thin(planet))
    self.assertIn('the packs lose', text)
    for pack in packs:
      size, steps = before[pack.id]
      expected = size * 0.5 if steps <= 4 else size
      self.assertAlmostEqual(pack.size, expected)

  def test_unaimed_thinning_takes_a_quarter_of_every_pack(self):
    planet = game_lib.new_game('great_migration').planet
    planet.omega = 20.0
    before = planet.herds.total('wolves')
    planet.apply_intervention('Gaia', _thin(planet))
    self.assertAlmostEqual(planet.herds.total('wolves'), before * 0.75)

  def test_census_tells_the_steward_how_many_animals_are_where(self):
    planet = game_lib.new_game('great_migration').planet
    payload = planet.census_payload()
    bison = payload['herds']['Bison herds']
    self.assertEqual(bison['herds'], 12)
    self.assertAlmostEqual(bison['size'], planet.herds.total('bison'))
    text = publishing.render_census(
        topics.Message('bio.census', json.dumps(payload), seq=0)
    )
    self.assertIn('Bison herds: 12 on the map, total size 12 (', text)
    self.assertIn('Wolf packs: 6 on the map, total size', text)
    self.assertIn('mostly near', text)

  def test_census_tells_whether_the_packs_are_booming(self):
    planet = game_lib.new_game('great_migration').planet
    self.assertNotIn('change', planet.census_payload()['herds']['Wolf packs'])
    for _ in range(5):
      planet.advance()
    wolves = planet.census_payload()['herds']['Wolf packs']
    then = planet.history[-4]['herds_wolves']
    self.assertEqual(wolves['over'], 3)
    self.assertAlmostEqual(
        wolves['change'], planet.herds.total('wolves') / then - 1.0, places=2
    )
    text = publishing.render_census(
        topics.Message('bio.census', json.dumps(planet.census_payload()), seq=0)
    )
    self.assertRegex(
        text,
        r'Wolf packs: \d+ on the map, total size [\d.]+ \(mostly near [^)]*'
        r'(up|down) \d+% in 3 ticks\)',
    )

  def test_progress_counts_the_herds_and_the_packs(self):
    planet = game_lib.new_game('great_migration').planet
    labels = [i['label'] for i in planet.progress()]
    self.assertEqual(
        labels, ['turn', 'bison herds', 'wolf packs', 'climate', 'strain']
    )

  def test_a_steward_keeps_the_herds_and_left_alone_wolves_eat_them_out(self):
    kept = sweep.run_one('great_migration', 'default', 0)
    self.assertEqual(kept['outcome'], 'victory', kept['reason'])
    lost = sweep.run_one('great_migration', 'passive', 0)
    self.assertEqual(lost['outcome'], 'defeat')
    # Herds never come back: the game ends when a kind is gone.
    self.assertEqual(lost['reason'], 'the last of the bison herds are gone')
    self.assertLess(lost['ticks'], 40)

  def test_thinning_where_there_are_no_packs_costs_nothing(self):
    planet = game_lib.new_game('great_migration').planet
    planet.omega = 10.0
    near = None
    packs = planet.herds.herds['wolves']
    for row in range(planet.tiles.rows):
      for col in range(planet.tiles.cols):
        steps = planet._near_target((row, col))  # pylint: disable=protected-access
        if all(steps[p.row, p.col] > 4 for p in packs):
          near = (row, col)
          break
      if near:
        break
    assert near is not None
    planet.set_target('Gaia', *near)
    text = planet.apply_intervention('Gaia', _thin(planet))
    self.assertIn('no packs within 4 tiles', text)
    self.assertEqual(planet.omega, 10.0)
    planet.herds.herds['wolves'] = []
    text = planet.apply_intervention('Gaia', _thin(planet))
    self.assertIn('there are none', text)
    self.assertEqual(planet.omega, 10.0)

  def test_the_game_ends_when_a_kind_is_gone(self):
    planet = game_lib.new_game('great_migration').planet
    planet.herds.herds['wolves'] = []
    planet.advance()
    self.assertEqual(planet.outcome, 'defeat')
    self.assertEqual(
        planet.outcome_reason, 'the last of the wolf packs are gone'
    )


if __name__ == '__main__':
  absltest.main()

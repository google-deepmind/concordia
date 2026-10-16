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

"""Tests for speciation and evolutionary dynamics in Deep Time."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time.world import climate as climate_lib
from examples.deep_time.world import daisyworld
from examples.deep_time.world import data
from examples.deep_time.world import grid as grid_lib
from examples.deep_time.world import speciation
import numpy as np

GRID = grid_lib.RegionGrid(6, 8)


def _world() -> daisyworld.Daisyworld:
  land = np.ones(GRID.size)
  return daisyworld.from_config(
      data.load('daisyworld.yaml'),
      {'white': 0.3, 'black': 0.2},
      climate_lib.EnergyBalanceClimate(data.load('climate.yaml'), GRID, land),
      land,
      1.2,
  )


def _speciation(**overrides) -> speciation.Speciation:
  config = data.load('species.yaml')['speciation']
  config.update(overrides)
  return speciation.Speciation(config)


def _key(parent: str) -> str:
  return parent + '9'


class SpeciationTest(parameterized.TestCase):

  def test_spread_is_positive_across_latitudes(self):
    world = _world()
    gap, cold, local = _speciation().spread(world, 'white')
    self.assertGreater(gap, 0.0)
    self.assertTrue(cold.any())
    self.assertLess(local[cold].mean(), local[~cold].mean())

  def test_split_needs_a_streak_and_founds_a_species(self):
    world = _world()
    rules = _speciation(split_threshold_c=0.1, split_ticks=3, earliest_tick=0)
    self.assertIsNone(rules.check(world, GRID, _key, tick=1))
    self.assertIsNone(rules.check(world, GRID, _key, tick=2))
    total_before = world.cover('white') + world.cover('black')
    split = rules.check(world, GRID, _key, tick=3)
    assert split is not None
    self.assertIn(split.child, world.species())
    self.assertEqual(split.child, split.parent + '9')
    self.assertGreater(world.cover(split.child), 0.0)
    total_after = sum(world.cover(s) for s in world.species())
    # Founders move from the parent; only the newcomer's seed bank is new.
    self.assertBetween(
        total_after - total_before, 0.0, world.params.min_cover + 1e-6
    )
    parent_albedo = world.traits(split.parent).albedo
    if split.colder:
      self.assertLess(split.albedo, parent_albedo)
    else:
      self.assertGreater(split.albedo, parent_albedo)
    self.assertEqual(rules.founded, 1)
    self.assertTrue(split.where.endswith(('N', 'S')))

  def test_guards(self):
    world = _world()
    early = _speciation(split_threshold_c=0.1, split_ticks=1, earliest_tick=5)
    self.assertIsNone(early.check(world, GRID, _key, tick=4))
    full = _speciation(split_threshold_c=0.1, split_ticks=1, max_species=2)
    self.assertIsNone(full.check(world, GRID, _key, tick=10))
    calm = _speciation(split_threshold_c=99.0, split_ticks=1)
    self.assertIsNone(calm.check(world, GRID, _key, tick=10))
    with self.assertRaises(ValueError):
      world.add_species('white', world.traits('white'), np.zeros(GRID.size))

  @parameterized.parameters(
      (
          'Name: Frost daisies. Lore: They bloom under snow.',
          ('Frost daisies', 'They bloom under snow.'),
      ),
      (
          'name: "ember bells". lore: Hot and proud.\nMore text',
          ('Ember bells daisies', 'Hot and proud.'),
      ),
      ('I would call them nothing.', None),
  )
  def test_parse_name(self, text, expected):
    self.assertEqual(speciation.parse_name(text), expected)

  def test_parse_name_uses_the_scenario_noun(self):
    self.assertEqual(
        speciation.parse_name('Name: Vent. Lore: Warm water.', 'mats'),
        ('Vent mats', 'Warm water.'),
    )

  def test_state_round_trip(self):
    rules = _speciation(split_threshold_c=0.1)
    rules.check(_world(), GRID, _key, tick=1)
    restored = _speciation()
    restored.set_state(rules.get_state())
    self.assertEqual(restored.streaks, rules.streaks)


if __name__ == '__main__':
  absltest.main()

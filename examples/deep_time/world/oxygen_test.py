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

"""Tests for oxygen, methane and the Great Oxidation."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time.world import data
from examples.deep_time.world import oxygen
from examples.deep_time.world import planet as planet_lib
import numpy as np

METABOLISMS = {
    'methanogens': oxygen.ANAEROBIC,
    'cyanobacteria': oxygen.OXYGENIC,
    'aerobes': oxygen.AEROBIC,
}


def _chemistry(**initial) -> oxygen.Chemistry:
  config = data.load('oxygen.yaml')
  config['initial'] = {**config['initial'], **initial}
  return oxygen.Chemistry(config, METABOLISMS)


def _intervention(planet: planet_lib.Planet, key: str):
  return {i.key: i for i in planet.interventions()}[key]


def _play(seed: int = 0, policy=None) -> planet_lib.Planet:
  planet = game_lib.new_game('great_oxidation', seed=seed).planet
  while planet.outcome is None:
    if policy is not None:
      key = policy(planet)
      if key:
        planet.apply_intervention('Gaia', _intervention(planet, key))
    planet.advance()
  return planet


def _seed_aerobes_after_goe(planet: planet_lib.Planet) -> str | None:
  chemistry = planet.chemistry
  assert chemistry is not None
  if chemistry.oxygen() > 0.01 and planet.daisies.cover('aerobes') < 0.08:
    return 'seed_aerobes'
  return None


class ChemistryTest(absltest.TestCase):

  def test_growth_factors(self):
    chemistry = _chemistry(oxygen=0.0)
    self.assertEqual(chemistry.growth_factor(oxygen.ANAEROBIC), 1.0)
    self.assertEqual(chemistry.growth_factor(oxygen.AEROBIC), 0.0)
    self.assertEqual(chemistry.growth_factor(oxygen.OXYGENIC), 1.0)
    poisoned = _chemistry(oxygen=chemistry.toxic_oxygen)
    self.assertEqual(poisoned.growth_factor(oxygen.ANAEROBIC), 0.0)
    half = _chemistry(oxygen=chemistry.aerobic_half)
    self.assertAlmostEqual(half.growth_factor(oxygen.AEROBIC), 0.5)

  def test_methane_forcing_is_concave(self):
    one = _chemistry(methane=1.0).forcing()
    four = _chemistry(methane=4.0).forcing()
    self.assertAlmostEqual(four, 2 * one)
    self.assertEqual(_chemistry(methane=0.0).forcing(), 0.0)

  def test_reductant_absorbs_oxygen_until_exhausted(self):
    context = game_lib.new_game('great_oxidation')
    daisies = context.planet.daisies
    rich = _chemistry(oxygen=0.0, reductant=10.0)
    poor = _chemistry(oxygen=0.0, reductant=0.0)
    for tick in range(5):
      rich.step(daisies, tick)
      poor.step(daisies, tick)
    self.assertLess(rich.oxygen(), 1e-9)
    self.assertLess(rich.reductant(), 10.0)
    # Without iron, oxygen builds up and starts burning the methane.
    self.assertGreater(poor.oxygen(), 1000 * max(rich.oxygen(), 1e-9))
    self.assertLess(poor.methane(), rich.methane())
    self.assertIsNone(rich.oxidized_at)

  def test_books_balance_and_state_round_trips(self):
    context = game_lib.new_game('great_oxidation')
    daisies = context.planet.daisies
    chemistry = _chemistry(reductant=0.1)
    for tick in range(4):
      chemistry.step(daisies, tick)
    self.assertTrue(chemistry.ledger.is_conserved())
    resumed = _chemistry()
    resumed.set_state(chemistry.get_state())
    self.assertEqual(resumed.payload(), chemistry.payload())
    chemistry.step(daisies, 4)
    resumed.step(daisies, 4)
    self.assertEqual(resumed.payload(), chemistry.payload())


class OzoneTest(absltest.TestCase):

  def test_oxygen_builds_a_shield_against_ultraviolet(self):
    bare = _chemistry(oxygen=0.0)
    self.assertAlmostEqual(bare.ozone(), 0.0)
    half = _chemistry(oxygen=bare.ozone_half)
    self.assertAlmostEqual(half.ozone(), 0.5)
    sun = np.array([0.5, 1.0, 1.25])
    np.testing.assert_allclose(bare.uv(sun), sun / 1.25)
    np.testing.assert_allclose(half.uv(sun), 0.5 * sun / 1.25)

  def test_unshielded_sunlight_slows_growth_most_in_the_tropics(self):
    chemistry = _chemistry(oxygen=0.0)
    chemistry.metabolisms = {'mats': oxygen.OXYGENIC}

    class _Daisies:

      def land_fraction(self):
        return np.ones(3)

      def species(self):
        return ['mats']

    sun = np.array([0.5, 1.0, 1.25])
    seared = chemistry.multipliers(_Daisies(), sun)['mats']  # pyrefly: ignore[bad-argument-type]
    self.assertLess(seared[2], seared[1])
    self.assertLess(seared[1], seared[0])
    self.assertAlmostEqual(seared[2], 1.0 - chemistry.uv_penalty)
    plain = chemistry.multipliers(_Daisies())['mats']  # pyrefly: ignore[bad-argument-type]
    np.testing.assert_allclose(plain, np.ones(3))

  def test_the_shield_is_announced_once_and_remembered(self):
    chemistry = _chemistry(oxygen=0.0)
    self.assertIsNone(chemistry.shielded_at)
    chemistry.ledger.deposit(oxygen.AIR, oxygen.OXYGEN, 1.0, 'test')

    class _Daisies:

      def species(self):
        return []

      def cover(self, unused_species):
        return 0.0

    events = chemistry.step(_Daisies(), tick=7)  # pyrefly: ignore[bad-argument-type]
    self.assertTrue(any('ozone shield forms' in e for e in events))
    self.assertEqual(chemistry.shielded_at, 7)
    self.assertFalse(
        any('ozone' in e for e in chemistry.step(_Daisies(), tick=8))  # pyrefly: ignore[bad-argument-type]
    )
    restored = _chemistry()
    restored.set_state(chemistry.get_state())
    self.assertEqual(restored.shielded_at, 7)
    self.assertIn('ozone', chemistry.payload())


class GreatOxidationTest(absltest.TestCase):

  def test_passive_planet_oxidises_but_aerobes_never_arrive(self):
    planet = _play()
    chemistry = planet.chemistry
    assert chemistry is not None
    self.assertIsNotNone(chemistry.oxidized_at)
    self.assertBetween(chemistry.oxidized_at, 8, 32)
    self.assertLess(chemistry.methane(), 0.1)
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertIn('aerobe cover', planet.outcome_reason)

  def test_seeding_aerobes_after_the_crisis_wins(self):
    planet = _play(policy=_seed_aerobes_after_goe)
    self.assertEqual(planet.outcome, planet_lib.VICTORY)

  def test_iron_flood_delays_the_crisis(self):
    passive = _play()
    flooded = _play(
        policy=lambda p: 'iron_flood' if p.tick in (2, 6, 10) else None
    )
    assert passive.chemistry is not None and flooded.chemistry is not None
    self.assertIsNotNone(passive.chemistry.oxidized_at)
    self.assertGreater(
        flooded.chemistry.oxidized_at or planet_lib.math.inf,
        passive.chemistry.oxidized_at,
    )

  def test_daughter_species_inherit_metabolism(self):
    planet = game_lib.new_game('great_oxidation').planet
    chemistry = planet.chemistry
    assert chemistry is not None
    founded = []
    while planet.outcome is None and not founded:
      planet.advance()
      founded = planet.take_new_species()
    self.assertNotEmpty(founded)
    for split in founded:
      self.assertEqual(
          chemistry.metabolisms[split.child],
          chemistry.metabolisms[split.parent],
      )

  def test_methane_warms_and_its_loss_cools(self):
    planet = game_lib.new_game('great_oxidation').planet
    chemistry = planet.chemistry
    assert chemistry is not None
    self.assertGreater(planet.greenhouse_forcing(), chemistry.forcing() - 1e-9)
    self.assertGreater(chemistry.forcing(), 10.0)


if __name__ == '__main__':
  absltest.main()

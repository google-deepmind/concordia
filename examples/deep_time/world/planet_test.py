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

"""Tests for the planet's rules, schedule and judgement."""

import json
import re

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import game as game_lib
from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import publishing
import numpy as np

# A biome tally's words: 'The desert spreads from 40 to 60 of the 1200 tiles.'
TALLY = re.compile(r' of the \d+ tiles')


def _planet(**overrides) -> planet_lib.Planet:
  scenario = data.scenario('daisyworld')
  scenario.update(overrides)
  return planet_lib.Planet(
      scenario,
      data.load('daisyworld.yaml'),
      planet_lib.interventions_from_config(data.load('interventions.yaml')),
  )


def _intervention(planet: planet_lib.Planet, key: str):
  return {i.key: i for i in planet.interventions()}[key]


def _heuristic(planet: planet_lib.Planet) -> str:
  low, high = planet.band()
  temp = planet.temperature_c()
  if temp > high - 5:
    return 'bright_clouds' if planet.omega >= 4 else 'seed_white'
  if temp < low + 5:
    return 'dark_haze' if planet.omega >= 4 else 'seed_black'
  return 'wait'


class PlanetTest(absltest.TestCase):

  def test_luminosity_ramps_then_holds(self):
    planet = _planet()
    schedule = planet.scenario['luminosity']
    self.assertAlmostEqual(planet.luminosity_at(0), schedule['start'])
    ramp = schedule['ramp_ticks']
    self.assertAlmostEqual(planet.luminosity_at(ramp), schedule['end'])
    self.assertAlmostEqual(planet.luminosity_at(ramp + 5), schedule['end'])
    self.assertLess(planet.luminosity_at(ramp // 2), schedule['end'])

  def test_passive_steward_loses(self):
    planet = _planet()
    while planet.outcome is None:
      planet.advance()
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertLess(planet.tick, planet.max_ticks)

  def test_attentive_steward_wins(self):
    planet = _planet()
    while planet.outcome is None:
      planet.apply_intervention(
          'Gaia', _intervention(planet, _heuristic(planet))
      )
      planet.advance()
    self.assertEqual(planet.outcome, planet_lib.VICTORY)
    self.assertEqual(planet.tick, planet.max_ticks)

  def test_shocks_fire_on_schedule(self):
    planet = _planet()
    texts = []
    for _ in range(30):
      # (Biome news is not a shock.)
      texts.extend(t for t in planet.advance().texts() if not TALLY.search(t))
    self.assertLen(texts, len(planet.scenario['shocks']))
    self.assertIn('blight', texts[0])

  def test_interventions_cost_omega_and_respect_cooldown(self):
    planet = _planet()
    planet.omega = 10.0
    clouds = _intervention(planet, 'bright_clouds')
    message = planet.apply_intervention('Gaia', clouds)
    self.assertIn('bright clouds', message)
    self.assertEqual(planet.omega, 6.0)
    self.assertIn('recharges', planet.apply_intervention('Gaia', clouds))
    self.assertEqual(planet.omega, 6.0)
    planet.omega = 1.0
    seed = _intervention(planet, 'seed_white')
    self.assertIn('has only', planet.apply_intervention('Gaia', seed))

  def test_option_text_maps_back_to_intervention(self):
    planet = _planet()
    for intervention in planet.interventions():
      self.assertEqual(
          planet.intervention_for_option(intervention.option), intervention
      )
      self.assertEqual(
          planet.intervention_for_option(intervention.label.upper()),
          intervention,
      )
    self.assertIsNone(planet.intervention_for_option('dance'))

  def test_out_of_band_counter_and_defeat(self):
    planet = _planet(
        band_c=[100.0, 200.0], max_ticks_out_of_band=3, severe_out_of_band_c=500
    )
    for _ in range(3):
      planet.advance()
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertIn('out of band', planet.outcome_reason)

  def test_far_out_of_band_strain_mounts_twice_as_fast(self):
    planet = _planet(band_c=[100.0, 200.0], max_ticks_out_of_band=6)
    planet.advance()
    self.assertEqual(planet.out_of_band, 2.0)  # 80 C below: wrecked.
    for _ in range(2):
      planet.advance()
    self.assertEqual(planet.outcome, planet_lib.DEFEAT)
    self.assertIn('the climate broke', planet.outcome_reason)
    near = _planet(
        band_c=[100.0, 200.0], max_ticks_out_of_band=6, severe_out_of_band_c=500
    )
    near.advance()
    self.assertEqual(near.out_of_band, 1.0)

  def test_strong_moves_wreck_the_climate_within_a_turn(self):
    for key, sign in (('ash_winter', -1), ('runaway_greenhouse', 1)):
      planet = _planet()
      planet.omega = 50.0
      before = planet.temperature_c()
      (move,) = [i for i in planet.interventions() if i.key == key]
      self.assertIn('°C', move.push)
      words = planet.apply_intervention('Gaia', move)
      self.assertIn('sunlight', words)
      planet.advance()
      self.assertGreater(sign * (planet.temperature_c() - before), 15.0)

  def test_band_stress_heals_slowly_with_recovery(self):
    planet = _planet(
        band_c=[100.0, 200.0],
        max_ticks_out_of_band=99,
        band_recovery_per_tick=0.5,
        severe_out_of_band_c=500,
    )
    for _ in range(3):
      planet.advance()
    self.assertEqual(planet.out_of_band, 3.0)
    state = planet.get_state()
    healing = _planet(band_c=[-100.0, 200.0], band_recovery_per_tick=0.5)
    healing.set_state(state)
    healing.advance()
    self.assertEqual(healing.out_of_band, 2.5)
    resetting = _planet(band_c=[-100.0, 200.0], band_recovery_per_tick=None)
    resetting.set_state(state)
    resetting.advance()
    self.assertEqual(resetting.out_of_band, 0.0)

  def test_instrument_upgrades_cost_omega_until_the_best(self):
    tiers = data.load('instruments.yaml')['tiers']
    scenario = data.scenario('daisyworld')
    planet = planet_lib.Planet(
        scenario,
        data.load('daisyworld.yaml'),
        planet_lib.interventions_from_config(data.load('interventions.yaml')),
        instrument_tiers=tiers,
    )
    tier = planet.instrument()
    assert tier is not None
    self.assertEqual(tier['key'], scenario['instruments'])
    self.assertIn('2 ticks late', planet.indices_payload()['instruments'])
    planet.omega = 10.0
    upgrade = _intervention(planet, 'upgrade_instruments')
    while planet.can_upgrade_instruments():
      self.assertIn('built', planet.apply_intervention('Gaia', upgrade))
    tier = planet.instrument()
    assert tier is not None
    self.assertEqual(tier['key'], tiers[-1]['key'])
    omega = planet.omega
    self.assertIn(
        'best instruments', planet.apply_intervention('Gaia', upgrade)
    )
    self.assertEqual(planet.omega, omega)
    self.assertIn('on time', planet.instrument_text())

  def test_indices_are_bounded(self):
    planet = _planet()
    for _ in range(10):
      planet.advance()
      self.assertBetween(planet.biodiversity(), 0.0, 1.0)
      self.assertBetween(planet.stability(), 0.0, 1.0)

  def test_checkpoint_resumes_identically(self):
    planet = _planet()
    for _ in range(8):
      planet.advance()
    state = planet.get_state()
    resumed = _planet()
    resumed.set_state(state)
    for _ in range(8):
      planet.advance()
      resumed.advance()
    self.assertEqual(planet.history, resumed.history)

  def test_a_barren_world_keeps_seed_banks_on_all_its_land(self):
    planet = game_lib.new_game('snowball').planet
    land = planet.tiles.land
    self.assertTrue(planet.tile_life.reached[land].all())


class StarsTest(absltest.TestCase):

  def test_stars_are_open_until_the_game_is_decided(self):
    planet = game_lib.new_game('snowball').planet
    planet.advance()
    stars = planet.stars()
    self.assertEqual(
        [s['text'] for s in stars],
        [
            'Reach the goal',
            'Break the ice by turn 16',
            'Break the ice by turn 13',
        ],
    )
    self.assertEqual({s['met'] for s in stars}, {None})
    self.assertEqual(planet.star_count(), 0)

  def test_a_quicker_win_earns_more_stars(self):
    planet = game_lib.new_game('snowball').planet
    planet.advance()
    planet.outcome = planet_lib.VICTORY
    for tick, count in ((12, 3), (15, 2), (20, 1)):
      planet.tick = tick
      self.assertEqual(planet.star_count(), count, tick)
    planet.outcome = planet_lib.DEFEAT
    self.assertEqual(planet.star_count(), 0)  # Stars need the goal.

  def test_readings_strain_omega_and_colonists_earn_stars(self):
    planet = game_lib.new_game('great_acceleration').planet
    planet.advance()
    planet.outcome = planet_lib.VICTORY
    planet.history[-1]['energy_made'] = 1250.0
    planet.history[-1]['co2'] = 1.2
    self.assertEqual([s['met'] for s in planet.stars()], [True, True, False])
    planet.history[-1]['co2'] = 1.1
    self.assertEqual(planet.star_count(), 3)
    daisy = game_lib.new_game('daisyworld').planet
    daisy.advance()
    daisy.outcome = planet_lib.VICTORY
    daisy.history[-1]['out_of_band'] = 4.0
    self.assertFalse(daisy.stars()[1]['met'])  # Strain 4 > 3.
    colony = game_lib.new_game('light_delay').planet
    colony.advance()
    colony.outcome = planet_lib.VICTORY
    assert colony.colony is not None
    colony.colony.population = 30.0
    self.assertTrue(colony.stars()[2]['met'])


class ReserveTest(absltest.TestCase):

  def test_protected_woods_are_not_felled_until_the_reserve_ends(self):
    planet = game_lib.new_game('sandbox').planet
    for _ in range(3):
      planet.advance()  # The peoples start felling.
    planet.omega = 50.0
    (protect,) = [i for i in planet.interventions() if i.key == 'protect_wilds']
    felled = set(np.nonzero(planet._last_cut > 0)[0].tolist())
    words = planet.apply_intervention('Gaia', protect)
    self.assertIn('may not fell the woods', words)
    regions = sorted(planet.reserves)
    self.assertNotEmpty(regions)
    self.assertContainsSubset(regions, felled)  # Where axes were busiest.
    for _ in range(int(protect.effect['ticks'])):
      planet.advance()
      self.assertEqual(float(planet._last_cut[regions].sum()), 0.0)
      self.assertEqual(float(planet.protected()[regions].min()), 1.0)
    planet.advance()
    self.assertEqual(float(planet.protected().sum()), 0.0)  # Over.

  def test_an_aimed_reserve_covers_the_tapped_land_and_its_neighbours(self):
    planet = game_lib.new_game('sandbox').planet
    planet.omega = 50.0
    (protect,) = [i for i in planet.interventions() if i.key == 'protect_wilds']
    planet.set_target('Gaia', 10, 20)
    planet.apply_intervention('Gaia', protect)
    region = int(planet.tiles.region[10, 20])
    self.assertIn(region, planet.reserves)
    self.assertGreater(len(planet.reserves), 1)
    state = planet.get_state()
    other = game_lib.new_game('sandbox').planet
    other.set_state(state)
    self.assertEqual(other.reserves, planet.reserves)


class TransplantTest(absltest.TestCase):

  def test_seed_is_carried_for_no_omega_but_paid_for_in_cover(self):
    planet = game_lib.new_game('archipelago').planet
    for _ in range(4):
      planet.advance()
    (carry,) = [
        i for i in planet.interventions() if i.key == 'transplant_grass'
    ]
    self.assertEqual(carry.cost, 0)
    planet.omega = 0.0
    before = float(
        planet.climate.global_mean(planet.daisies.cover_map('grass'))
    )
    reached = int(planet.tile_life.reached.sum())
    words = planet.apply_intervention('Gaia', carry)
    self.assertIn('took root', words)
    after = float(planet.climate.global_mean(planet.daisies.cover_map('grass')))
    self.assertEqual(planet.omega, 0.0)  # No omega spent.
    # 4 in 10 of what was carried are lost (at most half the cover goes).
    carried = min(float(carry.effect['amount']), 0.5 * before)
    self.assertAlmostEqual(before - after, 0.4 * carried, delta=0.0005)
    self.assertGreater(int(planet.tile_life.reached.sum()), reached)  # A shore.
    self.assertEqual(planet.made[-1], (planet.tick, 'transplant_grass'))


class SeedVaultTest(absltest.TestCase):

  def test_a_ship_carries_no_more_of_the_wild_than_the_vault(self):
    planet = game_lib.new_game('sandbox').planet
    self.assertTrue(planet.has_vault())
    (gather,) = [i for i in planet.interventions() if i.key == 'gather_seeds']
    planet.omega = 50.0
    life = planet.biodiversity()
    words = planet.apply_intervention('Gaia', gather)
    self.assertAlmostEqual(planet.vault, 0.5 * life, places=6)
    self.assertIn('the seed vault holds', words)
    self.assertEqual(planet.indices_payload()['vault']['need'], 0.17)
    civ = planet.civ
    assert civ is not None
    key = next(iter(civ.polities))
    civ.ledger.deposit(key, 'seed_ships', 1.0, 'test')
    (event,) = planet._launch_seed_ships()
    self.assertIn('the seed vault', event.text)
    self.assertAlmostEqual(
        planet.launches[-1][2], min(planet.biodiversity(), planet.vault)
    )
    state = planet.get_state()
    planet.vault = 0.0
    planet.set_state(state)
    self.assertAlmostEqual(planet.vault, 0.5 * life, places=6)
    self.assertFalse(game_lib.new_game('daisyworld').planet.has_vault())


class SideObjectivesTest(absltest.TestCase):

  def _row(self, planet, **fields):
    planet.history.append({'tick': planet.tick, **fields})

  def test_a_hold_is_paid_when_held_through_its_window(self):
    planet = game_lib.new_game('great_oxidation').planet
    self.assertEqual(planet.side, {'methanogens': planet_lib.SIDE_OPEN})
    omega = planet.omega
    for tick in range(1, 21):
      planet.tick = tick
      self._row(planet, cover_methanogens=0.05)
      planet._judge_side()
      if tick < 20:
        self.assertEqual(planet.side['methanogens'], planet_lib.SIDE_OPEN)
    self.assertEqual(planet.side['methanogens'], planet_lib.SIDE_MET)
    self.assertAlmostEqual(planet.omega, omega + 2.0)
    planet.tick = 21
    self._row(planet, cover_methanogens=0.0)
    planet._judge_side()  # Judged once: no second reward, no change.
    self.assertEqual(planet.side['methanogens'], planet_lib.SIDE_MET)
    self.assertAlmostEqual(planet.omega, omega + 2.0)

  def test_a_hold_is_missed_the_first_turn_it_fails(self):
    planet = game_lib.new_game('great_oxidation').planet
    omega = planet.omega
    planet.tick = 7
    self._row(planet, cover_methanogens=0.019)
    planet._judge_side()
    self.assertEqual(planet.side['methanogens'], planet_lib.SIDE_MISSED)
    self.assertAlmostEqual(planet.omega, omega)
    (task,) = planet.side_payload()
    self.assertEqual(
        (task['state'], task['reward'], task['turns']), ('missed', 2.0, 13)
    )

  def test_a_reach_counts_only_in_its_window(self):
    planet = game_lib.new_game('archipelago').planet
    omega = planet.omega
    planet.tick = 11
    self._row(planet, cover_forests=0.5)  # Before the window: not yet.
    planet._judge_side()
    self.assertEqual(planet.side['woods'], planet_lib.SIDE_OPEN)
    planet.tick = 12
    self._row(planet, cover_forests=0.041)
    planet._judge_side()
    self.assertEqual(planet.side['woods'], planet_lib.SIDE_MET)
    self.assertAlmostEqual(planet.omega, omega + 4.0)
    late = game_lib.new_game('archipelago').planet
    late.tick = 28
    self._row(late, cover_forests=0.01)
    late._judge_side()
    self.assertEqual(late.side['woods'], planet_lib.SIDE_MISSED)

  def test_side_objectives_are_saved_and_read_on_the_omega_meter(self):
    planet = game_lib.new_game('great_oxidation').planet
    planet.side['methanogens'] = planet_lib.SIDE_MET
    state = planet.get_state()
    fresh = game_lib.new_game('great_oxidation').planet
    fresh.set_state(state)
    self.assertEqual(fresh.side, {'methanogens': planet_lib.SIDE_MET})
    payload = game_lib.new_game('great_oxidation').planet.indices_payload()
    text = publishing.render_indices(
        topics.Message('gaia.indices', json.dumps(payload), seq=0)
    )
    self.assertIn(
        'Side objective: Keep the Methanogen mats on 2% of the planet until'
        ' turn 20 (+2 omega) open, 20 turns left.',
        text,
    )
    self.assertEqual(game_lib.new_game('snowball').planet.side, {})


if __name__ == '__main__':
  absltest.main()

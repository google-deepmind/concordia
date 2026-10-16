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

"""Tests for seed banks: the cost of spreading, the source of comebacks."""

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.world import data
from examples.deep_time.world import seedbank as seedbank_lib

CONFIG = data.load('species.yaml')['seed_bank']


class SeedBankTest(absltest.TestCase):

  def test_spreading_spends_and_rest_refills(self):
    banks = seedbank_lib.SeedBanks(CONFIG, ['white'])
    start = banks.level['white']
    banks.spend('white', 'spread')
    self.assertAlmostEqual(
        banks.level['white'], start + CONFIG['change']['spread']
    )
    banks.spend('white', 'dormant')
    banks.spend('white', 'hold')
    self.assertGreater(banks.level['white'], start)
    for _ in range(20):
      banks.spend('white', 'dormant')
    self.assertEqual(banks.level['white'], banks.capacity)

  def test_an_empty_bank_exhausts_a_spreader(self):
    banks = seedbank_lib.SeedBanks(CONFIG, ['white'])
    exhausted = [banks.spend('white', 'spread') for _ in range(8)]
    self.assertFalse(exhausted[0])
    self.assertTrue(exhausted[-1])
    self.assertIn('nearly spent', banks.text('white'))
    self.assertFalse(banks.spend('white', 'hold'))  # Holding never exhausts.

  def test_smothering_spends_the_bank_fast_and_exhausts(self):
    banks = seedbank_lib.SeedBanks(CONFIG, ['white'])
    turns = 0
    while not banks.spend('white', 'smother'):
      turns += 1
    self.assertLessEqual(turns, 2)  # Two or three turns of it, no more.
    self.assertLess(banks.level['white'], CONFIG['exhausted'])

  def test_a_vanishing_species_sprouts_again_once(self):
    banks = seedbank_lib.SeedBanks(CONFIG, ['white'])
    self.assertEqual(banks.reseed('white', 0.5), 0.0)
    amount = banks.reseed('white', 0.001)
    self.assertEqual(amount, banks.reseed_amount)
    banks.charge('white', amount)
    self.assertEqual(banks.reseed('white', 0.001), 0.0)  # Spent.

  def test_a_species_at_the_floor_makes_no_seeds(self):
    banks = seedbank_lib.SeedBanks(CONFIG, ['white'], [])
    for _ in range(20):
      banks.spend('white', 'hold', share_of_land=0.01)
    self.assertEqual(banks.level['white'], 0.0)
    banks.spend('white', 'hold', share_of_land=0.5)
    self.assertGreater(banks.level['white'], 0.0)

  def test_species_not_yet_alive_have_no_seeds(self):
    banks = seedbank_lib.SeedBanks(CONFIG, ['white', 'aerobes'], ['white'])
    self.assertEqual(banks.level['aerobes'], 0.0)
    self.assertEqual(banks.reseed('aerobes', 0.0), 0.0)

  def test_great_oxidation_aerobes_do_not_sprout_for_free(self):
    context = game_lib.new_game('great_oxidation')
    planet = context.planet
    for _ in range(6):
      report = planet.advance()
      self.assertFalse(
          any('seed bank' in e.text for e in report.events), report.events
      )

  def test_planet_exhausts_spreaders_and_shows_the_bank(self):
    context = game_lib.new_game('daisyworld')
    planet = context.planet
    banks = planet.seed_banks
    assert banks is not None
    planet.set_behaviour('white', 'spread')
    banks.level['white'] = 0.0
    planet.advance()
    self.assertEqual(
        planet.daisies.behaviour('white'),
        (banks.exhausted_growth, banks.exhausted_death),
    )
    self.assertIn('Seed bank: 0%', planet.genome_text('white'))
    state = context.get_state()
    other = game_lib.new_game('daisyworld')
    other.set_state(state)
    assert other.planet.seed_banks is not None
    self.assertEqual(other.planet.seed_banks.level, banks.level)

  def test_mock_species_rest_when_the_bank_runs_low(self):
    options = {
        'a': 'spread aggressively',
        'b': 'hold steady',
        'c': 'go dormant',
    }
    prompt = (
        'Genome of the white daisies: optimum 22.0 C; petal albedo 0.75.'
        ' Seed bank: 20%. Census: white daisies cover 30% (local 22.0 C,'
        ' growth 0.80).'
    )
    choice = mock_language_model.species_policy(prompt, options)
    self.assertNotEqual(choice, options['a'])


if __name__ == '__main__':
  absltest.main()

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

"""Tests for evolution rules and their use by the planet."""

import dataclasses
import re

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time.world import daisyworld
from examples.deep_time.world import data
from examples.deep_time.world import evolution as evolution_lib
from examples.deep_time.world import planet as planet_lib

# A biome tally's words: 'The desert spreads from 40 to 60 of the 1200 tiles.'
TALLY = re.compile(r' of the \d+ tiles')


def _evolution() -> evolution_lib.Evolution:
  return evolution_lib.Evolution(data.load('species.yaml'))


def _planet() -> planet_lib.Planet:
  return planet_lib.Planet(
      data.scenario('daisyworld'),
      data.load('daisyworld.yaml'),
      planet_lib.interventions_from_config(data.load('interventions.yaml')),
      _evolution(),
  )


class EvolutionTest(parameterized.TestCase):

  @parameterized.parameters(
      ('Warmer optimum: we run hot.', 'warmer'),
      ('I would choose paler petals, not a warmer optimum.', 'paler'),
      ('no change', 'none'),
      ('Darker petals.', 'darker'),
  )
  def test_parse_adaptation_takes_earliest_option(self, text, key):
    option = _evolution().parse_adaptation(text)
    assert option is not None
    self.assertEqual(option.key, key)

  def test_parse_adaptation_rejects_unknown(self):
    self.assertIsNone(_evolution().parse_adaptation('grow wings'))

  def test_behaviour_lookup(self):
    evolution = _evolution()
    behaviour = evolution.behaviour_for_option('Go dormant')
    assert behaviour is not None
    self.assertEqual(behaviour.key, 'dormant')
    self.assertIsNone(evolution.behaviour_for_option('fly away'))

  def test_steps_are_bounded_and_cost_tolerance(self):
    planet = _planet()
    evolution = _evolution()
    before = planet.daisies.traits('black')
    for _ in range(40):
      evolution.apply_step(
          planet.daisies, 'black', evolution.option('warmer'), 1.0
      )
      evolution.apply_step(
          planet.daisies, 'black', evolution.option('darker'), 1.0
      )
    after = planet.daisies.traits('black')
    self.assertEqual(after.optimum_c, evolution.optimum_range_c[1])
    self.assertEqual(after.albedo, evolution.albedo_range[0])
    self.assertGreater(after.growth_width, before.growth_width)


class GenomeTest(parameterized.TestCase):

  def test_genome_levels_shape_growth_and_death(self):
    effects = daisyworld.GenomeEffects.from_config(
        data.load('species.yaml')['genome']
    )
    plain = daisyworld.SpeciesTraits(0.5, 22.5, 0.003)
    tolerant = dataclasses.replace(plain, tolerance=5)
    vigorous = dataclasses.replace(plain, vigor=5)
    hardy = dataclasses.replace(plain, hardiness=5)
    self.assertLess(effects.niche_width(tolerant), effects.niche_width(plain))
    self.assertGreater(
        effects.growth_factor(vigorous), effects.growth_factor(plain)
    )
    self.assertLess(effects.growth_factor(hardy), effects.growth_factor(plain))
    self.assertLess(effects.death_factor(hardy), effects.death_factor(plain))
    self.assertEqual(daisyworld.GenomeEffects().growth_factor(vigorous), 1.0)

  def test_budget_forces_trade_offs(self):
    planet = _planet()
    evolution = _evolution()
    for _ in range(12):
      evolution.apply_step(
          planet.daisies, 'white', evolution.option('vigorous'), 1.0
      )
    traits = planet.daisies.traits('white')
    self.assertLessEqual(traits.levels(), evolution.genome_budget + 1e-9)
    self.assertGreater(traits.vigor, 2.0)
    self.assertLess(traits.tolerance + traits.hardiness, 4.0)
    self.assertLessEqual(traits.vigor, 10.0)

  @parameterized.parameters(
      ('Broader tolerance, because summers swing.', 'tolerant'),
      ('more vigor!', 'vigorous'),
      ('I choose more hardiness over paler petals', 'hardy'),
  )
  def test_new_options_parse(self, text, key):
    option = _evolution().parse_adaptation(text)
    assert option is not None
    self.assertEqual(option.key, key)

  def test_genome_sheet_and_old_states(self):
    planet = _planet()
    text = planet.genome_text('black')
    self.assertIn('tolerance 2.0, vigor 2.0, hardiness 2.0', text)
    self.assertIn('6 of 12 levels', text)
    state = planet.daisies.get_state()
    for traits in state['traits'].values():
      for name in ('tolerance', 'vigor', 'hardiness'):
        del traits[name]
    planet.daisies.set_state(state)
    self.assertEqual(planet.daisies.traits('black').levels(), 0.0)


class PlanetEvolutionTest(absltest.TestCase):

  def test_adaptation_phases_in_and_reports(self):
    planet = _planet()
    message = planet.queue_adaptation('white', 'warmer optimum, please')
    self.assertIn('begin evolving a warmer optimum', message)
    self.assertIn('Currently evolving: warmer', planet.genome_text('white'))

    def evolution_news(report):  # (Biome news is not about evolution.)
      return [e for e in report.events if not TALLY.search(e.text)]

    first = planet.advance()
    self.assertAlmostEqual(planet.daisies.traits('white').optimum_c, 23.25)
    self.assertEqual(evolution_news(first), [])
    second = planet.advance()
    self.assertAlmostEqual(planet.daisies.traits('white').optimum_c, 24.0)
    news = evolution_news(second)
    self.assertIn('have evolved a warmer optimum', news[0].text)
    self.assertEqual(news[0].topic, planet_lib.BIO_EVENTS)
    self.assertEqual(planet.adaptation_counts['white'], 1)
    self.assertEqual(planet.history[-1]['adaptations_white'], 1)

  def test_no_change_answers_do_nothing(self):
    planet = _planet()
    self.assertIn('stay as they are', planet.queue_adaptation('black', 'meh'))
    self.assertEqual(planet.pending_adaptations, [])

  def test_behaviour_changes_growth(self):
    spread, dormant = _planet(), _planet()
    spread.set_behaviour('white', 'spread')
    dormant.set_behaviour('white', 'dormant')
    for _ in range(3):
      spread.advance()
      dormant.advance()
    self.assertGreater(
        spread.daisies.cover('white'), dormant.daisies.cover('white')
    )
    self.assertEqual(spread.history[-1]['behaviour_white'], 'spread')

  def test_checkpoint_keeps_pending_adaptations(self):
    planet = _planet()
    planet.queue_adaptation('black', 'darker petals')
    planet.set_behaviour('black', 'dormant')
    planet.advance()
    resumed = _planet()
    resumed.set_state(planet.get_state())
    planet.advance()
    resumed.advance()
    self.assertEqual(planet.history, resumed.history)
    self.assertEqual(resumed.daisies.behaviour('black'), (0.4, 0.5))


if __name__ == '__main__':
  absltest.main()

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

"""Tests for symbiosis pacts."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time.world import data
from examples.deep_time.world import symbiosis
import numpy as np

NAMES = {'white': 'White daisies', 'black': 'Black daisies'}


def _rules() -> symbiosis.Symbiosis:
  return symbiosis.Symbiosis(data.load('symbiosis.yaml'))


class SymbiosisTest(parameterized.TestCase):

  def test_propose_answer_expire(self):
    rules = _rules()
    proposal = rules.propose('white', 'black', 3, NAMES)
    assert proposal is not None
    self.assertIn('White daisies propose a mixed-meadows pact', proposal.text)
    self.assertIsNone(rules.propose('white', 'black', 4, NAMES))
    self.assertEqual(rules.pending_for('black'), proposal)
    self.assertEqual(rules.answer('black', True, 5), proposal)
    self.assertTrue(rules.has_pact('black', 'white'))
    self.assertEqual(rules.partners('white'), ['black'])
    self.assertIsNone(rules.propose('black', 'white', 6, NAMES))
    self.assertEqual(rules.expire(14), [])
    self.assertEqual(rules.expire(15), ['black+white'])
    self.assertFalse(rules.has_pact('white', 'black'))
    self.assertEqual((rules.formed, rules.refused), (1, 0))

  def test_refusal_and_unknown_answer(self):
    rules = _rules()
    rules.propose('white', 'black', 1, NAMES)
    rules.answer('black', False, 2)
    self.assertFalse(rules.has_pact('white', 'black'))
    self.assertEqual(rules.refused, 1)
    self.assertIsNone(rules.answer('black', True, 3))

  @parameterized.parameters(
      ('Accept: together we are stronger.', True),
      ('We decline; we would rather accept nothing.', False),
      ('I am not sure.', None),
  )
  def test_parse_answer(self, text, expected):
    self.assertEqual(symbiosis.parse_answer(text), expected)

  def test_multipliers_reward_shared_land(self):
    context = game_lib.new_game('daisyworld')
    daisies = context.planet.daisies
    rules = _rules()
    self.assertEqual(rules.multipliers(daisies), {})
    rules.propose('white', 'black', 0, NAMES)
    rules.answer('black', True, 0)
    boosts = rules.multipliers(daisies)
    self.assertTrue(np.all(boosts['white'] > 1.0))
    self.assertTrue(np.all(boosts['white'] <= 1.0 + rules.bonus))
    self.assertEqual(set(boosts), {'white', 'black'})

  def test_state_round_trip(self):
    rules = _rules()
    rules.propose('white', 'black', 1, NAMES)
    restored = _rules()
    restored.set_state(rules.get_state())
    self.assertEqual(restored.pending_for('black'), rules.pending_for('black'))


class PactFlowTest(absltest.TestCase):

  def test_species_offer_pacts_and_partners_settle_them(self):
    context = game_lib.new_game('daisyworld')
    planet = context.planet
    white = actors_lib.SpeciesActor('White daisies', 'white', planet)
    option = 'propose a pact to the Black daisies'
    self.assertIn(option, white.options())
    text = white.resolve(option)
    self.assertIn('mixed-meadows pact', text)
    self.assertEqual(white.drain_outbox(), [('diplomacy.black.white', text)])
    self.assertNotIn(option, white.options())
    # The partner answers on its own menu (its mind's counsel is context).
    black = actors_lib.SpeciesActor('Black daisies', 'black', planet)
    accept = 'accept the pact offered by the White daisies'
    self.assertEqual(black.options()[:1], [accept])
    # The mind's reply alone settles nothing: answers are actions.
    context.bus.publish(
        'bio.species.black.pact', 'Reply to White daisies: accept, gladly.'
    )
    assert planet.symbiosis is not None
    self.assertFalse(planet.symbiosis.has_pact('white', 'black'))
    news = black.resolve(accept)
    self.assertIn('accept the pact', news)
    self.assertIn(('diplomacy.white.black', news), black.drain_outbox())
    assert planet.symbiosis is not None
    self.assertTrue(planet.symbiosis.has_pact('white', 'black'))
    planet.advance()
    self.assertIn('white', planet.daisies.get_state()['modifiers']['mutualism'])


if __name__ == '__main__':
  absltest.main()

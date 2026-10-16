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

"""Tests for plans: balanced production chains (the ratio planner)."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model

MODERN = [
    'coal_mining',
    'steam_power',
    'steelmaking',
    'metallurgy',
    'electricity',
    'glassmaking',
    'photovoltaics',
]


def _game():
  context = game_lib.new_game('great_acceleration')
  civ = context.planet.civ
  assert civ is not None
  civ.polities['basalt'].techs += MODERN
  return context, context.planet, civ


class PlansTest(absltest.TestCase):

  def test_plan_balances_the_chain_within_its_share(self):
    _, _, civ = _game()
    shares, external = civ.plan('basalt', 'panel_factory')
    self.assertAlmostEqual(sum(shares.values()), civ.plan_share)
    for key in ('panel_factory', 'glassworks', 'wire_mill', 'steelworks'):
      self.assertGreater(shares[key], 0.0)
    # Glass and panels run one for one (one glass per panel, both one labor).
    self.assertAlmostEqual(shares['glassworks'], shares['panel_factory'])
    self.assertNotIn('solar_power', shares)
    self.assertIsInstance(external, dict)

  def test_solar_powers_the_chain_once_panels_stand(self):
    _, _, civ = _game()
    civ.ledger.deposit('basalt', 'panels', 50.0, 'test')
    shares, _ = civ.plan('basalt', 'panel_factory')
    self.assertIn('solar_power', shares)
    self.assertNotIn('coal_plant', shares)

  def test_people_adopt_a_plan_in_one_decision(self):
    _, planet, civ = _game()
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    option = 'plan a balanced chain to build solar panels'
    self.assertIn(option, basalt.options())
    self.assertLessEqual(len(basalt.options()), actors_lib.MAX_OPTIONS)
    text = basalt.resolve(option)
    self.assertStartsWith(text, 'Basalt clans plan a balanced chain to build')
    allocation = civ.polities['basalt'].allocation
    self.assertAlmostEqual(sum(allocation.values()), 1.0)
    self.assertGreater(allocation['panel_factory'], 0.0)
    self.assertNotIn(option, basalt.options())  # Already planned.
    # Without mathematics there is no planning.
    civ.polities['amber'].techs.remove('mathematics')
    self.assertEqual(civ.plannable('amber'), [])

  def test_newest_targets_stay_on_the_menu(self):
    _, planet, civ = _game()
    civ.polities['basalt'].techs += ['chemistry', 'rocketry']
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    plans = [o for o in basalt.options() if o.startswith('plan')]
    self.assertLen(plans, actors_lib.MAX_PLANS)
    self.assertIn('plan a balanced chain to build rockets', plans)
    civ.polities['basalt'].techs.remove('photovoltaics')
    civ.polities['basalt'].techs.append('photovoltaics')  # Learned last.
    self.assertIn(
        'plan a balanced chain to build solar panels', basalt.options()
    )

  def test_plan_survives_a_checkpoint(self):
    context, planet, civ = _game()
    planet.advance()
    shares = civ.plan('basalt', 'panel_factory')
    state = context.get_state()
    other, _, other_civ = _game()
    other.set_state(state)
    self.assertEqual(other_civ.plan('basalt', 'panel_factory'), shares)

  def test_mock_plans_the_solar_chain_when_coal_is_bound(self):
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. '
    )
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'plan a balanced chain to build solar panels',
        'c': actors_lib.KEEP_LABOR,
    }
    policy = mock_language_model.polity_policy
    self.assertNotEqual(policy(sheet, options), options['b'])
    bound = sheet + 'We are bound by coal treaties with the Amber folk.'
    self.assertEqual(policy(bound, options), options['b'])


if __name__ == '__main__':
  absltest.main()

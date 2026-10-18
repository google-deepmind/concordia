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

"""Tests for standing orders as reflexes."""

import json

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.components import policy_compiler


def _context():
  context = game_lib.new_game('fire_and_iron')
  assert context.automation is not None
  assert context.planet.civ is not None and context.planet.trade is not None
  return context, context.automation, context.planet.civ, context.planet.trade


class AutomationTest(absltest.TestCase):

  def test_reflexes_need_research(self):
    _, automation, civ, _ = _context()
    self.assertEqual(automation.allowed_ops('basalt'), {'json_field'})
    self.assertEqual(automation.offers('basalt'), {})
    civ.polities['basalt'].techs = ['granaries']
    self.assertIn('threshold', automation.allowed_ops('basalt'))

  def test_writing_makes_reflexes_count_goods_on_the_road(self):
    _, automation, civ, _ = _context()
    civ.polities['basalt'].techs = ['granaries']
    path = automation._spec('basalt', 'bid', 'ore')['pipeline'][0]['path']  # pylint: disable=protected-access
    self.assertEqual(path, 'polities.basalt.stocks.ore')
    civ.polities['basalt'].techs = ['granaries', 'writing']
    path = automation._spec('basalt', 'bid', 'ore')['pipeline'][0]['path']  # pylint: disable=protected-access
    self.assertEqual(path, 'polities.basalt.position.ore')

  def test_only_relevant_orders_are_offered(self):
    _, automation, civ, _ = _context()
    civ.polities['basalt'].techs = ['granaries']
    civ.ledger.deposit('basalt', 'ore', 5.0, 'test')
    offers = automation.offers('basalt')
    self.assertIn(
        'set a standing order: sell ore whenever we hold more than 6', offers
    )
    self.assertIn(
        'set a standing order: buy charcoal whenever we hold less than 2',
        offers,
    )
    self.assertNotIn(
        'set a standing order: sell grain whenever we hold more than 14',
        offers,
    )

  def test_adopted_reflex_places_orders_without_turns(self):
    context, automation, civ, trade = _context()
    civ.polities['basalt'].techs = ['granaries']
    civ.ledger.deposit('basalt', 'ore', 9.0, 'test')
    actor = actors_lib.PolityActor(
        'Basalt clans', 'basalt', context.planet, 0.15, automation
    )
    option = 'set a standing order: sell ore whenever we hold more than 6'
    self.assertIn(option, actor.options())
    self.assertEqual(
        actor.resolve(option), 'Basalt clans set a standing order to sell ore.'
    )
    self.assertNotIn(option, actor.options())
    context.publish_state()  # The census shows 9 ore: the reflex fires.
    self.assertIn(('ore', 'ask', 3), trade.intents['basalt'])
    self.assertEqual(automation.reflex_orders, 1)
    self.assertIn('sell ore (placed 1 times)', automation.describe('basalt'))
    actor.resolve('ask to buy iron')
    self.assertAlmostEqual(automation.ratio(), 0.5)
    cancel = 'cancel the standing order for selling ore'
    self.assertIn(cancel, actor.options())
    actor.resolve(cancel)
    self.assertEqual(automation.engine.names('basalt'), [])

  def test_compile_installs_only_safe_orders(self):
    _, automation, civ, _ = _context()
    civ.polities['basalt'].techs = ['granaries']

    def order(name, emit='civ.basalt.orders', template='ask ore', **extra):
      return {
          'name': name,
          'input': extra.get('input', 'civ.census'),
          'pipeline': [
              {'op': 'json_field', 'path': 'polities.basalt.stocks.ore'},
              extra.get('stage', {'op': 'threshold', 'above': 6}),
          ],
          'emit': {'topic': emit, 'template': template},
      }

    text = 'Our orders: ' + json.dumps([
        order('ok'),
        order('steal', emit='civ.amber.orders'),
        order('spy', input='civ.amber.plan'),
        order('gold', template='ask gold'),
        order('smooth', stage={'op': 'moving_average', 'n': 3}),
        order('again'),  # Same order as 'ok'.
        order('buy', template='bid charcoal'),
    ])
    installed = automation.compile('basalt', text)
    self.assertEqual(installed, ['basalt:policy:ok', 'basalt:policy:buy'])
    self.assertEqual(automation.compile('basalt', 'nonsense'), [])
    self.assertEqual(automation.compile('basalt', text), [])

  def test_policy_compiler_costs_nothing_until_reflexes_unlock(self):
    context, automation, civ, _ = _context()
    model = mock_language_model.DeepTimeMockModel()
    compiler = policy_compiler.PolicyCompiler(model, context, 'cedar')
    self.assertEqual(compiler._make_pre_act_value(), '')  # pylint: disable=protected-access
    self.assertEqual(model.stats()['text_calls'], 0)
    civ.polities['cedar'].techs = ['granaries']
    answer = mock_language_model.policy_answer(
        'You are the scribes of Cedar league. Our land is rich in forest,'
        ' poor in ore. ... "topic": "civ.cedar.orders" ...'
    )
    self.assertEqual(
        automation.compile('cedar', answer),
        ['cedar:policy:sell_charcoal', 'cedar:policy:buy_ore'],
    )

  def test_a_draft_is_counsel_until_the_people_adopts_it(self):
    context, automation, civ, _ = _context()
    civ.polities['cedar'].techs = ['granaries']
    draft = mock_language_model.policy_answer(
        'You are the scribes of Cedar league. Our land is rich in forest,'
        ' poor in ore. ... "topic": "civ.cedar.orders" ...'
    )
    context.bus.publish('civ.cedar.policy', draft, source='Cedar league')
    # Published by the scribes' mind: nothing installed, an option offered.
    self.assertEqual(automation.engine.names('cedar'), [])
    self.assertEqual(automation.draft('cedar'), draft)
    self.assertIn('sell charcoal', automation.summary('cedar', draft))
    cedar = actors_lib.PolityActor(
        'Cedar league', 'cedar', context.planet, 0.15, automation=automation
    )
    self.assertIn(actors_lib.ADOPT_ORDERS, cedar.options())
    news = cedar.resolve(actors_lib.ADOPT_ORDERS)
    self.assertIn("adopt their scribes' 2 standing orders", news)
    self.assertLen(automation.engine.names('cedar'), 2)
    # Adopted: no longer offered, and kept in the state.
    self.assertIsNone(automation.draft('cedar'))
    self.assertNotIn(actors_lib.ADOPT_ORDERS, cedar.options())
    self.assertIn('cedar', automation.get_state()['adopted'])

  def test_state_round_trip(self):
    context, automation, civ, _ = _context()
    civ.polities['cedar'].techs = ['granaries']
    automation.adopt('cedar', 'ask', 'charcoal')
    state = context.get_state()
    other, other_automation, other_civ, _ = _context()
    other_civ.polities['cedar'].techs = ['granaries']
    other.set_state(state)
    self.assertEqual(other_automation.engine.names(), ['cedar:ask:charcoal'])


if __name__ == '__main__':
  absltest.main()

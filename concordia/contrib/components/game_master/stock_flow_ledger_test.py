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

"""Tests for the stock-flow ledger."""

import json
import threading

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.contrib.components.game_master import stock_flow_ledger as sfl
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

BLOOMERY = sfl.Recipe(
    'bloomery',
    inputs={'ore': 2.0, 'charcoal': 1.0},
    outputs={'iron': 1.0, 'slag': 1.0},
)


class _EchoAct(entity_component.ActingComponent):

  def get_action_attempt(self, contexts, action_spec):
    del action_spec
    return contexts.get('__ledger__', '')

  def get_state(self):
    return {}

  def set_state(self, state):
    del state


class LedgerTest(absltest.TestCase):

  def test_deposit_withdraw_transfer(self):
    ledger = sfl.Ledger()
    ledger.deposit('Aurelia', 'grain', 10, reason='harvest')
    self.assertEqual(ledger.transfer('Aurelia', 'Borea', 'grain', 4), 4)
    self.assertEqual(ledger.withdraw('Borea', 'grain', 1, reason='eaten'), 1)
    self.assertEqual(ledger.stock('Aurelia', 'grain'), 6)
    self.assertEqual(ledger.stock('Borea', 'grain'), 3)
    self.assertEqual(ledger.total('grain'), 9)
    self.assertTrue(ledger.is_conserved())
    kinds = [f.kind for f in ledger.flows()]
    self.assertEqual(kinds, ['deposit', 'transfer', 'withdraw'])
    self.assertEqual(ledger.flows()[0].reason, 'harvest')

  def test_insufficient_stock_and_partial(self):
    ledger = sfl.Ledger()
    ledger.deposit('A', 'x', 2)
    with self.assertRaises(sfl.InsufficientStockError):
      ledger.withdraw('A', 'x', 5)
    self.assertEqual(ledger.withdraw('A', 'x', 5, partial=True), 2)
    self.assertEqual(ledger.stock('A', 'x'), 0)
    with self.assertRaises(ValueError):
      ledger.deposit('A', 'x', -1)

  def test_negative_allowed(self):
    ledger = sfl.Ledger(allow_negative=True)
    ledger.withdraw('A', 'debt', 3)
    self.assertEqual(ledger.stock('A', 'debt'), -3)
    self.assertTrue(ledger.is_conserved())

  def test_recipe_conversion_and_statistics(self):
    ledger = sfl.Ledger()
    ledger.deposit('A', 'ore', 7)
    ledger.deposit('A', 'charcoal', 2)
    self.assertEqual(ledger.max_batches('A', BLOOMERY), 2)
    with self.assertRaises(sfl.InsufficientStockError):
      ledger.apply_recipe('A', BLOOMERY, batches=3)
    self.assertEqual(ledger.apply_recipe('A', BLOOMERY, 3, partial=True), 2)
    self.assertEqual(ledger.holdings('A'), {'ore': 3, 'iron': 2, 'slag': 2})
    stats = ledger.production()
    self.assertEqual(stats['iron']['produced'], 2)
    self.assertEqual(stats['ore']['consumed'], 4)
    self.assertEqual(stats['ore']['produced'], 7)
    self.assertTrue(ledger.is_conserved())
    self.assertEqual(
        ledger.max_batches('A', sfl.Recipe('sun', {}, {'e': 1})), float('inf')
    )

  def test_limiting_input_is_the_bottleneck(self):
    ledger = sfl.Ledger()
    ledger.deposit('A', 'ore', 7)
    ledger.deposit('A', 'charcoal', 2)
    # Two batches' worth of charcoal, three and a half of ore.
    self.assertEqual(ledger.limiting_input('A', BLOOMERY), 'charcoal')
    ledger.deposit('A', 'charcoal', 10)
    self.assertEqual(ledger.limiting_input('A', BLOOMERY), 'ore')
    # Nothing in stock: the first input in the recipe's order.
    self.assertEqual(ledger.limiting_input('B', BLOOMERY), 'ore')
    self.assertIsNone(
        ledger.limiting_input('A', sfl.Recipe('sun', {}, {'e': 1}))
    )

  def test_links_capacity_and_latency(self):
    ledger = sfl.Ledger()
    ledger.add_link(sfl.Link('river', 'A', 'B', 'iron', capacity=5, latency=2))
    ledger.deposit('A', 'iron', 8)
    self.assertEqual(ledger.ship('river', 6), 5)
    self.assertEqual(ledger.ship('river', 1), 0)
    self.assertEqual(ledger.in_transit('iron'), 5)
    self.assertEqual(ledger.total('iron'), 8)
    self.assertEqual(ledger.total('iron', include_transit=False), 3)
    self.assertTrue(ledger.is_conserved())
    self.assertEqual(ledger.advance(), [])
    self.assertEqual(ledger.ship('river', 3), 3)
    deliveries = ledger.advance()
    self.assertEqual([(f.kind, f.amount) for f in deliveries], [('deliver', 5)])
    self.assertEqual(ledger.stock('B', 'iron'), 5)
    ledger.advance()
    self.assertEqual(ledger.stock('B', 'iron'), 8)
    self.assertEqual(ledger.in_transit(), 0)
    with self.assertRaises(ValueError):
      ledger.add_link(sfl.Link('river', 'A', 'B', 'iron'))

  def test_zero_latency_link_delivers_immediately(self):
    ledger = sfl.Ledger()
    ledger.add_link(sfl.Link('pipe', 'A', 'B', 'water'))
    ledger.deposit('A', 'water', 2)
    ledger.ship('pipe', 2)
    self.assertEqual(ledger.stock('B', 'water'), 2)

  def test_statistics_are_per_tick(self):
    ledger = sfl.Ledger()
    ledger.deposit('A', 'x', 1)
    ledger.advance()
    ledger.deposit('A', 'x', 2)
    self.assertEqual(ledger.production(0)['x']['produced'], 1)
    self.assertEqual(ledger.production()['x']['produced'], 2)
    ledger.forget_statistics_before(1)
    self.assertEqual(ledger.production(0), {})

  def test_audit_limit(self):
    ledger = sfl.Ledger(audit_limit=3)
    for _ in range(5):
      ledger.deposit('A', 'x', 1)
    self.assertLen(ledger.flows(), 3)
    self.assertEqual(ledger.total('x'), 5)
    self.assertTrue(ledger.is_conserved())

  def test_state_round_trip_is_json_safe(self):
    ledger = sfl.Ledger()
    ledger.add_link(sfl.Link('road', 'A', 'B', 'ore', capacity=3, latency=1))
    ledger.deposit('A', 'ore', 5)
    ledger.ship('road', 3)
    state = json.loads(json.dumps(ledger.get_state()))
    restored = sfl.Ledger()
    restored.set_state(state)
    self.assertEqual(restored.in_transit('ore'), 3)
    self.assertEqual(restored.remaining_capacity('road'), 0)
    restored.advance()
    self.assertEqual(restored.stock('B', 'ore'), 3)
    self.assertTrue(restored.is_conserved())
    self.assertEqual(restored.production(0)['ore']['shipped'], 3)

  def test_concurrent_transfers_conserve(self):
    ledger = sfl.Ledger()
    ledger.deposit('A', 'x', 1000)

    def worker():
      for _ in range(100):
        ledger.transfer('A', 'B', 'x', 1)
        ledger.transfer('B', 'A', 'x', 0.5)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    self.assertAlmostEqual(ledger.stock('B', 'x'), 200)
    self.assertTrue(ledger.is_conserved())


class BalanceChainTest(absltest.TestCase):

  def test_iron_chain_works_back_to_raw_labor(self):
    recipes = [
        sfl.Recipe('woodcut', {}, {'wood': 2.0}),
        sfl.Recipe('kiln', {'wood': 2.0}, {'charcoal': 1.0}),
        sfl.Recipe('mine', {}, {'ore': 1.0}),
        sfl.Recipe(
            'bloomery',
            {'ore': 2.0, 'charcoal': 1.0},
            {'iron': 1.0, 'slag': 1.0},
        ),
        sfl.Recipe('smithy', {'iron': 1.0}, {'tools': 1.0}),
    ]
    plan = sfl.balance_chain(recipes, 'smithy', batches=3.0)
    self.assertEqual(
        dict(plan.batches),
        {
            'woodcut': 3.0,
            'kiln': 3.0,
            'mine': 6.0,
            'bloomery': 3.0,
            'smithy': 3.0,
        },
    )
    self.assertEqual(dict(plan.external), {})

  def test_shared_inputs_add_up_and_raw_inputs_are_external(self):
    recipes = [
        sfl.Recipe('power', {'coal': 1.0}, {'energy': 2.0}),
        sfl.Recipe('solar', {}, {'energy': 5.0}),
        sfl.Recipe('glass', {'sand': 2.0, 'energy': 1.0}, {'glass': 1.0}),
        sfl.Recipe('wire', {'copper': 1.0, 'energy': 1.0}, {'wire': 2.0}),
        sfl.Recipe('panels', {'glass': 1.0, 'wire': 1.0}, {'panels': 1.0}),
    ]
    plan = sfl.balance_chain(recipes, 'panels', batches=2.0)
    self.assertEqual(plan.batches['glass'], 2.0)
    self.assertEqual(plan.batches['wire'], 1.0)
    self.assertEqual(plan.batches['power'], 1.5)  # Energy 2 + 1, at 2 each.
    self.assertNotIn('solar', plan.batches)
    self.assertEqual(
        dict(plan.external), {'coal': 1.5, 'copper': 1.0, 'sand': 4.0}
    )
    solar = sfl.balance_chain(
        recipes, 'panels', batches=2.0, producers={'energy': 'solar'}
    )
    self.assertAlmostEqual(solar.batches['solar'], 0.6)
    self.assertNotIn('coal', solar.external)

  def test_producers_that_make_nothing_count_as_absent(self):
    recipes = [
        sfl.Recipe('dry_mine', {}, {'ore': 0.0}),
        sfl.Recipe('smelter', {'ore': 1.0}, {'iron': 1.0}),
    ]
    plan = sfl.balance_chain(recipes, 'smelter')
    self.assertEqual(dict(plan.batches), {'smelter': 1.0})
    self.assertEqual(dict(plan.external), {'ore': 1.0})

  def test_co_products_run_at_the_most_demanded_rate(self):
    recipes = [
        sfl.Recipe('refinery', {}, {'fuel': 1.0, 'gas': 1.0}),
        sfl.Recipe('engine', {'fuel': 1.0, 'gas': 0.5}, {'work': 1.0}),
    ]
    plan = sfl.balance_chain(recipes, 'engine', batches=2.0)
    self.assertEqual(plan.batches['refinery'], 2.0)  # Not 2 + 1.

  def test_net_output_of_self_consuming_producers(self):
    recipes = [
        sfl.Recipe('breeder', {'seed': 1.0}, {'seed': 3.0}),
        sfl.Recipe('mill', {'seed': 4.0}, {'flour': 1.0}),
    ]
    plan = sfl.balance_chain(recipes, 'mill', producers={'seed': 'breeder'})
    self.assertEqual(plan.batches['breeder'], 2.0)  # Net 2 seeds a batch.
    self.assertEqual(dict(plan.external), {})

  def test_readme_example(self):
    recipes = [
        sfl.Recipe('mine', {}, {'ore': 1.0}),
        sfl.Recipe('kiln', {'wood': 2.0}, {'charcoal': 1.0}),
        sfl.Recipe('bloomery', {'ore': 2.0, 'charcoal': 1.0}, {'iron': 1.0}),
    ]
    plan = sfl.balance_chain(recipes, 'bloomery', batches=3.0)
    self.assertEqual(
        dict(plan.batches), {'bloomery': 3.0, 'mine': 6.0, 'kiln': 3.0}
    )
    self.assertEqual(dict(plan.external), {'wood': 6.0})

  def test_circular_chains_and_unknown_targets_raise(self):
    recipes = [
        sfl.Recipe('a', {'b': 1.0}, {'a': 1.0}),
        sfl.Recipe('b', {'a': 1.0}, {'b': 1.0}),
    ]
    with self.assertRaisesRegex(ValueError, 'circular'):
      sfl.balance_chain(recipes, 'a')
    with self.assertRaises(KeyError):
      sfl.balance_chain(recipes, 'c')
    with self.assertRaisesRegex(ValueError, 'unique'):
      sfl.balance_chain(recipes + [recipes[0]], 'a')


class ComponentTest(absltest.TestCase):

  def test_component_shows_holdings_and_checkpoints(self):
    component = sfl.StockFlowLedger(holders=['A'], resources=['iron'])
    component.ledger.deposit('A', 'iron', 2.5)
    component.ledger.deposit('A', 'slag', 1)
    component.ledger.deposit('B', 'iron', 1)
    agent = entity_agent.EntityAgent(
        agent_name='GM',
        act_component=_EchoAct(),
        context_components={sfl.DEFAULT_LEDGER_COMPONENT_KEY: component},
    )
    self.assertIn(
        'A: iron=2.5',
        agent.act(entity_lib.free_action_spec(call_to_action='Books?')),
    )
    restored = sfl.StockFlowLedger()
    restored.set_state(component.get_state())
    self.assertEqual(restored.ledger.stock('B', 'iron'), 1)
    self.assertEqual(
        sfl.format_holdings(restored.ledger), 'A: iron=2.5, slag=1\nB: iron=1'
    )


if __name__ == '__main__':
  absltest.main()

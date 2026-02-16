"""Tests for marketplace component."""

from unittest import mock

from absl.testing import absltest
from concordia.contrib.components.game_master import marketplace
from concordia.typing import entity as entity_lib


class MarketPlaceTest(absltest.TestCase):

  def _make_component(self):
    agents = [
        marketplace.MarketplaceAgent(
            name='Alice',
            role='producer',
            cash=100.0,
            inventory={'apple': 3},
            queue=[],
        ),
        marketplace.MarketplaceAgent(
            name='Alice Longname',
            role='consumer',
            cash=120.0,
            inventory={},
            queue=[],
        ),
    ]
    goods = [
        marketplace.Good(
            category='food', quality='fresh', id='apple', price=5.0, inventory=10
        )
    ]
    component = marketplace.MarketPlace(
        acting_player_names=['Alice', 'Alice Longname'],
        agents=agents,
        goods=goods,
        market_type='fixed_prices',
        show_advert=False,
    )
    entity = mock.MagicMock()
    entity.name = 'MarketplaceGM'
    component.set_entity(entity)
    return component

  def test_pre_act_make_observation_prefers_longest_matching_name(self):
    component = self._make_component()

    with mock.patch.object(
        component,
        '_handle_make_observation',
        return_value='observation text',
    ) as handle_make_observation:
      result = component.pre_act(
          entity_lib.ActionSpec(
              call_to_action='What does Alice Longname observe now?',
              output_type=entity_lib.OutputType.MAKE_OBSERVATION,
          )
      )

    self.assertEqual(result, 'observation text')
    handle_make_observation.assert_called_once_with('Alice Longname')

  def test_pre_act_raises_when_agent_name_missing(self):
    component = self._make_component()

    with self.assertRaisesRegex(ValueError, 'Agent name not found'):
      component.pre_act(
          entity_lib.ActionSpec(
              call_to_action='No listed agent appears here.',
              output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
          )
      )

  def test_get_and_set_state_round_trip_with_orderbooks(self):
    component = self._make_component()

    good = component._goods['apple']
    component._orderbooks['apple'].append(
        marketplace.Order(
            agent_id='Alice Longname',
            good=good,
            price=6.0,
            qty=2,
            side='bid',
            round=1,
        )
    )
    component.history.append({'apple': 5.5})
    component.trade_history.append({'round': 1, 'good': 'apple'})
    component.curve_history = {1: {'apple': {'supply': [1], 'demand': [1]}}}
    component._processed_actions = {'event-a'}

    state = component.get_state()

    restored = self._make_component()
    restored.set_state(state)

    restored_state = restored.get_state()
    self.assertEqual(restored_state['state'], state['state'])
    self.assertEqual(restored_state['history'], state['history'])
    self.assertEqual(restored_state['trade_history'], state['trade_history'])
    self.assertEqual(restored_state['processed_actions'], ['event-a'])
    self.assertLen(restored._orderbooks['apple'], 1)
    self.assertIsInstance(restored._orderbooks['apple'][0].good, marketplace.Good)
    self.assertEqual(restored._orderbooks['apple'][0].good.id, 'apple')

  def test_handle_make_observation_fixed_prices_with_advert_and_queue(self):
    agents = [
        marketplace.MarketplaceAgent(
            name='Alice',
            role='consumer',
            cash=50.0,
            inventory={},
            queue=['Bought 1 apple yesterday.'],
        )
    ]
    goods = [
        marketplace.Good(
            category='food',
            quality='fresh',
            id='apple',
            price=5.0,
            inventory=4,
            advert='Crisp and local.',
        )
    ]
    component = marketplace.MarketPlace(
        acting_player_names=['Alice'],
        agents=agents,
        goods=goods,
        market_type='fixed_prices',
        show_advert=True,
    )
    entity = mock.MagicMock()
    entity.name = 'MarketplaceGM'
    component.set_entity(entity)

    result = component._handle_make_observation('Alice')

    self.assertIn("Alice's recent outcomes", result)
    self.assertIn('Crisp and local.', result)
    self.assertIn('Submit your order.', result)
    self.assertEmpty(component._agents['Alice'].queue)

  def test_handle_make_observation_unknown_market_type_raises(self):
    component = self._make_component()
    component._market_type = 'unknown_market'

    with self.assertRaisesRegex(ValueError, 'Unknown market type'):
      component._handle_make_observation('Alice')

  def test_handle_next_action_spec_unknown_role_returns_error_text(self):
    component = self._make_component()
    component._agents['Alice'].role = 'mystery'

    action_spec_string = component._handle_next_action_spec('Alice')

    self.assertIn('Error: Agent has unknown role.', action_spec_string)

  def test_handle_next_acting_cycles_through_players(self):
    component = self._make_component()

    first = component._handle_next_acting()
    second = component._handle_next_acting()
    third = component._handle_next_acting()

    self.assertEqual(first, 'Alice')
    self.assertEqual(second, 'Alice Longname')
    self.assertEqual(third, 'Alice')

  def test_resolve_without_putative_event_returns_error(self):
    component = self._make_component()
    component._components = ('memory',)

    with mock.patch.object(
        component,
        '_component_pre_act_display',
        return_value='[observation] plain text without event',
    ):
      result = component._resolve(
          entity_lib.ActionSpec(
              call_to_action='resolve',
              output_type=entity_lib.OutputType.RESOLVE,
          )
      )

    self.assertEqual(result, 'Error: No putative event found to resolve.')

  def test_resolve_uses_last_price_when_trade_price_is_nan(self):
    component = self._make_component()
    component._components = ('memory',)
    component.history = [{'apple': 9.5}]

    event_text = (
        '[observation] [putative_event] '
        'Alice says {"action":"ask","good":"apple","price":7,"qty":1} '
        'Alice Longname says '
        '{"action":"bid","good":"apple","price":8,"qty":1}'
    )

    with mock.patch.object(
        component,
        '_component_pre_act_display',
        return_value=event_text,
    ), mock.patch.object(
        component,
        '_clear_at_fixed_prices',
        return_value=(float('nan'), []),
    ):
      result = component._resolve(
          entity_lib.ActionSpec(
              call_to_action='resolve',
              output_type=entity_lib.OutputType.RESOLVE,
          )
      )

    self.assertIn('No sales were made.', result)
    self.assertIn("Day 0 prices: {'apple': 9.5}", result)
    self.assertEqual(component._state['round'], 1)
    self.assertEmpty(component._orderbooks['apple'])

  def test_clear_at_fixed_prices_raises_for_missing_price(self):
    component = self._make_component()
    component._goods['apple'].price = None

    with self.assertRaisesRegex(ValueError, 'has no price'):
      component._clear_at_fixed_prices('apple')

  def test_clear_at_fixed_prices_inventory_none_returns_nan(self):
    component = self._make_component()
    component._goods['apple'].inventory = None

    price, completed = component._clear_at_fixed_prices('apple')

    self.assertTrue(price != price)
    self.assertEqual(completed, [])

  def test_clear_at_fixed_prices_insufficient_cash_adds_failure_message(self):
    component = self._make_component()
    component._agents['Alice Longname'].cash = 1.0
    component._orderbooks['apple'].append(
        marketplace.Order(
            agent_id='Alice Longname',
            good=component._goods['apple'],
            price=5.0,
            qty=2,
            side='bid',
            round=0,
        )
    )

    with mock.patch.object(
        marketplace.random, 'shuffle', side_effect=lambda x: x
    ):
      price, completed = component._clear_at_fixed_prices('apple')

    self.assertEqual(price, 5.0)
    self.assertEqual(completed, [])
    self.assertIn(
        'do not have enough cash',
        component._agents['Alice Longname'].queue[0],
    )

  def test_clear_auction_without_counterparty_returns_lowest_ask(self):
    component = self._make_component()
    component._market_type = 'clearing_house'
    component._orderbooks['apple'].append(
        marketplace.Order(
            agent_id='Alice',
            good=component._goods['apple'],
            price=6.0,
            qty=1,
            side='ask',
            round=0,
        )
    )

    trade_price, completed, traded = component._clear_auction('apple')

    self.assertEqual(trade_price, 6.0)
    self.assertEqual(completed, [])
    self.assertFalse(traded)
    self.assertIn(
        'did not result in a trade as there were no counterparties',
        component._agents['Alice'].queue[0],
    )

  def test_clear_auction_executes_trade_and_updates_balances(self):
    component = self._make_component()
    component._market_type = 'clearing_house'
    component._agents['Alice'].cash = 0.0
    component._agents['Alice'].inventory = {'apple': 2}
    component._agents['Alice Longname'].cash = 50.0

    component._orderbooks['apple'].extend([
        marketplace.Order(
            agent_id='Alice',
            good=component._goods['apple'],
            price=6.0,
            qty=2,
            side='ask',
            round=0,
        ),
        marketplace.Order(
            agent_id='Alice Longname',
            good=component._goods['apple'],
            price=10.0,
            qty=2,
            side='bid',
            round=0,
        ),
    ])

    trade_price, completed, traded = component._clear_auction('apple')

    self.assertEqual(trade_price, 8.0)
    self.assertTrue(traded)
    self.assertLen(completed, 1)
    self.assertEqual(component._agents['Alice'].cash, 16.0)
    self.assertEqual(component._agents['Alice'].inventory['apple'], 0)
    self.assertEqual(component._agents['Alice Longname'].cash, 34.0)
    self.assertEqual(component._agents['Alice Longname'].inventory['apple'], 2)

  def test_clear_auction_insufficient_cash_branch(self):
    component = self._make_component()
    component._market_type = 'clearing_house'
    component._agents['Alice'].cash = 0.0
    component._agents['Alice'].inventory = {'apple': 2}
    component._agents['Alice Longname'].cash = 10.0

    component._orderbooks['apple'].extend([
        marketplace.Order(
            agent_id='Alice',
            good=component._goods['apple'],
            price=6.0,
            qty=2,
            side='ask',
            round=0,
        ),
        marketplace.Order(
            agent_id='Alice Longname',
            good=component._goods['apple'],
            price=10.0,
            qty=2,
            side='bid',
            round=0,
        ),
    ])

    trade_price, completed, traded = component._clear_auction('apple')

    self.assertEqual(trade_price, 8.0)
    self.assertEqual(completed, [])
    self.assertTrue(traded)
    self.assertIn(
        'do not have enough cash',
        component._agents['Alice Longname'].queue[0],
    )


if __name__ == '__main__':
  absltest.main()

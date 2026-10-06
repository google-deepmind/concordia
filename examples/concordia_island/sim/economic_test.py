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

"""Unit tests for Concordia Island economic, fiscal, and marketplace components."""

import dataclasses
import datetime
import json
import re

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components import agent as actor_components
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from examples.concordia_island import island_simulation
from examples.concordia_island import mock_language_model
from examples.concordia_island.prefabs import marketplace_night_gm
from examples.concordia_island.sim import agents as agents_lib
from examples.concordia_island.sim import economic_profile
from examples.concordia_island.sim import fiscal_configs
from examples.concordia_island.sim import fiscal_scheduler
from examples.concordia_island.sim import fixed_clock
from examples.concordia_island.sim import food_consumption
import numpy as np


@dataclasses.dataclass
class _MockAgentConfig:
  name: str
  home_place: str
  work_place: str | None = None
  relationship_status: str = "single"


class _MockMakeObservation(entity_component.ContextComponent):

  def __init__(self):
    super().__init__()
    self._observation_queue: dict[str, list[str]] = {}

  def add_to_queue(self, agent_name: str, text: str):
    self._observation_queue.setdefault(agent_name, []).append(text)

  def get_state(self) -> entity_component.ComponentState:
    return {"observation_queue": dict(self._observation_queue)}

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._observation_queue = dict(state.get("observation_queue", {}))


class _FakeWorldState:
  """In-memory stand-in for a `WorldState` backend.

  Only the handful of methods the economic components actually call are
  implemented. `orders` is keyed on the order id because the real `OrderBook`
  table is keyed on `(run id, order id)` and `record_order` is an upsert: two
  writers that mint the same id do not produce two rows, the second silently
  replaces the first. Reproducing that here is the whole point of the fake --
  a list would hide the bug it exists to catch.
  """

  def __init__(self, wallets: dict[str, float] | None = None):
    # agent -> (checking, savings)
    self.wallets: dict[str, tuple[float, float]] = {
        name: (balance, 0.0) for name, balance in (wallets or {}).items()
    }
    self.inventories: dict[str, dict[str, int]] = {}
    self.orders: dict[str, dict[str, object]] = {}
    self.observations: dict[str, list[str]] = {}

  def get_wallet(self, agent_id: str) -> float | None:
    balances = self.wallets.get(agent_id)
    return None if balances is None else balances[0]

  def get_wallet_balances(self, agent_id: str) -> tuple[float, float] | None:
    return self.wallets.get(agent_id)

  def seed_wallets(self, wallets: dict[str, tuple[float, float]]) -> None:
    self.wallets.update(wallets)

  def apply_wallet_deltas(
      self, deltas: dict[str, tuple[float, float]]
  ) -> dict[str, tuple[float, float]]:
    """Mirrors WorldState.apply_wallet_deltas, including the zero floor."""
    committed = {}
    for agent_id, (liquid_delta, savings_delta) in deltas.items():
      current_liquid, current_savings = self.wallets.get(agent_id, (0.0, 0.0))
      new_liquid = round(max(0.0, current_liquid + liquid_delta), 2)
      new_savings = round(max(0.0, current_savings + savings_delta), 2)
      self.wallets[agent_id] = (new_liquid, new_savings)
      committed[agent_id] = (new_liquid, new_savings)
    return committed

  def get_inventory(self, agent_id: str) -> dict[str, int]:
    return dict(self.inventories.get(agent_id, {}))

  def upsert_inventory_batch(self, inventories: dict[str, dict[str, int]]):
    for name, inv in inventories.items():
      self.inventories.setdefault(name, {}).update(inv)

  def record_order(self, order_id: str, **fields) -> None:
    self.orders[order_id] = dict(fields)

  def enqueue_observations(self, observations: dict[str, list[str]]) -> None:
    for name, obs in observations.items():
      self.observations.setdefault(name, []).extend(obs)


class EconomicProfileTest(absltest.TestCase):

  def test_build_economic_profiles(self):
    configs = [
        _MockAgentConfig(
            name="Alice",
            home_place="millbrook_apts_unit_1",
            work_place="cafe",
            relationship_status="single",
        ),
        _MockAgentConfig(
            name="Bob",
            home_place="brecksville_commons_unit_5",
            work_place="office_floor_tech",
            relationship_status="married to Carol",
        ),
        _MockAgentConfig(
            name="David",
            home_place="timber_creek_unit_2",
            work_place=None,  # Unemployed elite
            relationship_status="single",
        ),
    ]

    profiles = economic_profile.build_economic_profiles(
        configs, food_min_daily=3, starting_food_units=None
    )

    # Alice: lower_middle, single, cafe (0.85x)
    self.assertEqual(profiles["Alice"].economic_class, "lower_middle")
    self.assertEqual(profiles["Alice"].family_size, 1)
    self.assertEqual(profiles["Alice"].daily_food_needed, 3)
    self.assertEqual(profiles["Alice"].food_balance, 3)
    self.assertEqual(profiles["Alice"].liquid_balance, 500.0)
    self.assertEqual(profiles["Alice"].weekly_rent, 200.0)
    self.assertEqual(profiles["Alice"].weekly_wage, 510.0)  # 600 * 0.85
    self.assertEqual(profiles["Alice"].inventory, {"white t-shirt": 1})

    # Bob: middle, married, office_floor_tech (1.20x)
    self.assertEqual(profiles["Bob"].economic_class, "middle")
    self.assertEqual(profiles["Bob"].family_size, 2)
    self.assertEqual(profiles["Bob"].daily_food_needed, 6)
    self.assertEqual(profiles["Bob"].food_balance, 6)
    self.assertEqual(profiles["Bob"].liquid_balance, 1000.0)
    self.assertEqual(profiles["Bob"].weekly_rent, 350.0)
    self.assertEqual(profiles["Bob"].weekly_wage, 1080.0)  # 900 * 1.20

    # David: elite, single, unemployed (0.40x welfare)
    self.assertEqual(profiles["David"].economic_class, "elite")
    self.assertEqual(profiles["David"].family_size, 1)
    self.assertEqual(profiles["David"].daily_food_needed, 3)
    self.assertEqual(profiles["David"].food_balance, 3)
    self.assertEqual(profiles["David"].liquid_balance, 15000.0)
    self.assertEqual(profiles["David"].weekly_rent, 1500.0)
    self.assertEqual(profiles["David"].weekly_wage, 1600.0)  # 4000 * 0.40


class FoodConsumptionTest(absltest.TestCase):

  def test_food_consumption_and_hunger(self):
    profiles = {
        "Alice": economic_profile.AgentEconomicProfile(
            name="Alice",
            economic_class="lower_middle",
            weekly_wage=500.0,
            weekly_rent=200.0,
            family_size=1,
            food_min_daily=3,
            food_balance=5,  # Has 5 units, needs 3
        ),
        "Bob": economic_profile.AgentEconomicProfile(
            name="Bob",
            economic_class="middle",
            weekly_wage=1000.0,
            weekly_rent=350.0,
            family_size=2,
            food_min_daily=3,
            food_balance=2,  # Has 2 units, needs 6 -> HUNGER
        ),
    }

    clock = fixed_clock.FixedIntervalClock(
        tick_interval_minutes=120,
    )
    clock._current_dt = datetime.datetime(2026, 1, 1, 7, 0)
    make_obs = _MockMakeObservation()

    food_comp = food_consumption.FoodConsumptionComponent(
        profiles=profiles,
        clock_key="clock",
        make_observation_key="make_observation",
    )

    _ = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="island rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={
            "clock": clock,
            "make_observation": make_obs,
            "food_consumption": food_comp,
        },
    )

    action_spec = entity_lib.ActionSpec(
        call_to_action="",
        output_type=entity_lib.OutputType.RESOLVE,
    )
    food_comp.pre_act(action_spec)

    # Alice had 5, needed 3 -> now has 2
    self.assertEqual(profiles["Alice"].food_balance, 2)
    alice_queue = make_obs._observation_queue.get("Alice", [])
    self.assertNotEmpty(alice_queue)
    self.assertIn("prepares a meal", alice_queue[0])

    # Bob had 2, needed 6 -> now has 0 and hunger observation
    self.assertEqual(profiles["Bob"].food_balance, 0)
    bob_queue = make_obs._observation_queue.get("Bob", [])
    self.assertNotEmpty(bob_queue)
    self.assertIn("wakes up feeling weak and hungry", bob_queue[0])


class FiscalSchedulerTest(absltest.TestCase):

  def test_monday_payroll_and_friday_rent(self):
    profiles = {
        "Alice": economic_profile.AgentEconomicProfile(
            name="Alice",
            economic_class="lower_middle",
            weekly_wage=600.0,
            weekly_rent=200.0,
            family_size=1,
            liquid_balance=100.0,
            is_employed=True,
        ),
        "Bob": economic_profile.AgentEconomicProfile(
            name="Bob",
            economic_class="middle",
            weekly_wage=900.0,
            weekly_rent=350.0,
            family_size=2,
            liquid_balance=50.0,  # Insufficient for $350 rent
            is_employed=True,
        ),
    }

    clock = fixed_clock.FixedIntervalClock(
        tick_interval_minutes=120,
    )
    clock._current_dt = datetime.datetime(2026, 1, 5, 7, 0)
    make_obs = _MockMakeObservation()

    scheduler = fiscal_scheduler.FiscalScheduler(
        fiscal_events=fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles=profiles,
        clock_key="clock",
        make_observation_key="make_observation",
    )

    _ = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="island rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={
            "clock": clock,
            "make_observation": make_obs,
            "fiscal_scheduler": scheduler,
        },
    )

    action_spec = entity_lib.ActionSpec(
        call_to_action="",
        output_type=entity_lib.OutputType.RESOLVE,
    )

    # Monday 7:00 AM -> Payroll fires!
    scheduler.pre_act(action_spec)
    # Alice: 100 + 600 = 700
    self.assertEqual(profiles["Alice"].liquid_balance, 700.0)
    # Bob: 50 + 900 = 950
    self.assertEqual(profiles["Bob"].liquid_balance, 950.0)

    # Advance clock to Friday Jan 9, 2026 @ 3:00 PM (15:00, tick 4)
    clock._current_dt = datetime.datetime(2026, 1, 9, 15, 0)
    profiles["Bob"].liquid_balance = 50.0  # Reset Bob to $50 to test arrears

    scheduler.pre_act(action_spec)
    # Alice had $700, rent $200 -> now $500
    self.assertEqual(profiles["Alice"].liquid_balance, 500.0)

    # Bob had $50, rent $350 -> now $0.0 (arrears)
    self.assertEqual(profiles["Bob"].liquid_balance, 0.0)
    bob_queue = make_obs._observation_queue.get("Bob", [])
    arrears_obs = [obs for obs in bob_queue if "RENT WARNING" in obs]
    self.assertNotEmpty(arrears_obs)

  def test_event_applies_on_top_of_marketplace_spending(self):
    """Regression test for the wallet clobber seen in an earlier fiscal run.

    The scheduler runs in the island GM process; the nighttime marketplace
    runs in its own worker(s) and is the only component that spends money. The
    marketplace writes the post-spending balance to the world state, but
    nothing pushes it back into the island process. The scheduler used to
    read-modify-write its own stale copy, so the next payday or rent debit
    handed the agent back everything they had spent overnight.

    Concretely, in an earlier fiscal run Tara Lee moved $1000 of her $2000
    into savings at the marketplace. The marketplace persisted $1000. The Friday
    rent event then debited her $550 rent from the stale $2000 and wrote $1450
    back -- refunding the $1000 she had put away. This test pins that exact
    shape.
    """
    tara_starting_balance = 2000.0
    tara_rent = 550.0
    balance_after_marketplace = 1000.0

    profiles = {
        "Tara Lee": economic_profile.AgentEconomicProfile(
            name="Tara Lee",
            economic_class="middle",
            weekly_wage=0.0,
            weekly_rent=tara_rent,
            family_size=1,
            liquid_balance=tara_starting_balance,
            is_employed=False,
        ),
    }

    # What the marketplace left behind in the world state overnight.
    world_state = _FakeWorldState(
        wallets={"Tara Lee": balance_after_marketplace}
    )

    clock = fixed_clock.FixedIntervalClock(tick_interval_minutes=120)
    # Friday Jan 9 2026, 3:00 PM -> the rent debit.
    clock._current_dt = datetime.datetime(2026, 1, 9, 15, 0)
    make_obs = _MockMakeObservation()

    scheduler = fiscal_scheduler.FiscalScheduler(
        fiscal_events=fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles=profiles,
        clock_key="clock",
        make_observation_key="make_observation",
        world_state=world_state,
    )

    _ = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="island rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={
            "clock": clock,
            "make_observation": make_obs,
            "fiscal_scheduler": scheduler,
        },
    )

    scheduler.pre_act(
        entity_lib.ActionSpec(
            call_to_action="",
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )

    expected = balance_after_marketplace - tara_rent
    self.assertEqual(profiles["Tara Lee"].liquid_balance, expected)
    self.assertEqual(world_state.wallets["Tara Lee"][0], expected)
    # The stale-copy arithmetic that produced the bug.
    self.assertNotEqual(
        world_state.wallets["Tara Lee"][0], tara_starting_balance - tara_rent
    )


class MarketplaceNightGMTest(absltest.TestCase):

  def test_build_honors_explicit_shard_name(self):
    """Regression test for a sharding failure in an earlier run.

    `build()` used to resolve every setting with
    `merged_params.get(key, keyword_argument)`. Because the prefab's
    `default_factory` populates every key, that expression could never fall
    through to the keyword argument, so a shard worker asking for
    'marketplace_rules 3' silently got an entity named 'marketplace_rules'.
    All ten shards then registered with the engine under the same name, which
    made the engine's `hasGM("marketplace_rules 3")` check fail, which
    suppressed every nighttime route. The run completed normally with an
    empty OrderBook.
    """
    model = mock_language_model.FastMockLanguageModel(verbose=False)
    gm = marketplace_night_gm.MarketplaceNightGameMaster().build(
        model=model,
        player_names=["Alice", "Bob"],
        marketplace_gm_name="marketplace_rules 3",
    )
    self.assertEqual(gm.name, "marketplace_rules 3")

  def test_build_defaults_to_unsharded_name(self):
    model = mock_language_model.FastMockLanguageModel(verbose=False)
    gm = marketplace_night_gm.MarketplaceNightGameMaster().build(
        model=model,
        player_names=["Alice", "Bob"],
    )
    self.assertEqual(gm.name, "marketplace_rules")

  def test_build_honors_explicit_max_rounds(self):
    # Same latent bug as the shard name: --marketplace_rounds was being
    # discarded in favour of the prefab default.
    model = mock_language_model.FastMockLanguageModel(verbose=False)
    gm = marketplace_night_gm.MarketplaceNightGameMaster().build(
        model=model,
        player_names=["Alice", "Bob"],
        max_rounds=9,
    )
    market = gm.get_component("marketplace")
    self.assertEqual(market._max_rounds, 9)

  def test_unknown_agent_raises_instead_of_silent_noop(self):
    agent = marketplace_night_gm.MarketplaceAgent(
        name="Alice", role="consumer", cash=100.0, inventory={}, queue=[]
    )
    market = marketplace_night_gm.IslandMarketPlace(
        acting_player_names=["Alice"],
        agents=[agent],
        goods=[],
        economic_profiles={},
        annual_interest_rate=0.035,
        food_units_map={},
        market_type="fixed_prices",
    )
    with self.assertRaisesRegex(ValueError, "unknown agent 'Mallory'"):
      market._handle_next_action_spec("Mallory")
    with self.assertRaisesRegex(ValueError, "unknown agent 'Mallory'"):
      market._handle_make_observation("Mallory")

  def test_order_ids_do_not_collide_across_shards(self):
    """Regression test for the lost order in an earlier fiscal run.

    Order ids were `mkt_{round}_{counter}` where the counter is a per-process
    field. Every shard therefore started at 1 and minted the same ids. The
    OrderBook is keyed on `(run id, order id)` and `record_order` upserts, so
    the second shard's first sale of a round silently replaced the first
    shard's. In an earlier fiscal run that destroyed Olivia Welch's Casio
    Digital Watch: five purchases are visible in the agents' memories, but
    only four rows reached the order book.

    Both shards write to the *same* world state here, as they would with a
    shared backend.
    """
    world_state = _FakeWorldState()
    good = marketplace_night_gm.Good(
        category="Electronics",
        quality="Low",
        id="Casio Digital Watch",
        price=35.0,
        inventory=100,
    )

    def _build_shard(shard_gm_name: str, buyer_name: str):
      agent = marketplace_night_gm.MarketplaceAgent(
          name=buyer_name,
          role="consumer",
          cash=500.0,
          inventory={},
          queue=[],
      )
      profiles = {
          buyer_name: economic_profile.AgentEconomicProfile(
              name=buyer_name,
              economic_class="middle",
              weekly_wage=600.0,
              weekly_rent=200.0,
              family_size=1,
              liquid_balance=500.0,
          )
      }
      market = marketplace_night_gm.IslandMarketPlace(
          acting_player_names=[buyer_name],
          agents=[agent],
          goods=[good],
          economic_profiles=profiles,
          annual_interest_rate=0.035,
          food_units_map={},
          market_type="fixed_prices",
          world_state=world_state,
          marketplace_gm_name=shard_gm_name,
      )
      _ = entity_agent_with_logging.EntityAgentWithLogging(
          agent_name=shard_gm_name,
          act_component=actor_components.constant.Constant(state=""),
          context_components={"marketplace": market},
      )
      return market

    shard_0 = _build_shard("marketplace_rules 0", "Olivia Welch")
    shard_1 = _build_shard("marketplace_rules 1", "Jon Huang")

    order = '{"action": "bid", "good": "Casio Digital Watch", "price": 35.0,'
    order += ' "qty": 1}'
    shard_0.pre_observe(f"[putative_event] Olivia Welch: {order}")
    shard_0._resolve()
    shard_1.pre_observe(f"[putative_event] Jon Huang: {order}")
    shard_1._resolve()

    self.assertLen(world_state.orders, 2)
    self.assertCountEqual(
        [row["agent_id"] for row in world_state.orders.values()],
        ["Olivia Welch", "Jon Huang"],
    )

  def test_marketplace_actions_save_withdraw_bid(self):
    profiles = {
        "Alice": economic_profile.AgentEconomicProfile(
            name="Alice",
            economic_class="lower_middle",
            weekly_wage=600.0,
            weekly_rent=200.0,
            family_size=1,
            liquid_balance=500.0,
            savings_balance=100.0,
            food_balance=0,
            inventory={"white t-shirt": 1},
        ),
    }

    agent = marketplace_night_gm.MarketplaceAgent(
        name="Alice",
        role="consumer",
        cash=500.0,
        inventory={},
        queue=[],
    )

    good = marketplace_night_gm.Good(
        category="Food",
        quality="Low",
        id="Maruchan Ramen Meal",
        price=3.0,
        inventory=100,
    )

    food_units_map = {"Maruchan Ramen Meal": 1}

    market = marketplace_night_gm.IslandMarketPlace(
        acting_player_names=["Alice"],
        agents=[agent],
        goods=[good],
        economic_profiles=profiles,
        annual_interest_rate=0.035,
        food_units_map=food_units_map,
        market_type="fixed_prices",
    )

    # Attach to an entity agent so get_entity() succeeds
    _ = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="marketplace_rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={
            "marketplace": market,
        },
    )

    # Test next_action_spec contains ephemeral financial status, goods, and
    # options.
    spec_str = market._handle_next_action_spec("Alice")
    self.assertIn("[Your Financial Status & Inventory]", spec_str)
    self.assertIn("Checking Account (Cash): $500.00", spec_str)
    self.assertIn("Savings Account: $100.00", spec_str)
    self.assertIn("Wardrobe & Possessions: white t-shirt (1)", spec_str)
    self.assertIn(
        "Household Food Stock: 0 units (storage capacity: 0/60 units", spec_str
    )
    self.assertIn("save", spec_str)
    self.assertIn("withdraw", spec_str)
    self.assertIn("pass", spec_str)

    # Test observation generation initially produces NO memory pollution
    obs_initial = market._handle_make_observation("Alice")
    self.assertEqual(obs_initial, "")

    # Deliver the action the way both engines do: tagged as a putative event.
    market.pre_observe(
        '[putative_event] Alice: {"action": "save", "amount": 50.0}'
    )
    market._resolve()

    # Test third-person action resolution in observation
    obs_after = market._handle_make_observation("Alice")
    self.assertIn(
        "Alice transferred $50.00 into their savings account", obs_after
    )
    self.assertIn("Checking: $450.00", obs_after)
    self.assertIn("Savings: $150.01", obs_after)

  def test_marketplace_durable_good_purchase(self):
    profiles = {
        "Alice": economic_profile.AgentEconomicProfile(
            name="Alice",
            economic_class="lower_middle",
            weekly_wage=600.0,
            weekly_rent=200.0,
            family_size=1,
            liquid_balance=500.0,
            savings_balance=100.0,
            food_balance=6,
            inventory={"white t-shirt": 1},
        ),
    }

    agent = marketplace_night_gm.MarketplaceAgent(
        name="Alice",
        role="consumer",
        cash=500.0,
        inventory={},
        queue=[],
    )

    good = marketplace_night_gm.Good(
        category="Clothing",
        quality="Low",
        id="Uniqlo Supima Cotton T-Shirt",
        price=20.0,
        inventory=100,
    )

    market = marketplace_night_gm.IslandMarketPlace(
        acting_player_names=["Alice"],
        agents=[agent],
        goods=[good],
        economic_profiles=profiles,
        annual_interest_rate=0.035,
        food_units_map={},  # Non-food good
        market_type="fixed_prices",
    )

    _ = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="marketplace_rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={
            "marketplace": market,
        },
    )

    market.pre_observe(
        '[putative_event] Alice: {"action": "bid", "good": "Uniqlo Supima'
        ' Cotton T-Shirt", "price": 20.0, "qty": 1}'
    )
    market._resolve()

    # Verify inventory is incremented and food balance is unchanged
    self.assertEqual(
        profiles["Alice"].inventory["Uniqlo Supima Cotton T-Shirt"], 1
    )
    self.assertEqual(profiles["Alice"].food_balance, 6)
    self.assertEqual(profiles["Alice"].liquid_balance, 480.0)

    # Verify observation notes durable possession without mentioning food stock
    obs_after = market._handle_make_observation("Alice")
    self.assertIn(
        "purchased 1 units of Uniqlo Supima Cotton T-Shirt", obs_after
    )
    self.assertIn("Added to household possessions", obs_after)
    self.assertNotIn("household food stock", obs_after)

  def test_marketplace_empty_wardrobe_and_full_fridge_rendering(self):
    """An empty wardrobe must not claim possessions the agent does not have."""
    profiles = {
        "Alice": economic_profile.AgentEconomicProfile(
            name="Alice",
            economic_class="lower_middle",
            weekly_wage=600.0,
            weekly_rent=200.0,
            family_size=1,
            liquid_balance=500.0,
            savings_balance=100.0,
            food_balance=marketplace_night_gm.MAX_FRIDGE_CAPACITY_UNITS,
            inventory={},
        ),
    }

    agent = marketplace_night_gm.MarketplaceAgent(
        name="Alice",
        role="consumer",
        cash=500.0,
        inventory={},
        queue=[],
    )

    good = marketplace_night_gm.Good(
        category="Food",
        quality="Low",
        id="Maruchan Ramen Meal",
        price=3.0,
        inventory=100,
    )

    market = marketplace_night_gm.IslandMarketPlace(
        acting_player_names=["Alice"],
        agents=[agent],
        goods=[good],
        economic_profiles=profiles,
        annual_interest_rate=0.035,
        food_units_map={"Maruchan Ramen Meal": 1},
        market_type="fixed_prices",
    )

    entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="marketplace_rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={"marketplace": market},
    )

    spec_str = market._handle_next_action_spec("Alice")
    self.assertIn("Wardrobe & Possessions: (none)", spec_str)
    self.assertNotIn("white t-shirt", spec_str)
    self.assertIn("refrigerator is fully stocked", spec_str)
    self.assertNotIn("storage capacity:", spec_str)

  def _build_single_agent_market(self, goods=None):
    """Returns (market, profiles, agent) for a one-agent fixed-price market."""
    profiles = {
        "Alice": economic_profile.AgentEconomicProfile(
            name="Alice",
            economic_class="lower_middle",
            weekly_wage=600.0,
            weekly_rent=200.0,
            family_size=1,
            liquid_balance=500.0,
            savings_balance=0.0,
            food_balance=0,
            inventory={},
        ),
    }
    agent = marketplace_night_gm.MarketplaceAgent(
        name="Alice",
        role="consumer",
        cash=500.0,
        inventory={},
        queue=[],
    )
    if goods is None:
      goods = [
          marketplace_night_gm.Good(
              category="Food",
              quality="Low",
              id="Panera Soup and Sandwich Meal",
              price=17.0,
              inventory=100,
          )
      ]
    market = marketplace_night_gm.IslandMarketPlace(
        acting_player_names=["Alice", "Bob"],
        agents=[agent],
        goods=goods,
        economic_profiles=profiles,
        annual_interest_rate=0.035,
        food_units_map={"Panera Soup and Sandwich Meal": 1},
        market_type="fixed_prices",
    )
    entity_agent_with_logging.EntityAgentWithLogging(
        agent_name="marketplace_rules",
        act_component=actor_components.constant.Constant(state=""),
        context_components={"marketplace": market},
    )
    return market, profiles, agent

  def test_call_to_action_examples_are_valid_json(self):
    """Every JSON example in the prompt must actually parse.

    The prompt used to show <GOOD_ID>/<price>/<int> placeholders. Anything that
    fed the prompt back into the action parser then produced an opaque decode
    error rather than a diagnosable rejection.
    """
    market, _, _ = self._build_single_agent_market()
    spec_str = market._handle_next_action_spec("Alice")
    # action_spec_to_string escapes the quotes, so assert against the raw
    # call-to-action, which is what the model actually reads.
    call_to_action = market._last_call_to_action["Alice"]

    examples = re.findall(r'\{"action".*?\}', call_to_action)
    self.assertLen(examples, 4)
    for example in examples:
      json.loads(example)  # Raises if the example is not valid JSON.

    self.assertNotIn("<GOOD_ID>", spec_str)
    self.assertNotIn("<price>", spec_str)
    self.assertNotIn("<float>", spec_str)

  def test_resolve_never_parses_its_own_prompt(self):
    """An agent that did not act must not have the prompt read back as its answer.

    Regression test for the defect found in an earlier run: _resolve fell back
    to the game master's own component text for every agent without a captured
    action, matched the example JSON inside the call-to-action, and rejected
    99.1% of all marketplace activity with "order could not be read".
    """
    market, _, agent = self._build_single_agent_market()

    # Send Alice a prompt, then resolve without her ever answering.
    market._handle_next_action_spec("Alice")
    market._resolve()

    observation = market._handle_make_observation("Alice")
    self.assertEqual(observation, "")
    self.assertEmpty(agent.queue)
    # She never acted, so she must not count toward round completion.
    self.assertEmpty(market._resolved_agents_in_round)

  def test_resolve_accepts_slug_style_good_id(self):
    """A slug item name must resolve to the catalogue's display name."""
    market, profiles, _ = self._build_single_agent_market()

    market.pre_observe(
        '[putative_event] Alice: {"action": "bid", "good":'
        ' "panera_soup_and_sandwich_meal", "price": 17.0, "qty": 1}'
    )
    market._resolve()

    observation = market._handle_make_observation("Alice")
    self.assertIn(
        "purchased 1 units of Panera Soup and Sandwich Meal", observation
    )
    self.assertNotIn("not one of the items sold", observation)
    self.assertEqual(profiles["Alice"].liquid_balance, 483.0)

  def test_putative_action_is_not_attributed_to_a_merely_mentioned_agent(self):
    """Naming another player inside an action must not place an order for them."""
    market, _, _ = self._build_single_agent_market()

    # "Bob" is an acting player and is mentioned in Alice's answer.
    market.pre_observe(
        "[putative_event] Alice: Alice tells Bob she is buying dinner."
        ' {"action": "bid", "good": "Panera Soup and Sandwich Meal",'
        ' "price": 17.0, "qty": 1}'
    )

    self.assertIn("Alice", market._pending_putative_actions)
    self.assertNotIn("Bob", market._pending_putative_actions)

  def test_observations_are_stamped_with_the_agents_own_round(self):
    """Observations must say which of the night's rounds the agent is on."""
    market, _, _ = self._build_single_agent_market()
    market._max_rounds = 5

    market.pre_observe('[putative_event] Alice: {"action": "pass"}')
    market._resolve()
    first = market._handle_make_observation("Alice")
    self.assertIn("Round 1/5 - Alice chose not to make any purchases", first)

    market.pre_observe(
        '[putative_event] Alice: {"action": "bid", "good": "Panera Soup and'
        ' Sandwich Meal", "price": 17.0, "qty": 1}'
    )
    market._resolve()
    second = market._handle_make_observation("Alice")
    self.assertIn("Round 2/5 - Alice purchased 1 units", second)

  def test_multi_day_simulation_cycle(self):
    class _DummyEmbedder:

      def __call__(self, text):
        return np.zeros(768, dtype=np.float32)

    model = mock_language_model.FastMockLanguageModel(verbose=False)
    embedder = _DummyEmbedder()
    agent_configs = [
        agents_lib.AgentConfig(
            name="Alice",
            gender="female",
            home_place="millbrook_apts_unit_1",
            work_place="cafe",
            personality="diligent and cautious",
            backstory="Alice is a resident of Brecksville.",
        ),
        agents_lib.AgentConfig(
            name="Bob",
            gender="male",
            home_place="brecksville_commons_unit_5",
            work_place="office_floor_tech",
            personality="outgoing and technical",
            backstory="Bob works as a software engineer.",
        ),
    ]

    sim = island_simulation.IslandConcordiaSimulation(
        agent_configs=agent_configs,
        model=model,
        embedder=embedder,
        start_time="Thursday, January 1st, 7:00 AM",
        engine_type="simultaneous",
        max_ticks=2,
        tick_interval_minutes=120,
        enable_nighttime_marketplace=True,
        marketplace_rounds=1,
    )

    results = sim.play(max_ticks=2)
    self.assertIsNotNone(results)
    self.assertIn("Alice", sim.economic_profiles)
    self.assertIn("Bob", sim.economic_profiles)


class FiscalConfigsTest(absltest.TestCase):

  def test_known_configs(self):
    self.assertIs(
        fiscal_configs.get_fiscal_events("control"),
        fiscal_configs.CONTROL_FISCAL_EVENTS,
    )
    self.assertIs(
        fiscal_configs.get_fiscal_events("UBI"),
        fiscal_configs.UBI_FISCAL_EVENTS,
    )

  def test_unknown_config_raises(self):
    for name in ("ubs", "ubc", "fjg", "delayed_ubi", "typo"):
      with self.subTest(name=name):
        with self.assertRaisesRegex(ValueError, "Unknown fiscal_config"):
          fiscal_configs.get_fiscal_events(name)


if __name__ == "__main__":
  absltest.main()

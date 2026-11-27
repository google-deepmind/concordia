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

  def apply_debits_checking_then_savings(
      self, debits: dict[str, float]
  ) -> dict[str, tuple[float, float, float, float]]:
    """Mirrors WorldState.apply_debits_checking_then_savings."""
    out = {}
    for agent_id, amount in debits.items():
      cb, sb = self.wallets.get(agent_id, (0.0, 0.0))
      from_checking = min(amount, max(0.0, cb))
      from_savings = min(amount - from_checking, max(0.0, sb))
      ca, sa = round(cb - from_checking, 2), round(sb - from_savings, 2)
      self.wallets[agent_id] = (ca, sa)
      out[agent_id] = (cb, ca, sb, sa)
    return out

  def get_inventory(self, agent_id: str) -> dict[str, int]:
    return dict(self.inventories.get(agent_id, {}))

  def upsert_inventory_batch(self, inventories: dict[str, dict[str, int]]):
    for name, inv in inventories.items():
      self.inventories.setdefault(name, {}).update(inv)

  def apply_inventory_deltas(
      self, deltas: dict[str, dict[str, int]]
  ) -> dict[str, dict[str, tuple[int, int]]]:
    """Mirrors WorldState.apply_inventory_deltas, including the zero floor."""
    committed = {}
    for name, items in deltas.items():
      inv = self.inventories.setdefault(name, {})
      out = committed.setdefault(name, {})
      for item, delta in items.items():
        before = int(inv.get(item, 0))
        inv[item] = max(0, before + int(delta))
        out[item] = (before, inv[item])
    return committed

  def record_order(self, order_id: str, **fields) -> None:
    self.orders[order_id] = dict(fields)

  def enqueue_observations(self, observations: dict[str, list[str]]) -> None:
    for name, obs in observations.items():
      self.observations.setdefault(name, []).extend(obs)


_HOMES_BY_CLASS = {
    "lower_middle": "millbrook_apts_unit_1",
    "middle": "brecksville_commons_unit_5",
    "upper_middle": "palm_heights_3",
    "upper": "ocean_view_estates_1",
    "elite": "timber_creek_unit_2",
}


class EconomicProfileTest(absltest.TestCase):

  def _expected_rent(
      self, quintile: str, region: str = economic_profile.DEFAULT_BLS_REGION
  ) -> float:
    annual = economic_profile.BLS_ANNUAL_VALUES[region][
        economic_profile.BLS_RENT_CATEGORY
    ][quintile]
    return round(annual / 52, 2)

  def _expected_income(self, quintile: str, region: str) -> float:
    annual = economic_profile.BLS_ANNUAL_VALUES[region][
        economic_profile.BLS_INCOME_CATEGORY
    ][quintile]
    return round(annual / 52, 2)

  def test_bls_constants_cover_every_region_category_and_quintile(self):
    self.assertCountEqual(
        economic_profile.BLS_ANNUAL_VALUES, ["midwest", "us"]
    )
    self.assertCountEqual(
        economic_profile.CLASS_TO_BLS_QUINTILE,
        economic_profile.TIER_CONFIG,
    )
    self.assertCountEqual(
        economic_profile.CLASS_TO_BLS_QUINTILE.values(),
        ["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    for region, categories in economic_profile.BLS_ANNUAL_VALUES.items():
      self.assertEqual(
          sorted(categories), ["income_before_taxes", "shelter"], region
      )
      for category, by_quintile in categories.items():
        with self.subTest(region=region, category=category):
          self.assertEqual(sorted(by_quintile), ["Q1", "Q2", "Q3", "Q4", "Q5"])
          values = [by_quintile[q] for q in sorted(by_quintile)]
          self.assertTrue(all(v > 0 for v in values))
          # Both rise with the income quintile.
          self.assertEqual(values, sorted(values))

  def test_us_values_are_whole_dollars_as_published_in_table_1101(self):
    for category, by_quintile in economic_profile.BLS_ANNUAL_VALUES[
        "us"
    ].items():
      for quintile, value in by_quintile.items():
        with self.subTest(category=category, quintile=quintile):
          self.assertEqual(value, round(value))

  def test_weekly_rent_and_wage_are_annual_over_52_rounded_to_cents(self):
    for region in ("us", "midwest"):
      for (
          econ_class,
          quintile,
      ) in economic_profile.CLASS_TO_BLS_QUINTILE.items():
        with self.subTest(region=region, econ_class=econ_class):
          self.assertEqual(
              economic_profile.bls_weekly_rent(econ_class, region=region),
              self._expected_rent(quintile, region),
          )
          self.assertEqual(
              economic_profile.bls_weekly_income(econ_class, region=region),
              self._expected_income(quintile, region),
          )

  def test_build_economic_profiles_honors_region(self):
    configs = [_MockAgentConfig(name="Eve", home_place="palm_heights_3")]
    for region in ("us", "midwest"):
      profiles = economic_profile.build_economic_profiles(
          configs, bls_region=region
      )
      self.assertEqual(
          profiles["Eve"].weekly_rent, self._expected_rent("Q3", region)
      )

  def test_employed_rent_share_matches_bls_by_quintile(self):
    """Employed agents' rent / wage equals BLS Shelter / income per quintile."""
    for region in ("us", "midwest"):
      values = economic_profile.BLS_ANNUAL_VALUES[region]
      for (
          econ_class,
          quintile,
      ) in economic_profile.CLASS_TO_BLS_QUINTILE.items():
        with self.subTest(region=region, econ_class=econ_class):
          configs = [
              _MockAgentConfig(
                  name="A",
                  home_place=_HOMES_BY_CLASS[econ_class],
                  work_place="cafe",
              )
          ]
          profile = economic_profile.build_economic_profiles(
              configs, bls_region=region
          )["A"]
          self.assertEqual(profile.economic_class, econ_class)
          self.assertEqual(
              profile.weekly_wage, self._expected_income(quintile, region)
          )
          self.assertAlmostEqual(
              profile.weekly_rent / profile.weekly_wage,
              values["shelter"][quintile]
              / values["income_before_taxes"][quintile],
              places=3,
          )

  def test_bls_region_for_setting(self):
    for setting, expected in (
        ("brecksville_1000", "midwest"),
        ("brecksville", "midwest"),
        ("Brecksville_1000", "midwest"),
        ("island", "us"),
        ("kerala", "us"),
        ("", "us"),
        (None, "us"),
    ):
      with self.subTest(setting=setting):
        self.assertEqual(
            economic_profile.bls_region_for_setting(setting), expected
        )

  def test_build_economic_profiles_requires_region(self):
    configs = [_MockAgentConfig(name="Eve", home_place="palm_heights_3")]
    with self.assertRaises(TypeError):
      # Intentionally omit the required keyword-only argument.
      # pylint: disable-next=missing-kwoa
      getattr(economic_profile, "build_economic_profiles")(configs)

  def test_unknown_region_raises(self):
    with self.assertRaises(KeyError):
      economic_profile.bls_weekly_rent("middle", region="atlantis")
    with self.assertRaises(KeyError):
      economic_profile.bls_weekly_income("middle", region="atlantis")

  def test_unknown_class_raises(self):
    with self.assertRaises(KeyError):
      economic_profile.bls_weekly_rent("aristocrat")
    with self.assertRaises(KeyError):
      economic_profile.bls_weekly_income("aristocrat", region="midwest")

  def test_unknown_category_raises(self):
    with self.assertRaises(KeyError):
      economic_profile.bls_weekly_rent("middle", category="housing")

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
        configs,
        food_min_daily=3,
        starting_food_units=None,
        bls_region="midwest",
    )

    # Alice: lower_middle, single, employed
    self.assertEqual(profiles["Alice"].economic_class, "lower_middle")
    self.assertEqual(profiles["Alice"].family_size, 1)
    self.assertEqual(profiles["Alice"].daily_food_needed, 3)
    self.assertEqual(profiles["Alice"].food_balance, 3)
    self.assertEqual(profiles["Alice"].liquid_balance, 500.0)
    self.assertEqual(
        profiles["Alice"].weekly_rent, self._expected_rent("Q1", "midwest")
    )
    self.assertEqual(
        profiles["Alice"].weekly_wage, self._expected_income("Q1", "midwest")
    )
    self.assertEqual(profiles["Alice"].inventory, {"white t-shirt": 1})

    # Bob: middle, married, employed (no occupation multiplier)
    self.assertEqual(profiles["Bob"].economic_class, "middle")
    self.assertEqual(profiles["Bob"].family_size, 2)
    self.assertEqual(profiles["Bob"].daily_food_needed, 6)
    self.assertEqual(profiles["Bob"].food_balance, 6)
    self.assertEqual(profiles["Bob"].liquid_balance, 1000.0)
    self.assertEqual(
        profiles["Bob"].weekly_rent, self._expected_rent("Q2", "midwest")
    )
    self.assertEqual(
        profiles["Bob"].weekly_wage, self._expected_income("Q2", "midwest")
    )

    # David: elite, single, no workplace (welfare)
    self.assertEqual(profiles["David"].economic_class, "elite")
    self.assertEqual(profiles["David"].family_size, 1)
    self.assertEqual(profiles["David"].daily_food_needed, 3)
    self.assertEqual(profiles["David"].food_balance, 3)
    self.assertEqual(profiles["David"].liquid_balance, 15000.0)
    self.assertEqual(
        profiles["David"].weekly_rent, self._expected_rent("Q5", "midwest")
    )
    # No workplace: welfare at NO_WORKPLACE_INCOME_FRACTION of class income.
    self.assertEqual(
        profiles["David"].weekly_wage,
        round(
            self._expected_income("Q5", "midwest")
            * economic_profile.NO_WORKPLACE_INCOME_FRACTION,
            2,
        ),
    )


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
    clock._current_dt = datetime.datetime(2026, 1, 1, 9, 0)
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
    # Bob's 2 units were eaten as a small meal, not discarded.
    self.assertIn("only 2 of the 6 food units", bob_queue[0])
    self.assertIn("[Thursday, January 1, 9:00 AM]", bob_queue[0])


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
    clock._current_dt = datetime.datetime(2026, 1, 5, 9, 0)
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

    # Monday 9:00 AM -> Payroll fires!
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


def _employed_profile(name: str) -> economic_profile.AgentEconomicProfile:
  return economic_profile.AgentEconomicProfile(
      name=name,
      economic_class="lower_middle",
      weekly_wage=600.0,
      weekly_rent=200.0,
      family_size=1,
      liquid_balance=100.0,
      is_employed=True,
  )


def _economic_gm(events, profiles, world_state, clock, food=None):
  """Builds a minimal island GM with a FiscalScheduler (+ FoodConsumption).

  Args:
    events: Fiscal events to schedule.
    profiles: Mapping of agent name to AgentEconomicProfile.
    world_state: Shared world state instance, or None.
    clock: FixedIntervalClock instance.
    food: Optional FoodConsumptionComponent instance.

  Returns:
    (fiscal scheduler, MakeObservation stand-in holding the observations).
  """
  make_obs = _MockMakeObservation()
  fiscal = fiscal_scheduler.FiscalScheduler(
      fiscal_events=events,
      profiles=profiles,
      clock_key="clock",
      make_observation_key="make_observation",
      world_state=world_state,
  )
  components = {
      "clock": clock,
      "make_observation": make_obs,
      "fiscal_scheduler": fiscal,
  }
  if food is not None:
    components["food_consumption"] = food
  _ = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name="island rules",
      act_component=actor_components.constant.Constant(state=""),
      context_components=components,
  )
  return fiscal, make_obs


def _observations(make_obs: _MockMakeObservation, name: str) -> list[str]:
  return make_obs._observation_queue.get(name, [])


_RESOLVE = entity_lib.ActionSpec(
    call_to_action="", output_type=entity_lib.OutputType.RESOLVE
)


class PolicyTimingTest(absltest.TestCase):
  """Policy arms start on 2026-01-08, after the Day-6 layoff, not on day 1."""

  def _run(self, events, profiles, world_state, times):
    clock = fixed_clock.FixedIntervalClock(tick_interval_minutes=120)
    clock._current_dt = times[0]
    fiscal, make_obs = _economic_gm(events, profiles, world_state, clock)
    for t in times:
      clock._current_dt = t
      fiscal.pre_act(_RESOLVE)
    return make_obs

  def test_ubi_is_a_monthly_lump_sum_from_policy_start(self):
    profiles = {"Alice": _employed_profile("Alice")}
    profiles["Alice"].is_employed = False
    profiles["Alice"].weekly_wage = 0.0
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    start = datetime.date(2026, 1, 1)
    times = []
    for d in range(45):
      for h in (7, 9, 15):
        times.append(
            datetime.datetime.combine(
                start + datetime.timedelta(days=d), datetime.time(h)
            )
        )
    clock = fixed_clock.FixedIntervalClock(tick_interval_minutes=120)
    clock._current_dt = times[0]
    fiscal, make_obs = _economic_gm(
        fiscal_configs.UBI_FISCAL_EVENTS, profiles, world_state, clock
    )
    received = []
    for t in times:
      before = world_state.wallets["Alice"][0]
      clock._current_dt = t
      fiscal.pre_act(_RESOLVE)
      after = world_state.wallets["Alice"][0]
      if after > before:
        received.append((t, round(after - before, 2)))
    self.assertEqual(
        received,
        [
            (
                datetime.datetime(2026, 1, 8, 9),
                fiscal_configs.UBI_MONTHLY_AMOUNT,
            ),
            (
                datetime.datetime(2026, 2, 8, 9),
                fiscal_configs.UBI_MONTHLY_AMOUNT,
            ),
        ],
    )
    ubi = [o for o in _observations(make_obs, "Alice") if "UBI DEPOSIT" in o]
    self.assertLen(ubi, 2)
    self.assertStartsWith(ubi[0], "// bank [Thursday, January 8, 9:00 AM]")

  def test_monthly_event_clamps_to_month_end(self):
    event = dataclasses.replace(
        fiscal_configs.UBI_FISCAL_EVENTS[1], start_date="2026-01-31"
    )
    profile = _employed_profile("Alice")
    profile.is_employed = False
    world_state = _FakeWorldState(wallets={"Alice": 0.0})
    self._run(
        [event],
        {"Alice": profile},
        world_state,
        [
            datetime.datetime(2026, 1, 31, 9),
            datetime.datetime(2026, 2, 27, 9),
            datetime.datetime(2026, 2, 28, 9),  # Last day of February.
            datetime.datetime(2026, 3, 30, 9),
            datetime.datetime(2026, 3, 31, 9),
        ],
    )
    self.assertEqual(
        world_state.wallets["Alice"][0],
        3 * fiscal_configs.UBI_MONTHLY_AMOUNT,
    )

  def test_ubi_not_paid_before_policy_start(self):
    profiles = {"Alice": _employed_profile("Alice")}
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    # Monday Jan 5: payday only. Previously UBI was also paid here.
    self._run(
        fiscal_configs.UBI_FISCAL_EVENTS,
        profiles,
        world_state,
        [datetime.datetime(2026, 1, 5, 9)],
    )
    self.assertEqual(world_state.wallets["Alice"][0], 700.0)

  def test_morning_events_wait_for_9am(self):
    profiles = {"Alice": _employed_profile("Alice")}
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    self._run(
        fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles,
        world_state,
        [datetime.datetime(2026, 1, 5, 7)],
    )
    self.assertEqual(world_state.wallets["Alice"][0], 100.0)

  def test_missed_tick_still_fires_later_that_day(self):
    profiles = {"Alice": _employed_profile("Alice")}
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    # No step at 9:00 AM on payday; the first one is at 1 PM.
    make_obs = self._run(
        fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles,
        world_state,
        [datetime.datetime(2026, 1, 5, 13), datetime.datetime(2026, 1, 5, 15)],
    )
    self.assertEqual(world_state.wallets["Alice"][0], 700.0)
    paychecks = [o for o in _observations(make_obs, "Alice") if "PAYROLL" in o]
    self.assertLen(paychecks, 1)
    self.assertStartsWith(paychecks[0], "// bank [Monday, January 5, 9:00 AM]")

  def test_rent_shortfall_is_taken_from_savings(self):
    profiles = {"Alice": _employed_profile("Alice")}  # rent 200
    world_state = _FakeWorldState()
    world_state.wallets["Alice"] = (50.0, 500.0)
    make_obs = self._run(
        fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles,
        world_state,
        [datetime.datetime(2026, 1, 9, 15)],
    )
    self.assertEqual(world_state.wallets["Alice"], (0.0, 350.0))
    obs = _observations(make_obs, "Alice")[-1]
    self.assertIn("RENT PAYMENT", obs)
    self.assertIn("$150.00 was taken from savings (savings now $350.00)", obs)

  def test_rent_shortfall_from_savings_without_world_state(self):
    profile = _employed_profile("Alice")  # rent 200
    profile.liquid_balance = 50.0
    profile.savings_balance = 500.0
    make_obs = self._run(
        fiscal_configs.CONTROL_FISCAL_EVENTS,
        {"Alice": profile},
        None,
        [datetime.datetime(2026, 1, 9, 15)],
    )
    self.assertEqual(profile.liquid_balance, 0.0)
    self.assertEqual(profile.savings_balance, 350.0)
    self.assertIn(
        "$150.00 was taken from savings", _observations(make_obs, "Alice")[-1]
    )

  def test_rent_arrears_only_when_checking_and_savings_both_short(self):
    profiles = {"Alice": _employed_profile("Alice")}  # rent 200
    world_state = _FakeWorldState()
    world_state.wallets["Alice"] = (50.0, 100.0)
    make_obs = self._run(
        fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles,
        world_state,
        [datetime.datetime(2026, 1, 9, 15)],
    )
    self.assertEqual(world_state.wallets["Alice"], (0.0, 0.0))
    obs = _observations(make_obs, "Alice")[-1]
    self.assertIn("RENT WARNING", obs)
    self.assertIn("Only $150.00 was deducted", obs)

  def test_food_credit_is_an_inventory_delta(self):
    event = economic_profile.FiscalEvent(
        name="food_credit",
        event_type="credit",
        target_account="food",
        amount_source="profile_field",
        profile_field="daily_food_needed",
        start_date="2026-01-08",
        tick_of_day=1,
        observation_template=(
            "// home [{day_name}, {date}, 9:00 AM]: {amount} food units."
        ),
    )
    profile = _employed_profile("Alice")
    profile.food_balance = 0
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    world_state.inventories["Alice"] = {"food_units": 4}
    self._run(
        [event],
        {"Alice": profile},
        world_state,
        [datetime.datetime(2026, 1, 7, 9), datetime.datetime(2026, 1, 8, 9)],
    )
    # Added to the shared pantry, not written from this GM's copy (0).
    self.assertEqual(
        world_state.inventories["Alice"]["food_units"],
        4 + profile.daily_food_needed,
    )
    self.assertEqual(profile.food_balance, 4 + profile.daily_food_needed)


class MorningMealTest(absltest.TestCase):
  """The meal is processed once per day, after the night marketplace."""

  def _gm(self, profiles, world_state, fiscal_events=None):
    clock = fixed_clock.FixedIntervalClock(tick_interval_minutes=120)
    food = food_consumption.FoodConsumptionComponent(
        profiles=profiles,
        clock_key="clock",
        make_observation_key="make_observation",
        world_state=world_state,
        fiscal_scheduler_key="fiscal_scheduler",
    )
    _, make_obs = _economic_gm(
        fiscal_events or fiscal_configs.CONTROL_FISCAL_EVENTS,
        profiles,
        world_state,
        clock,
        food=food,
    )
    return food, clock, make_obs

  def _meals(self, make_obs, name):
    return [
        o for o in _observations(make_obs, name) if o.startswith("// home [")
    ]

  def test_partial_pantry_is_eaten_and_reported_truthfully(self):
    profile = _employed_profile("Alice")
    profile.food_min_daily = 3
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    world_state.inventories["Alice"] = {"food_units": 2}
    food, clock, make_obs = self._gm({"Alice": profile}, world_state)
    clock._current_dt = datetime.datetime(2026, 1, 8, 9)
    food.pre_act(_RESOLVE)
    obs = self._meals(make_obs, "Alice")[-1]
    self.assertIn("feeling weak and hungry", obs)
    self.assertIn("only 2 of the 3 food units", obs)
    self.assertIn("ate a small meal", obs)
    self.assertNotIn("nothing to eat", obs)
    # Dated, so a repeat a week later is not de-duplicated in memory.
    self.assertStartsWith(obs, "// home [Thursday, January 8, 9:00 AM]")
    self.assertEqual(world_state.inventories["Alice"]["food_units"], 0)

  def test_hungry_streak_is_reported(self):
    profile = _employed_profile("Alice")
    profile.food_balance = 0
    food, _, make_obs = self._gm({"Alice": profile}, None)
    food.process_due(datetime.datetime(2026, 1, 8, 9))
    food.process_due(datetime.datetime(2026, 1, 9, 9))
    meals = self._meals(make_obs, "Alice")
    self.assertLen(meals, 2)
    self.assertIn("the pantry is empty and there is nothing to eat.", meals[0])
    self.assertIn("(day 2 in a row without enough food at home)", meals[1])

  def test_first_meal_uses_starting_pantry_when_world_state_has_no_row(self):
    profile = _employed_profile("Alice")
    profile.food_min_daily = 3
    profile.food_balance = 12
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    food, clock, make_obs = self._gm({"Alice": profile}, world_state)
    clock._current_dt = datetime.datetime(2026, 1, 1, 9)
    food.pre_act(_RESOLVE)
    meals = self._meals(make_obs, "Alice")
    self.assertLen(meals, 1)
    self.assertIn("prepares a meal", meals[0])
    self.assertIn("(9 food units left at home)", meals[0])
    self.assertEqual(world_state.inventories["Alice"]["food_units"], 9)

  def test_meal_waits_for_marketplace_and_reports_fresh_balance(self):
    profile = _employed_profile("Alice")
    profile.food_min_daily = 3
    world_state = _FakeWorldState(wallets={"Alice": 100.0})
    world_state.inventories["Alice"] = {"food_units": 0}
    food, clock, make_obs = self._gm({"Alice": profile}, world_state)
    # Monday 7:00 AM: the night marketplace may still be running. No meal.
    clock._current_dt = datetime.datetime(2026, 1, 5, 7)
    food.pre_act(_RESOLVE)
    self.assertEmpty(self._meals(make_obs, "Alice"))
    # The marketplace then sells Alice 5 units for $20.
    world_state.apply_wallet_deltas({"Alice": (-20.0, 0.0)})
    world_state.apply_inventory_deltas({"Alice": {"food_units": 5}})
    # 9:00 AM: payroll (+$600) is applied before the meal, which eats the
    # food bought last night and reports the balance after both.
    clock._current_dt = datetime.datetime(2026, 1, 5, 9)
    food.pre_act(_RESOLVE)
    meals = self._meals(make_obs, "Alice")
    self.assertLen(meals, 1)
    self.assertIn("(2 food units left at home)", meals[0])
    self.assertIn("Checking account balance: $680.00", meals[0])
    self.assertEqual(world_state.inventories["Alice"]["food_units"], 2)
    # Once per day.
    clock._current_dt = datetime.datetime(2026, 1, 5, 11)
    food.pre_act(_RESOLVE)
    self.assertLen(self._meals(make_obs, "Alice"), 1)

  def test_meal_once_per_calendar_day_across_months(self):
    profile = _employed_profile("Alice")
    profile.food_balance = 100
    food, _, make_obs = self._gm({"Alice": profile}, None)
    food.process_due(datetime.datetime(2026, 1, 8, 9))
    food.process_due(datetime.datetime(2026, 1, 8, 15))
    # Same day of the month, next month: a new day, so a new meal.
    food.process_due(datetime.datetime(2026, 2, 8, 9))
    self.assertLen(self._meals(make_obs, "Alice"), 2)
    self.assertEqual(profile.food_balance, 100 - 2 * profile.daily_food_needed)

  def test_checkpoint_restore_does_not_make_the_meal_due_again(self):
    profile = _employed_profile("Alice")
    profile.food_balance = 12
    food, _, make_obs = self._gm({"Alice": profile}, None)
    food.process_due(datetime.datetime(2026, 1, 13, 9))
    stale_checkpoint = food.get_state()
    food.process_due(datetime.datetime(2026, 1, 14, 9))
    self.assertLen(self._meals(make_obs, "Alice"), 2)
    # Restoring a checkpoint written before today's meal.
    food.set_state(stale_checkpoint)
    self.assertEqual(
        food.get_state()["last_processed_day"],
        datetime.date(2026, 1, 14).toordinal(),
    )
    food.process_due(datetime.datetime(2026, 1, 14, 15))
    self.assertLen(self._meals(make_obs, "Alice"), 2)
    # The next day is eaten normally.
    food.process_due(datetime.datetime(2026, 1, 15, 9))
    self.assertLen(self._meals(make_obs, "Alice"), 3)

  def test_world_state_failure_raises(self):
    class _BrokenWorldState(_FakeWorldState):

      def get_inventory(self, agent_id):
        raise ConnectionError("world state unavailable")

    profile = _employed_profile("Alice")
    food, _, make_obs = self._gm(
        {"Alice": profile}, _BrokenWorldState(wallets={"Alice": 100.0})
    )
    with self.assertRaisesRegex(ConnectionError, "world state unavailable"):
      food.process_due(datetime.datetime(2026, 1, 8, 9))
    self.assertEmpty(self._meals(make_obs, "Alice"))


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

  def test_purchase_composes_with_concurrent_meal(self):
    """The marketplace's old absolute pantry write could undo a meal."""
    market, profiles, agent = self._build_single_agent_market()
    del agent

    class _ConcurrentMealWorldState(_FakeWorldState):
      """Applies a morning meal mid-resolve."""

      def get_clock(self):
        return (0, 112, 0)

      def enqueue_observations(self, observations):
        super().enqueue_observations(observations)
        # Another game master eats 3 units after the marketplace read the
        # pantry and before it persisted the purchase.
        self.apply_inventory_deltas({"Alice": {"food_units": -3}})

    world_state = _ConcurrentMealWorldState(wallets={"Alice": 500.0})
    world_state.inventories["Alice"] = {"food_units": 10}
    market._world_state = world_state
    market.pre_observe(
        '[putative_event] Alice: {"action": "bid", "good": "Panera Soup and'
        ' Sandwich Meal", "price": 17.0, "qty": 2}'
    )
    market._resolve()
    # 10 - 3 (meal) + 2 (purchase). The old absolute write left 12.
    self.assertEqual(world_state.inventories["Alice"]["food_units"], 9)
    self.assertEqual(profiles["Alice"].food_balance, 9)

  def test_purchase_seeds_missing_inventory_rows(self):
    market, profiles, _ = self._build_single_agent_market()
    profiles["Alice"].food_balance = 4
    world_state = _FakeWorldState(wallets={"Alice": 500.0})
    world_state.get_clock = lambda: (0, 112, 0)
    market._world_state = world_state
    market.pre_observe(
        '[putative_event] Alice: {"action": "bid", "good": "Panera Soup and'
        ' Sandwich Meal", "price": 17.0, "qty": 2}'
    )
    market._resolve()
    # The pantry the world state did not have yet starts from the profile.
    self.assertEqual(world_state.inventories["Alice"]["food_units"], 6)

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

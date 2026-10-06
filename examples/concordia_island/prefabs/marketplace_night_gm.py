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

"""A prefab Game Master for nighttime marketplace in Concordia Island.

Supports:
1. Four action choices: BID (buy goods), SAVE (checking -> savings @ 3.5% APY),
   WITHDRAW (savings -> checking), PASS (no-op).
2. Financial status context displayed in MAKE_OBSERVATION.
3. Daily interest compounding on savings accounts.
4. Food unit tracking that connects purchases to daytime food consumption.
5. Persistent state synchronization with AgentEconomicProfile.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime as dt_mod
import json
import random
import re
from typing import Any

from absl import logging
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.contrib.components.game_master import marketplace
from concordia.environment import engine as engine_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib

from examples.concordia_island.configs import marketplace_goods
from examples.concordia_island.sim import economic_profile
from examples.concordia_island.sim import fixed_clock
from examples.concordia_island.sim import marketplace_sharding
from examples.concordia_island.sim import step_count_next_gm

Good = marketplace.Good
Order = marketplace.Order
MarketplaceAgent = marketplace.MarketplaceAgent
MarketPlace = marketplace.MarketPlace

# Refrigerator/freezer storage bound for a household pantry, in food units.
# One flat bound is used for every household, on the assumption that real
# residential fridges do not vary much in size.
# Food purchases beyond this are framed to the agent as at risk of spoiling,
# which discourages unbounded hoarding of cheap staple food.
MAX_FRIDGE_CAPACITY_UNITS = 60


def _get_goods_and_services_list() -> list[Good]:
  """Return list of Good objects from GOODS_AND_SERVICES catalog."""
  goods = []
  for category, qualities in marketplace_goods.GOODS_AND_SERVICES.items():
    for quality, items in qualities.items():
      for good_name, specs in items.items():
        goods.append(
            Good(
                category=category,
                quality=quality,
                id=good_name,
                price=float(specs.get("price", 10.0)),
                inventory=int(specs.get("inventory", 1000)),
                advert=specs.get("advert", ""),
            )
        )
  return goods


def _get_food_units_map() -> dict[str, int]:
  """Return map of good_name -> food_units for all food items."""
  food_map = {}
  for category, qualities in marketplace_goods.GOODS_AND_SERVICES.items():
    if category.lower() == "food":
      for _, items in qualities.items():
        for good_name, specs in items.items():
          food_map[good_name] = int(specs.get("food_units", 1))
  return food_map


def _normalise_good_id(text: Any) -> str:
  """Folds an item name to a comparison key.

  The catalogue is keyed on display names such as "Panera Soup and Sandwich
  Meal", but the action prompt asks for a "good" field and models frequently
  answer with a slug such as "panera_soup_and_sandwich_meal", or with different
  capitalisation or punctuation. Folding both sides through this function lets
  those orders resolve to the intended item instead of being rejected.

  Args:
    text: The item name as written by the model, or a catalogue key.

  Returns:
    A lowercase key with every run of non-alphanumeric characters collapsed to
    a single underscore, e.g. "Amy's Bean and Rice Burrito" ->
    "amy_s_bean_and_rice_burrito".
  """
  return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


class IslandMarketPlace(MarketPlace):
  """Enhanced fixed-price marketplace with savings, withdrawals, and food tracking."""

  def __init__(
      self,
      acting_player_names: Sequence[str],
      agents: Sequence[MarketplaceAgent],
      goods: Sequence[Good],
      economic_profiles: (
          dict[str, economic_profile.AgentEconomicProfile] | None
      ) = None,
      annual_interest_rate: float = 0.035,
      food_units_map: dict[str, int] | None = None,
      market_type: str = "fixed_prices",
      show_advert: bool = True,
      components: Sequence[str] = (),
      pre_act_label: str = "\nMarketplace",
      world_state: Any = None,
      clock_key: str = "clock",
      max_rounds: int = 0,
      marketplace_gm_name: str = marketplace_sharding.MARKETPLACE_GM_PREFIX,
  ):
    super().__init__(
        acting_player_names=acting_player_names,
        agents=agents,
        goods=goods,
        market_type=market_type,
        show_advert=show_advert,
        components=components,
        pre_act_label=pre_act_label,
    )
    self._profiles = economic_profiles or {}
    self._annual_interest_rate = annual_interest_rate
    self._food_units_map = food_units_map or _get_food_units_map()
    self._interest_applied_this_round: set[int] = set()
    self._interest_applied_days: set[int] = set()
    self._resolved_agents_in_round: set[str] = set()
    self._world_state = world_state
    self._order_counter = 0
    # Which marketplace shard this instance is. Order ids must carry it: the
    # counter below is per-process, so with N shards every shard would mint the
    # same ids and, because `record_order` upserts on (run id, order id), one
    # shard's sales would silently overwrite another's. An earlier fiscal run
    # lost Olivia Welch's Casio Digital Watch to this collision -- five
    # purchases appear in the agents' memories but only four rows exist in the
    # OrderBook.
    self._shard_idx = marketplace_sharding.parse_shard_index(
        marketplace_gm_name
    )
    self._pending_putative_actions: dict[str, str] = {}
    self._clock_key = clock_key
    # Money moved this round, per agent, as (checking delta, savings delta).
    # Flushed to the optional world state at the end of _resolve through
    # `WorldState.apply_wallet_deltas`. Deltas rather than absolute balances
    # because the island GM's FiscalScheduler may move the same money; see
    # `world_state_protocol.WorldState.apply_wallet_deltas`.
    self._pending_wallet_deltas: dict[str, list[float]] = {}
    # Cumulative units purchased per agent per good, across the whole run.
    # Food is consumed each day, so the live inventory does not show what an
    # agent has bought over time; this does. It is reported in the action
    # prompt as a plain list, with no commentary, so that it does not push the
    # agent toward or away from any particular item or category.
    self._purchase_history: dict[str, dict[str, int]] = {}
    # The exact call-to-action most recently sent to each agent. Kept so that
    # _resolve can guarantee it never tries to parse its own prompt back as if
    # it were the agent's answer. See _resolve for why that used to happen.
    self._last_call_to_action: dict[str, str] = {}
    # Models routinely answer with a slug ("panera_soup_and_sandwich_meal")
    # rather than the catalogue's display name ("Panera Soup and Sandwich
    # Meal"). Keep a normalised index so those orders resolve instead of being
    # rejected as unknown goods.
    self._goods_by_normalised_id: dict[str, str] = {
        _normalise_good_id(good_id): good_id for good_id in self._goods
    }
    # How many marketplace rounds each agent gets per night, and how many each
    # has actually played tonight. Used to stamp every marketplace observation
    # with "Round n/N" so the agent can tell how much of the night is left. 0
    # disables the stamp.
    self._max_rounds = max_rounds
    self._agent_round_counts: dict[str, int] = {}

  def _get_marketplace_timestamp(self) -> str:
    """Get a formatted timestamp for marketplace observations.

    Reads the clock component and returns a string like
    '// marketplace [Thursday, January 1st, 11:00 PM]: '
    so marketplace memories are chronologically anchored.

    Returns:
      A formatted timestamp prefix string.
    """
    try:
      gm = self.get_entity()
      clock = gm.get_component(self._clock_key)
      # Touch current_tick so clocks that sync lazily refresh _current_dt.
      _ = getattr(clock, "current_tick", None)
      # Unwrap ClockProxy if needed.
      dt = getattr(clock, "_current_dt", None)
      if dt is None:
        inner_clock = getattr(clock, "_clock", None)
        if inner_clock is not None:
          dt = getattr(inner_clock, "_current_dt", None)
      if dt is None:
        raise AttributeError(f"Clock {clock} has no _current_dt")

      # Marketplace represents 11:00 PM of the day that just concluded.
      # By the time this method runs, FixedIntervalClock has already skipped
      # overnight to the next morning (e.g. dt = Friday 7:00 AM).
      # Subtract one day when the clock shows morning to recover the
      # concluded evening's date.
      if dt.hour < 12:
        market_dt = dt - dt_mod.timedelta(days=1)
      else:
        market_dt = dt

      day_name = market_dt.strftime("%A")
      month_name = market_dt.strftime("%B")
      day_num = market_dt.day
      # Ordinal suffix
      if 11 <= day_num % 100 <= 13:
        suffix = "th"
      elif day_num % 10 == 1:
        suffix = "st"
      elif day_num % 10 == 2:
        suffix = "nd"
      elif day_num % 10 == 3:
        suffix = "rd"
      else:
        suffix = "th"
      return (
          f"// marketplace [{day_name}, {month_name} {day_num}{suffix}, 11:00"
          " PM]: "
      )
    except (AttributeError, KeyError, RuntimeError) as e:
      logging.warning(
          "Marketplace: failed to get timestamp from clock (%s: %s). "
          "Using fallback timestamp.",
          type(e).__name__,
          e,
      )
      # Fall back to the optional world state's clock, if one is configured.
      if self._world_state is not None:
        try:
          clock_info = self._world_state.get_clock()
          if clock_info:
            tick = clock_info[0]
            ticks_per_day = (
                getattr(clock, "ticks_per_day", 8) if "clock" in locals() else 8
            )
            if not ticks_per_day or ticks_per_day <= 0:
              ticks_per_day = 8
            day_offset = (tick - 1) // ticks_per_day if tick > 0 else 0
            sim_dt = dt_mod.datetime(2026, 1, 1) + dt_mod.timedelta(
                days=day_offset
            )
            d_name = sim_dt.strftime("%A")
            m_name = sim_dt.strftime("%B")
            d_num = sim_dt.day
            sfx = (
                "th"
                if 11 <= d_num % 100 <= 13
                else {1: "st", 2: "nd", 3: "rd"}.get(d_num % 10, "th")
            )
            return (
                f"// marketplace [{d_name}, {m_name} {d_num}{sfx}, 11:00 PM]: "
            )
        except Exception:  # pylint: disable=broad-except
          pass
      return "// marketplace [11:00 PM]: "

  def _observation_prefix(self, agent_name: str, ts_prefix: str) -> str:
    """Stamps an agent's marketplace observations with their round number.

    Rounds are counted per agent, not globally, because the asynchronous engine
    lets agents run at different speeds: at any instant one agent may be on its
    second round of the night while another is still on its first. "Round 2/5"
    therefore means "your second of five turns tonight", which is the thing the
    agent can act on.

    Args:
      agent_name: The agent the observation is addressed to.
      ts_prefix: The timestamp prefix from _get_marketplace_timestamp.

    Returns:
      The timestamp prefix, followed by "Round n/N - " when a round budget is
      configured.
    """
    if not self._max_rounds:
      return ts_prefix
    played = self._agent_round_counts.get(agent_name, 0)
    played = max(1, min(played, self._max_rounds))
    return f"{ts_prefix}Round {played}/{self._max_rounds} - "

  def _record_wallet_delta(
      self,
      agent_name: str,
      liquid: float = 0.0,
      savings: float = 0.0,
  ) -> None:
    """Notes money that has moved, to be persisted at the end of the round.

    Every site that changes `agent.cash` or `profile.savings_balance` must call
    this. The in-process numbers are what the agent is shown immediately; this
    ledger is what reaches the optional world state.

    Args:
      agent_name: Whose money moved.
      liquid: Signed change to the checking account.
      savings: Signed change to the savings account.
    """
    if not liquid and not savings:
      return
    entry = self._pending_wallet_deltas.setdefault(agent_name, [0.0, 0.0])
    entry[0] += liquid
    entry[1] += savings

  def _flush_wallet_deltas(self) -> None:
    """Commits this round's money movements and re-syncs the local copies.

    Sends deltas rather than absolute balances so that a concurrent write from
    the island GM's FiscalScheduler composes with ours instead of one of us
    silently winning. The committed values come back from the transaction and
    are copied onto the profile and the MarketplaceAgent, so the in-process
    view is re-anchored to the world state on every round.
    """
    if self._world_state is None or not self._pending_wallet_deltas:
      self._pending_wallet_deltas = {}
      return

    deltas = {
        name: (round(entry[0], 2), round(entry[1], 2))
        for name, entry in self._pending_wallet_deltas.items()
    }
    # Clear before the call: if it raises, we must not re-apply these deltas
    # on a later round on top of a write that may in fact have committed.
    self._pending_wallet_deltas = {}

    committed = self._world_state.apply_wallet_deltas(deltas)
    for name, (new_liquid, new_savings) in committed.items():
      prof = self._profiles.get(name)
      if prof is not None:
        prof.liquid_balance = new_liquid
        prof.savings_balance = new_savings
      agent = self._agents.get(name)
      if agent is not None:
        agent.cash = new_liquid
    logging.info(
        "Marketplace: committed wallet deltas for %d agents: %s",
        len(committed),
        deltas,
    )

  def pre_observe(self, observation: str) -> str:
    """Capture putative event for the resolving agent."""
    if "[putative_event]" not in observation:
      return ""
    tag_end = observation.find("[putative_event]") + len("[putative_event]")
    raw = observation[tag_end:].strip()

    # The asynchronous engine sends one entity's answer at a time, prefixed
    # with that entity's name ("Victor Tapia: ...").
    # Trust that prefix when it is present. Matching on a bare substring would
    # also hand this answer to any other player whose name happens to be
    # mentioned inside it, and _resolve would then place this player's order on
    # that player's behalf.
    for name in self._acting_player_names:
      if raw.startswith(f"{name}:"):
        self._pending_putative_actions[name] = raw
        return ""

    # Simultaneous engine: a single event string carrying every player's
    # action, so substring matching is the only option.
    for name in self._acting_player_names:
      if name in raw:
        self._pending_putative_actions[name] = raw
    return ""

  def _require_known_agent(self, agent_name: str, context: str) -> None:
    """Raises if this game master was asked to act for an agent it does not own.

    When the nighttime marketplace is sharded, each shard owns a disjoint
    slice of the roster. A shard that is handed an agent from another slice
    has been mis-routed, and the only safe response is to fail immediately:
    silently returning an empty string produces a run that completes normally
    with an empty `OrderBook`, which is exactly how an earlier run wasted
    eleven hours of compute before anyone noticed.

    Args:
      agent_name: The agent the engine is asking about.
      context: Short description of the call site, for the error message.

    Raises:
      ValueError: If `agent_name` is not in this shard's roster.
    """
    if agent_name in self._agents:
      return
    raise ValueError(
        f"Marketplace game master received {context} for unknown agent "
        f"{agent_name!r}. This shard owns {len(self._acting_player_names)} "
        f"agents: {sorted(self._acting_player_names)!r}. This means the "
        "engine routed an entity to the wrong marketplace shard -- check "
        "that the agent->shard assignment used by TimeBasedNextGM and by "
        "the shard workers both come from "
        "sim.marketplace_sharding, and that --num_marketplace_gms matches "
        "the number of marketplace workers actually launched."
    )

  def _handle_next_action_spec(self, agent_name: str) -> str:
    """Generates the multi-action action spec with ephemeral financial, inventory, and categorized catalog context."""
    self._require_known_agent(agent_name, "an action request")
    profile = self._profiles.get(agent_name)
    agent = self._agents.get(agent_name)

    # Refresh wallet and inventory from the optional world state, if
    # configured, so the prompt reflects state written by other game masters
    # (e.g. morning consumption).
    if self._world_state is not None:
      try:
        wallet = self._world_state.get_wallet(agent_name)
        if wallet is not None:
          if profile is not None:
            profile.liquid_balance = wallet
          if agent is not None:
            agent.cash = wallet
        inv = self._world_state.get_inventory(agent_name)
        if inv and profile is not None:
          if "food_units" in inv:
            profile.food_balance = inv["food_units"]
          for item_name, qty in inv.items():
            if item_name != "food_units":
              profile.inventory[item_name] = qty
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Marketplace: Failed to refresh %s from world state: %s",
            agent_name,
            e,
        )

    liquid = (
        profile.liquid_balance
        if profile is not None
        else (agent.cash if agent else 0.0)
    )
    savings = profile.savings_balance if profile is not None else 0.0
    food_bal = profile.food_balance if profile is not None else 0
    rent = profile.weekly_rent if profile is not None else 0.0

    # Format current item inventory
    inv = profile.inventory if profile is not None else {}
    if inv:
      inv_str = ", ".join(
          f"{item} ({qty})" for item, qty in sorted(inv.items())
      )
    else:
      inv_str = "(none)"

    # Refrigerator/freezer storage capacity for the household.
    if food_bal >= MAX_FRIDGE_CAPACITY_UNITS:
      fridge_context = (
          f"{food_bal} units (refrigerator is fully stocked; extra groceries"
          " risk spoiling)"
      )
    else:
      fridge_context = (
          f"{food_bal} units (storage capacity:"
          f" {food_bal}/{MAX_FRIDGE_CAPACITY_UNITS} units)"
      )

    # Cumulative purchase record. Presented as a bare list of facts, with no
    # guidance attached, so that it does not bias the agent either toward
    # repeating a purchase or toward switching to something else.
    purchases = self._purchase_history.get(agent_name, {})
    if purchases:
      purchases_str = "\n".join(
          f"  - {item}: {qty} unit{'s' if qty != 1 else ''} purchased to date"
          for item, qty in sorted(purchases.items())
      )
    else:
      purchases_str = "  - (no purchases yet)"

    status_header = (
        "[Your Financial Status & Inventory]\n"
        f"- Checking Account (Cash): ${liquid:.2f}\n"
        f"- Savings Account: ${savings:.2f} (earning 3.5% APY)\n"
        f"- Household Food Stock: {fridge_context}\n"
        f"- Wardrobe & Possessions: {inv_str}\n"
        f"- Upcoming Obligations: Rent ${rent:.2f} due Friday\n"
        f"- Your purchases so far:\n{purchases_str}\n"
    )

    # Group goods by category
    categorized_goods: dict[str, list[str]] = {}
    for g in self._goods.values():
      if g.price is not None:
        cat = getattr(g, "category", "General") or "General"
        if self._show_advert and g.advert:
          item_string = (
              f"- {g.id}: ${g.price:.2f} ({g.inventory} available) — {g.advert}"
          )
        else:
          item_string = f"- {g.id}: ${g.price:.2f} ({g.inventory} available)"
        categorized_goods.setdefault(cat, []).append(item_string)

    catalog_sections = []
    for cat, items in categorized_goods.items():
      catalog_sections.append(f"### {cat}\n" + "\n".join(items))
    catalog_text = "\n\n".join(catalog_sections)

    # Every JSON example below is deliberately *valid* JSON. An earlier version
    # used <GOOD_ID>, <price> and <int> placeholders, which json.loads cannot
    # parse. When anything accidentally fed this prompt back into the action
    # parser the result was a silent decode error reported to the agent as
    # "your order could not be read". Valid placeholders mean that failure mode
    # now surfaces as an explicit unknown-item rejection instead.
    #
    # The placeholders name no real catalogue item on purpose: naming one would
    # anchor agents on it, and the prompt is meant to stay neutral across
    # categories.
    call_to_action = f"""
{status_header}
[Available Goods & Services at Tonight's Marketplace]
{catalog_text}

What will {agent_name} do tonight in the marketplace?

Choose one of the following actions:
1. BUY an item: {{"action": "bid", "good": "ITEM NAME", "price": 0.00, "qty": 0}}
   Copy ITEM NAME exactly as it is written in the list above, set price to that
   item's listed price, and set qty to the number of units you want.
2. SAVE money (transfer checking -> savings earning 3.5% APY): {{"action": "save", "amount": 0.00}}
3. WITHDRAW money (transfer savings -> checking): {{"action": "withdraw", "amount": 0.00}}
4. PASS (no purchases or transfers tonight): {{"action": "pass"}}

Ensure total purchase spending <= checking cash (${liquid:.2f}). Return ONLY the JSON.
"""
    # Remembered so that _resolve can strip this text out of anything it is
    # about to parse as the agent's answer.
    self._last_call_to_action[agent_name] = call_to_action
    action_spec = entity_lib.free_action_spec(call_to_action=call_to_action)
    return engine_lib.action_spec_to_string(action_spec)

  def _handle_make_observation(self, agent_name: str) -> str:
    """Returns only the pending Action Resolution for the agent to become durable memory."""
    self._require_known_agent(agent_name, "an observation request")
    agent = self._agents[agent_name]
    if agent.queue:
      resolution = "\n".join(agent.queue)
      agent.queue.clear()
      return resolution

    return ""

  def _resolve(self, action_spec: entity_lib.ActionSpec | None = None) -> str:
    """Processes buy, save, withdraw, and pass actions and clears market."""
    current_round = self._state["round"]

    # 1. Accrue daily interest on savings (once per day)
    current_day = None
    try:
      gm = self.get_entity()
      clock = gm.get_component(self._clock_key)
      if clock is not None:
        _ = clock.current_tick
        if hasattr(clock, "_current_dt"):
          current_day = getattr(clock, "_current_dt").day
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    should_apply_interest = False
    if current_day is not None:
      if current_day not in self._interest_applied_days:
        self._interest_applied_days.add(current_day)
        should_apply_interest = True
        # A new day means a new night's marketplace, so each agent starts again
        # at round 1 of max_rounds.
        self._agent_round_counts.clear()
    else:
      if current_round not in self._interest_applied_this_round:
        self._interest_applied_this_round.add(current_round)
        should_apply_interest = True

    if should_apply_interest:
      daily_rate = (1 + self._annual_interest_rate) ** (1 / 365) - 1
      for name, prof in self._profiles.items():
        if prof.savings_balance > 0.0:
          interest = round(prof.savings_balance * daily_rate, 4)
          prof.savings_balance = round(prof.savings_balance + interest, 2)
          self._record_wallet_delta(name, savings=interest)
          logging.info(
              "Marketplace interest for %s: +$%f (new savings: $%f)",
              name,
              interest,
              prof.savings_balance,
          )

    # Refresh profiles from the optional world state at the start of the tick.
    if self._world_state is not None:
      try:
        for name, prof in self._profiles.items():
          balances = self._world_state.get_wallet_balances(name)
          if balances is not None:
            wallet, savings = balances
            prof.liquid_balance = wallet
            prof.savings_balance = savings
            if name in self._agents:
              self._agents[name].cash = wallet
          inv = self._world_state.get_inventory(name)
          if inv:
            if "food_units" in inv:
              prof.food_balance = inv["food_units"]
            for item_name, qty in inv.items():
              if item_name != "food_units":
                prof.inventory[item_name] = qty
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Marketplace: Failed to refresh from world state: %s", e
        )

    # 2. Extract observations / putative event
    all_states = [
        self._component_pre_act_display(key) for key in self._components
    ]
    try:
      entity = self.get_entity()
      if entity is not None and hasattr(entity, "get_act_component"):
        act_comp = entity.get_act_component()
        if act_comp is not None and hasattr(act_comp, "get_state"):
          all_states.append(str(act_comp.get_state()))
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    component_states = "\n".join([s for s in all_states if s])
    observations = [
        obs.strip()
        for obs in component_states.split("[observation]")
        if obs.strip()
    ]

    putative_event_string = ""
    for obs in reversed(observations):
      if "[putative_event]" in obs:
        putative_event_string = obs
        break

    # There is deliberately no fallback to `component_states` here.
    #
    # The engine tags the acting entity's answer with [putative_event] before
    # asking the game master to resolve, and pre_observe captures it into
    # _pending_putative_actions. An agent with no captured action is simply an
    # agent who has not acted in this resolve call, which is the normal case
    # for the asynchronous engine: it resolves once per acting entity, so
    # exactly one of the acting players has an answer and the rest must be
    # left alone.
    #
    # A previous version fell back to the game master's entire concatenated
    # component text for those agents. That text contains the call-to-action
    # the game master had just sent out, so the parser below matched the
    # example JSON inside the prompt rather than any answer, and rejected the
    # order. In an earlier run that produced 38,734 "order could not be read"
    # messages against 46 successful purchases: 99.1% of all marketplace
    # activity was discarded, and every agent was also wrongly marked as having
    # resolved, which made the round counter race ahead.

    # 3. Parse each agent's action
    base_ts_prefix = self._get_marketplace_timestamp()
    processed_agents = []
    for agent_name in self._acting_player_names:
      agent = self._agents.get(agent_name)
      if agent is None:
        continue
      profile = self._profiles.get(agent_name)

      # Sync agent.cash with profile.liquid_balance
      if profile is not None:
        agent.cash = profile.liquid_balance

      if agent_name in self._pending_putative_actions:
        target_text = self._pending_putative_actions.pop(agent_name)
      elif putative_event_string and agent_name in putative_event_string:
        target_text = putative_event_string
      else:
        continue

      # Belt and braces: whatever the source, never let the prompt we sent this
      # agent be mistaken for the answer it sent back.
      sent_prompt = self._last_call_to_action.get(agent_name)
      if sent_prompt and sent_prompt in target_text:
        target_text = target_text.replace(sent_prompt, " ")

      processed_agents.append(agent_name)
      self._resolved_agents_in_round.add(agent_name)

      # This agent has now played one more round tonight. Every message queued
      # for them below is stamped with that round number.
      self._agent_round_counts[agent_name] = (
          self._agent_round_counts.get(agent_name, 0) + 1
      )
      ts_prefix = self._observation_prefix(agent_name, base_ts_prefix)

      pattern = re.compile(
          rf"\b{re.escape(agent_name)}\b.*?(?P<json>\{{.*?\}})", re.DOTALL
      )
      match = pattern.search(target_text)
      if not match:
        # Fallback: search for any JSON in the target_text if name was prefix
        pattern_any_json = re.compile(r"(?P<json>\{.*?\})", re.DOTALL)
        match = pattern_any_json.search(target_text)
      if not match:
        agent.queue.append(
            f"{ts_prefix}{agent_name} chose not to make any purchases or"
            f" transfers tonight. Checking balance remains ${agent.cash:.2f}."
        )
        continue

      json_string = match.group("json")
      try:
        action_json = json.loads(json_string)
        act_type = str(action_json.get("action", "")).lower()

        if act_type == "save":
          raw_amount = float(action_json.get("amount", 0.0))
          transfer = max(0.0, min(raw_amount, agent.cash))
          agent.cash = round(agent.cash - transfer, 2)
          if profile is not None:
            profile.liquid_balance = agent.cash
            profile.savings_balance = round(
                profile.savings_balance + transfer, 2
            )
            sav_bal = profile.savings_balance
          else:
            sav_bal = transfer
          self._record_wallet_delta(
              agent_name, liquid=-transfer, savings=transfer
          )
          agent.queue.append(
              f"{ts_prefix}{agent_name} transferred ${transfer:.2f} into their"
              f" savings account earning 3.5% APY. Checking: ${agent.cash:.2f},"
              f" Savings: ${sav_bal:.2f}."
          )
          logging.info(
              "SAVE action: %s saved $%f (checking: $%f, savings: $%f)",
              agent_name,
              transfer,
              agent.cash,
              sav_bal,
          )

        elif act_type == "withdraw":
          raw_amount = float(action_json.get("amount", 0.0))
          avail = profile.savings_balance if profile is not None else 0.0
          transfer = max(0.0, min(raw_amount, avail))
          if profile is not None:
            profile.savings_balance = round(
                profile.savings_balance - transfer, 2
            )
            profile.liquid_balance = round(profile.liquid_balance + transfer, 2)
            agent.cash = profile.liquid_balance
            sav_bal = profile.savings_balance
          else:
            agent.cash = round(agent.cash + transfer, 2)
            sav_bal = 0.0
          self._record_wallet_delta(
              agent_name, liquid=transfer, savings=-transfer
          )
          agent.queue.append(
              f"{ts_prefix}{agent_name} withdrew ${transfer:.2f} from their"
              f" savings account. Checking: ${agent.cash:.2f}, Savings:"
              f" ${sav_bal:.2f}."
          )
          logging.info(
              "WITHDRAW action: %s withdrew $%f (checking: $%f, savings: $%f)",
              agent_name,
              transfer,
              agent.cash,
              sav_bal,
          )

        elif act_type in ("pass", "none", "idle"):
          agent.queue.append(
              f"{ts_prefix}{agent_name} chose not to make any purchases or"
              f" transfers tonight. Checking balance remains ${agent.cash:.2f}."
          )

        elif act_type in ("bid", "ask"):
          good_id = action_json.get("good")
          price = action_json.get("price")
          qty = action_json.get("qty")
          # Accept the item name in whatever shape the model wrote it. The
          # catalogue is keyed on display names, but models often answer with a
          # slug or with different capitalisation or punctuation.
          if good_id and good_id not in self._goods:
            resolved_id = self._goods_by_normalised_id.get(
                _normalise_good_id(good_id)
            )
            if resolved_id is not None:
              logging.info(
                  "Normalised good id from %s: %r -> %r",
                  agent_name,
                  good_id,
                  resolved_id,
              )
              good_id = resolved_id
          if not good_id:
            agent.queue.append(
                f"{ts_prefix}{agent_name} placed an order but did not name an"
                " item, so nothing was purchased. Checking balance remains"
                f" ${agent.cash:.2f}."
            )
            logging.info(
                "Rejected order from %s: no good named (%r)",
                agent_name,
                action_json,
            )
          elif good_id not in self._goods:
            agent.queue.append(
                f"{ts_prefix}{agent_name} asked for '{good_id}', which is not"
                " one of the items sold at this marketplace, so nothing was"
                f" purchased. Checking balance remains ${agent.cash:.2f}."
            )
            logging.info(
                "Rejected order from %s: unknown good %r", agent_name, good_id
            )
          elif not price or not qty:
            agent.queue.append(
                f"{ts_prefix}{agent_name} placed an order for {good_id} without"
                " a valid price and quantity, so nothing was purchased."
                f" Checking balance remains ${agent.cash:.2f}."
            )
            logging.info(
                "Rejected order from %s: bad price/qty %r/%r",
                agent_name,
                price,
                qty,
            )
          else:
            order = Order(
                agent_id=agent_name,
                good=self._goods[good_id],
                price=float(price),
                qty=int(qty),
                side=act_type,
                round=current_round,
            )
            self._orderbooks[good_id].append(order)

        else:
          agent.queue.append(
              f"{ts_prefix}{agent_name} submitted an action of an unrecognised"
              f" type ('{act_type}'), so nothing happened. Checking balance"
              f" remains ${agent.cash:.2f}."
          )
          logging.info(
              "Rejected action from %s: unknown action type %r",
              agent_name,
              act_type,
          )

      except (json.JSONDecodeError, ValueError, TypeError) as e:
        logging.warning("Marketplace: parse error for %s: %s", agent_name, e)
        agent.queue.append(
            f"{ts_prefix}{agent_name}'s order could not be read by the"
            " marketplace and was not processed, so nothing was purchased."
            f" Checking balance remains ${agent.cash:.2f}."
        )

    # 4. Clear the fixed-prices market
    events = ["All agents resolved actions"]
    all_completed_orders = []
    sales_made = False

    for good_id, good_item in self._goods.items():
      if good_item.price is None or good_item.inventory is None:
        continue

      fixed_price = good_item.price
      submitted = [o for o in self._orderbooks[good_id] if o.side == "bid"]
      bids = [o for o in submitted if o.price >= fixed_price]

      # Bids below the listed price are not fillable. Tell the agent why
      # instead of dropping the order silently.
      for order in submitted:
        if order.price >= fixed_price:
          continue
        order_agent = self._agents.get(order.agent_id)
        if order_agent is None:
          continue
        order_agent.queue.append(
            f"{self._observation_prefix(order.agent_id, base_ts_prefix)}{order.agent_id}"
            f" offered ${order.price:.2f} per unit for {good_id}, but it is"
            f" sold at a fixed price of ${fixed_price:.2f}, so the order was"
            " not filled."
        )

      if not bids or good_item.inventory == 0:
        for order in bids:
          order_agent = self._agents[order.agent_id]
          order_agent.queue.append(
              f"{self._observation_prefix(order.agent_id, base_ts_prefix)}{order.agent_id}"
              f" attempted to purchase {order.qty} units of {good_id}, but the"
              " order could not be fulfilled as the item was out of stock."
          )
        continue

      random.shuffle(bids)

      for bid in bids:
        if good_item.inventory == 0:
          break
        buyer = self._agents[bid.agent_id]
        buyer_profile = self._profiles.get(bid.agent_id)
        qty_to_buy = min(bid.qty, good_item.inventory)

        # Enforce the household refrigerator/freezer bound for food. Without
        # this the stated capacity was purely cosmetic and agents accumulated
        # hundreds of units of a single staple.
        if good_id in self._food_units_map and buyer_profile is not None:
          units_per_item = max(1, self._food_units_map[good_id])
          headroom = MAX_FRIDGE_CAPACITY_UNITS - buyer_profile.food_balance
          storable_qty = max(0, headroom // units_per_item)
          if storable_qty <= 0:
            buyer.queue.append(
                f"{self._observation_prefix(buyer.name, base_ts_prefix)}{buyer.name}"
                f" attempted to purchase {qty_to_buy} units of {good_id}, but"
                " the household food store is already at"
                f" {buyer_profile.food_balance} of its"
                f" {MAX_FRIDGE_CAPACITY_UNITS} unit capacity, so the order was"
                " not filled."
            )
            continue
          if qty_to_buy > storable_qty:
            buyer.queue.append(
                f"{self._observation_prefix(buyer.name, base_ts_prefix)}{buyer.name}"
                f" ordered {qty_to_buy} units of {good_id}, but only"
                f" {storable_qty} would fit in the household food store"
                f" (currently {buyer_profile.food_balance} of"
                f" {MAX_FRIDGE_CAPACITY_UNITS} units), so the order was reduced"
                f" to {storable_qty}."
            )
            qty_to_buy = storable_qty

        trade_value = fixed_price * qty_to_buy

        if buyer.cash < trade_value:
          buyer.queue.append(
              f"{self._observation_prefix(buyer.name, base_ts_prefix)}{buyer.name}"
              f" attempted to purchase {qty_to_buy} units of {good_id}"
              f" (${trade_value:.2f}), but had insufficient funds in checking"
              f" (${buyer.cash:.2f})."
          )
          continue

        if qty_to_buy > 0:
          good_item.inventory -= qty_to_buy
          buyer.cash = round(buyer.cash - trade_value, 2)
          self._record_wallet_delta(buyer.name, liquid=-trade_value)
          buyer.inventory[good_id] = (
              buyer.inventory.get(good_id, 0) + qty_to_buy
          )
          # Cumulative record shown back to the agent in later prompts. Keyed
          # by the same agent id used by `_handle_next_action_spec` so the
          # write and read keys cannot drift apart.
          history = self._purchase_history.setdefault(bid.agent_id, {})
          history[good_id] = history.get(good_id, 0) + qty_to_buy

          # Sync with buyer's economic profile
          food_units_added = 0
          is_food = good_id in self._food_units_map
          if buyer_profile is not None:
            buyer_profile.liquid_balance = buyer.cash
            # If Food item, add food units
            if is_food:
              food_units_added = self._food_units_map[good_id] * qty_to_buy
              buyer_profile.food_balance += food_units_added
              logging.info(
                  "Food purchase: %s bought %dx %s (+%d food units, new total:"
                  " %d)",
                  buyer.name,
                  qty_to_buy,
                  good_id,
                  food_units_added,
                  buyer_profile.food_balance,
              )
            else:
              buyer_profile.inventory[good_id] = (
                  buyer_profile.inventory.get(good_id, 0) + qty_to_buy
              )
              logging.info(
                  "Item purchase: %s bought %dx %s (new inventory: %s)",
                  buyer.name,
                  qty_to_buy,
                  good_id,
                  buyer_profile.inventory,
              )

          if is_food:
            pantry_stock = (
                buyer_profile.food_balance
                if buyer_profile
                else buyer.inventory.get(good_id, 0)
            )
            buyer.queue.append(
                f"{self._observation_prefix(buyer.name, base_ts_prefix)}{buyer.name}"
                f" purchased {qty_to_buy} units of {good_id} for"
                f" ${trade_value:.2f}. Checking balance is now"
                f" ${buyer.cash:.2f} and household food stock is {pantry_stock}"
                " units."
            )
          else:
            buyer.queue.append(
                f"{self._observation_prefix(buyer.name, base_ts_prefix)}{buyer.name}"
                f" purchased {qty_to_buy} units of {good_id} for"
                f" ${trade_value:.2f}. Checking balance is now"
                f" ${buyer.cash:.2f}. Added to household possessions."
            )
          all_completed_orders.append(
              f"Buyer {buyer.name} bought {qty_to_buy} of {good_id} from the"
              " market."
          )
          sales_made = True

          # Record the order in the optional world state for later analysis.
          if self._world_state is not None:
            self._order_counter += 1
            try:
              self._world_state.record_order(
                  order_id=(
                      f"mkt_s{self._shard_idx}_{current_round}_"
                      f"{self._order_counter}"
                  ),
                  agent_id=buyer.name,
                  item_name=good_id,
                  is_buy=True,
                  price=fixed_price,
                  quantity=qty_to_buy,
                  is_active=False,
              )
            except Exception as e:  # pylint: disable=broad-except
              logging.warning("Failed to record order to world state: %s", e)

    for ob in self._orderbooks.values():
      ob.clear()

    if sales_made:
      events.append("Marketplace transactions completed.")
    else:
      events.append("No purchases made this round.")

    # Enqueue new observations in the optional world state for cross-GM
    # delivery.
    if self._world_state is not None:
      obs_to_enqueue = {}
      for name in processed_agents:
        agent = self._agents.get(name)
        if agent and agent.queue:
          obs_to_enqueue[name] = list(agent.queue)
      if obs_to_enqueue:
        try:
          self._world_state.enqueue_observations(obs_to_enqueue)
          logging.info(
              "Enqueued marketplace observations to world state for %d"
              " agents: %s",
              len(obs_to_enqueue),
              list(obs_to_enqueue.keys()),
          )
        except Exception as e:  # pylint: disable=broad-except
          logging.warning("Failed to enqueue marketplace observations: %s", e)

    # Persist money as deltas, in a transaction. Not as absolute balances: the
    # island GM's FiscalScheduler writes these same rows from another job,
    # and whichever of us wrote last used to erase the other's work entirely.
    # Deliberately not wrapped in try/except -- a silently dropped purchase is
    # indistinguishable from the agent never having bought anything, and that
    # corrupts the experiment rather than degrading it.
    self._flush_wallet_deltas()

    # Inventory is still an absolute write: the marketplace is its only writer
    # during the night, and FoodConsumption only touches food_units at 7 AM.
    if self._world_state is not None:
      try:
        inventory = {}
        for name, profile in self._profiles.items():
          # Always write food_units, including 0, so that a depleted pantry is
          # not left reading as a stale non-zero value in the world state.
          agent_inv = {"food_units": max(0, profile.food_balance)}
          for item_name, qty in profile.inventory.items():
            agent_inv[item_name] = qty
          inventory[name] = agent_inv
        if inventory:
          self._world_state.upsert_inventory_batch(inventory)
          logging.info(
              "World state sync: %d inventory entries persisted.",
              len(inventory),
          )
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to sync inventory to world state: %s", e)

    # Only advance the round counter once ALL acting agents have resolved this
    # round.
    if len(self._resolved_agents_in_round) >= len(self._acting_player_names):
      self._resolved_agents_in_round.clear()
      self._state["round"] += 1
      logging.info(
          "Marketplace: all %d agents resolved round %d -> starting round %d",
          len(self._acting_player_names),
          current_round,
          self._state["round"],
      )

    return "\n".join(events)

  def get_state(self) -> entity_component.ComponentState:
    state = super().get_state()
    state["interest_applied_days"] = list(self._interest_applied_days)
    state["order_counter"] = self._order_counter
    state["profile_inventories"] = {
        name: dict(p.inventory) for name, p in self._profiles.items()
    }
    state["purchase_history"] = {
        name: dict(items) for name, items in self._purchase_history.items()
    }
    return state

  def set_state(self, state: entity_component.ComponentState) -> None:
    super().set_state(state)
    self._interest_applied_days = set(state.get("interest_applied_days", []))
    self._order_counter = state.get("order_counter", self._order_counter)
    saved_invs = state.get("profile_inventories", {})
    if isinstance(saved_invs, dict):
      for name, inv in saved_invs.items():
        if name in self._profiles and isinstance(inv, dict):
          self._profiles[name].inventory = dict(inv)
    saved_history = state.get("purchase_history", {})
    if isinstance(saved_history, dict):
      self._purchase_history = {
          str(name): {str(item): int(qty) for item, qty in items.items()}
          for name, items in saved_history.items()
          if isinstance(items, dict)
      }
    # The base class rebuilds self._goods from the restored state, so the
    # normalised lookup has to be rebuilt alongside it.
    self._goods_by_normalised_id = {
        _normalise_good_id(good_id): good_id for good_id in self._goods
    }


class _NextActingEligiblePlayers(
    entity_component.ContextComponent,
):
  """A next_acting component that supports both async and sequential engines."""

  def __init__(
      self,
      player_names: Sequence[str] = (),
      pre_act_label: str = (
          gm_components.next_acting.DEFAULT_NEXT_ACTING_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._pre_act_label = pre_act_label
    self._rr_idx = 0

  def pre_act(
      self,
      action_spec: Any,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      if not self._player_names:
        return ""
      if action_spec.options and len(action_spec.options) == 1:
        candidate = action_spec.options[0]
        return candidate if candidate in self._player_names else ""
      eligible = [
          p for p in self._player_names
          if not action_spec.options or p in action_spec.options
      ]
      if not eligible:
        return ""
      chosen = eligible[self._rr_idx % len(eligible)]
      self._rr_idx += 1
      try:
        gm = self.get_entity()
        thread_id = __import__("threading").current_thread().ident
        if hasattr(gm, "set_capture_key_for_thread"):
          gm.set_capture_key_for_thread(thread_id, chosen)
      except Exception:  # pylint: disable=broad-except
        pass
      return chosen
    return ""

  def get_state(self) -> entity_component.ComponentState:
    return {"player_names": list(self._player_names), "rr_idx": self._rr_idx}

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._player_names = list(state.get("player_names", self._player_names))
    self._rr_idx = int(state.get("rr_idx", 0))


@dataclasses.dataclass
class MarketplaceNightGameMaster(prefab_lib.Prefab):
  """A prefab Game Master for nighttime fixed-price marketplace."""

  description: str = (
      "A prefab Game Master for nighttime fixed-price marketplace."
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          "name": "marketplace_rules",
          "player_names": (),
          "island_gm_name": "island rules",
          "marketplace_gm_name": "marketplace_rules",
          "max_rounds": 5,
          "starting_cash": 200.0,
          "clock_key": "clock",
          "economic_profiles": None,
          "annual_interest_rate": 0.035,
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank | None = None,
      player_names: Sequence[str] | None = None,
      clock_key: str | None = None,
      island_gm_name: str | None = None,
      marketplace_gm_name: str | None = None,
      max_rounds: int | None = None,
      starting_cash: float | None = None,
      economic_profiles: (
          dict[str, economic_profile.AgentEconomicProfile] | None
      ) = None,
      annual_interest_rate: float | None = None,
      world_state: Any = None,
      clock: Any = None,
      **kwargs,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build the Marketplace Night Game Master entity.

    Precedence is: explicit keyword argument > `self.params` > the documented
    default. Note that `self.params` is populated by a `default_factory` that
    supplies *every* key, so the older
    `merged_params.get(key, keyword_argument)` idiom could never fall through
    to the keyword argument -- it silently discarded whatever the caller
    passed. That is what broke marketplace sharding in an earlier run: each
    shard worker passed `marketplace_gm_name='marketplace_rules 3'` and got
    back an entity named plain `'marketplace_rules'`, so every shard
    registered with the engine under the same name (making
    `hasGM('marketplace_rules 3')` false, which suppressed all nighttime
    routing) and every shard serialised to the same checkpoint path.

    Args:
      model: Language model for the game master.
      memory_bank: Optional associative memory bank.
      player_names: Agents this game master is responsible for. For a
        marketplace shard this is the shard's slice, not the whole roster.
      clock_key: Component key of the clock.
      island_gm_name: Name of the daytime game master to hand control back to.
      marketplace_gm_name: This game master's own name. Must match the name
        the worker registers with the engine, i.e. `--gm_name`.
      max_rounds: Marketplace rounds per night.
      starting_cash: Fallback cash when an agent has no economic profile.
      economic_profiles: Per-agent economic profiles.
      annual_interest_rate: Interest rate on savings balances.
      world_state: Optional `WorldState` backend; None keeps state in-process.
      clock: Clock instance.
      **kwargs: Accepted for backward compatibility.

    Returns:
      The configured marketplace game master entity.
    """
    merged_params = dict(self.params)

    def _resolve(explicit, key, default):
      """Returns the first of explicit / params[key] / default that is set."""
      if explicit is not None:
        return explicit
      from_params = merged_params.get(key)
      return default if from_params is None else from_params

    clock = clock or kwargs.get("clock") or merged_params.get("clock")
    player_names = _resolve(player_names, "player_names", ())
    island_gm_name = _resolve(island_gm_name, "island_gm_name", "island rules")
    marketplace_gm_name = _resolve(
        marketplace_gm_name, "marketplace_gm_name", "marketplace_rules"
    )
    max_rounds = _resolve(max_rounds, "max_rounds", 5)
    starting_cash = _resolve(starting_cash, "starting_cash", 200.0)
    clock_key = _resolve(clock_key, "clock_key", "clock")
    profiles = _resolve(economic_profiles, "economic_profiles", None)
    interest_rate = _resolve(
        annual_interest_rate, "annual_interest_rate", 0.035
    )
    world_state = (
        world_state
        or kwargs.get("world_state")
        or merged_params.get("world_state")
    )

    raw_memory = memory_bank or basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=None
    )

    # 1. Build Marketplace agents
    marketplace_agents = []
    for name in player_names:
      cash = starting_cash
      if profiles and name in profiles:
        cash = profiles[name].liquid_balance
      marketplace_agents.append(
          MarketplaceAgent(
              name=name,
              role="consumer",
              cash=cash,
              inventory={},
              queue=[],
          )
      )

    goods = _get_goods_and_services_list()
    food_units_map = _get_food_units_map()

    # 2. Instantiate Enhanced MarketPlace Component
    marketplace_component = IslandMarketPlace(
        acting_player_names=player_names,
        agents=marketplace_agents,
        goods=goods,
        economic_profiles=profiles,
        annual_interest_rate=interest_rate,
        food_units_map=food_units_map,
        market_type="fixed_prices",
        show_advert=True,
        components=[
            actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ],
        world_state=world_state,
        # Same budget that bounds the night in StepCountNextGM below, so the
        # "Round n/N" stamp on observations matches the number of turns an
        # agent actually gets.
        max_rounds=max_rounds,
        # Namespaces this shard's order ids so two shards writing to the same
        # OrderBook cannot overwrite each other.
        marketplace_gm_name=marketplace_gm_name,
    )

    # 3. Next acting & next GM components & terminate
    next_acting = _NextActingEligiblePlayers(player_names=player_names)
    step_count_next_gm_comp = step_count_next_gm.StepCountNextGM(
        player_names=player_names,
        island_gm_name=island_gm_name,
        instagram_gm_name=marketplace_gm_name,
        max_steps=max_rounds,
        clock_key=clock_key,
    )
    terminate = gm_components.terminate.NeverTerminate()
    terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    associative_memory = actor_components.memory.AssociativeMemory(
        memory_bank=raw_memory
    )

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=100,
    )

    components_of_game_master = {
        observation_component_key: observation,
        memory_component_key: associative_memory,
        "marketplace": marketplace_component,
        # Register the same marketplace component under the standard SwitchAct
        # dispatch keys so MAKE_OBSERVATION, NEXT_ACTION_SPEC, and RESOLVE are
        # handled by the marketplace's pre_act() rather than falling through
        # to LLM YOLO path.
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            marketplace_component
        ),
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            marketplace_component
        ),
        gm_components.event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY: (
            marketplace_component
        ),
        gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: (
            next_acting
        ),
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY: (
            step_count_next_gm_comp
        ),
        terminate_key: terminate,
    }
    if clock is not None:
      if isinstance(clock, fixed_clock.FixedIntervalClock):
        components_of_game_master[clock_key] = fixed_clock.ClockProxy(clock)
      else:
        components_of_game_master[clock_key] = clock
    component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=marketplace_gm_name,
        act_component=act_component,
        context_components=components_of_game_master,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )

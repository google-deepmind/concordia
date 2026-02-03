# Copyright 2025 DeepMind Technologies Limited.
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

"""Marketplace component for Concordia, managing agents, goods, and orders."""

import abc
import collections
import dataclasses
import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


DefaultDict = collections.defaultdict
dataclass = dataclasses.dataclass


@dataclass
class Good:
  category: str
  quality: str
  id: str
  price: Optional[float] = None
  inventory: Optional[int] = None
  advert: Optional[str] = None


@dataclass
class Order:
  agent_id: str
  good: Good
  price: float
  qty: int
  side: str  # "bid" | "ask"
  round: int = 0


@dataclass
class MarketplaceAgent:
  name: str
  role: str
  cash: float
  inventory: Dict[str, int]
  queue: List[str]


class MarketPlace(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
    abc.ABC,
):
  """Component that manages the marketplace."""

  def __init__(
      self,
      acting_player_names: Sequence[str],
      agents: Sequence[MarketplaceAgent],
      goods: Sequence[Good],
      market_type: str = "clearing_house",
      # Possible values: "clearing_house", "fixed_prices",
      # "first_price_auction", "second_price_auction"
      show_advert: bool = False,
      components: Sequence[str] = (),
      pre_act_label: str = "\nMarketplace",
      history: Optional[List[Dict[str, float]]] = None,
  ):
    super().__init__()
    self._acting_player_names = acting_player_names
    self._current_player_index = 0
    self._start_of_round = True
    self._market_type = market_type
    self._show_advert = show_advert
    if history is None:
      self.history = []
    else:
      self.history = history

    self._agents = {agent.name: agent for agent in agents}
    self._state = {"round": 0}
    self._components = components
    self._pre_act_label = pre_act_label
    self._goods = {g.id: g for g in goods}
    self._orderbooks: Dict[str, List[Order]] = {g.id: [] for g in goods}
    self.curve_history: Dict[int, Dict[str, Any]] = {}
    self.trade_history: List[Dict[str, Any]] = []
    self._processed_actions = set()

  def _log_self(self, event_type: str, details: str = ""):
    """Logs an event related to the experiment component."""
    log_entry = {
        "component": self.__class__.__name__,
        "entity": (
            self.get_entity().name
            if hasattr(self, "get_entity") and self.get_entity()
            else "UnknownEntity"
        ),
        "event": event_type,
        "details": details,
    }
    if hasattr(self, "_logging_channel") and self._logging_channel is not None:
      self._logging_channel(log_entry)
    else:
      logging.info("%s LOG: %s", self.__class__.__name__, log_entry)

  def get_pre_act_label(self) -> str:
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    return f"Current round: {self._state['round']}"

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    return (
        f"{self.get_component_pre_act_label(key)}:\n"
        f"{self.get_named_component_pre_act_value(key)}"
    )

  def _handle_make_observation(self, agent_name: str) -> str:
    """Generates an observation."""

    agent = self._agents[agent_name]
    prices = self.history[-1] if self.history else {}
    # make first part of observation the agents last action from their queue
    last_action_obs = ""
    if agent.queue:
      last_action_obs = (
          agent_name + "'s recent outcomes:\n" + "\n".join(agent.queue) + "\n"
      )
      agent.queue.clear()

    if self._market_type == "clearing_house":
      common_obs = (
          f"{last_action_obs}"
          f"Round: {self._state['round']+1} is starting\n"
          f"Prices last round: {prices}\n"
          f"Cash: {agent.cash:.2f}\n"
          f"{agent_name}'s Inventory: {agent.inventory}\n"
      )
    elif self._market_type == "fixed_prices":
      common_obs = (
          f"{last_action_obs}"
          f"Round: {self._state['round']} is starting\n"
          f"Cash: {agent.cash:.2f}\n"
          f"{agent_name}'s Inventory: {agent.inventory}\n"
      )
    else:
      raise ValueError(f"Unknown market type: {self._market_type}")

    if self._market_type == "clearing_house":
      if agent.role == "producer":
        obs = f"{common_obs}Submit your order."
      else:
        items_available = [
            other_agent.inventory
            for other_agent in self._agents.values()
            if other_agent.role == "producer"
        ]
        obs = (
            f"{common_obs}"
            f"Items available from producers: {items_available}\n"
            "Submit your bid."
        )
    elif self._market_type == "fixed_prices":
      items_available = []
      for g in self._goods.values():
        if g.price is not None:
          if self._show_advert:
            item_string = (
                f"{g.id}: ${g.price}, {g.inventory} units available. {g.advert}"
            )
          else:
            item_string = f"{g.id}: ${g.price}, {g.inventory} units available"
          items_available.append(item_string)
      obs = (
          f"{common_obs}"
          f"Items available from producers: {items_available}\n"
          "Submit your order."
      )
    else:
      obs = f"{common_obs}"

    self._log_self(
        "pre_act_return",
        f"MAKE_OBSERVATION: returning observation (len: {len(obs)}).",
    )
    return obs

  def _handle_next_action_spec(self, agent_name: str) -> str:
    """Generates the next action spec."""
    agent = self._agents[agent_name]

    if agent.role == "producer":
      call_to_action = """
      What will {name} do today in the marketplace?
      Output the answer with exactly one JSON ask to SELL some of their stock.
      Format: {{"action":"ask","good":"GOOD_ID","price":<positive float>,"qty":<1-UNITS_IN_STOCK>}}
      Return only the JSON.
      """
    elif agent.role == "consumer":
      call_to_action = """
      What will {name} do today in the marketplace?
      Create exactly one JSON bid to BUY any one good you can afford.
      Format: {{"action":"bid","good":"GOOD_ID","price":<positive float>,"qty":<positive int>}}
      Ensure price*qty ≤ cash. Return only the JSON.
      """
    else:
      call_to_action = "Error: Agent has unknown role."

    action_spec = entity_lib.free_action_spec(
        call_to_action=call_to_action,
    )
    action_spec_string = engine_lib.action_spec_to_string(action_spec)

    return action_spec_string

  def _handle_next_acting(self) -> str:
    """Determines the next player to act."""
    agent_name = self._acting_player_names[self._current_player_index]
    self._current_player_index = (self._current_player_index + 1) % len(
        self._acting_player_names
    )
    self._log_self("next_acting", f"Next to act: {agent_name}")
    return agent_name

  def _logtrade_history(
      self,
      good_id: str,
      orders: List[Order],
      original_quantities: Dict[int, float],
      order_fulfillment: Dict[int, Dict[str, Any]],
  ):
    """Logs the outcome of orders for a specific good in the current round."""
    current_round = self._state["round"]
    for order in orders:
      order_id = id(order)
      original_qty = original_quantities.get(order_id)
      if original_qty is None:
        continue

      fulfillment = order_fulfillment.get(order_id)
      transaction_occurred = False
      if fulfillment and fulfillment["filled_qty"] > 0:
        filled_qty = fulfillment["filled_qty"]
        total_value = fulfillment["total_value"]
        # Calculate average price (VWAP)
        avg_price = total_value / filled_qty
        transaction_occurred = True
        status = "Partial" if filled_qty < original_qty else "Filled"
      else:
        filled_qty, total_value, avg_price, status = 0, 0.0, math.nan, "Failed"

      log_entry = {
          "round": current_round,
          "agent": order.agent_id,
          "good": good_id,
          "side": order.side,
          "order_price": order.price,
          "order_qty": original_qty,
          "status": status,
          "transaction_occurred": transaction_occurred,
          "transaction_price_avg": avg_price,
          "transaction_qty": filled_qty,
          "transaction_value": total_value,
      }
      self.trade_history.append(log_entry)

  def _clear_auction(self, good_id: str) -> tuple[float, list[str], bool]:
    bids = [o for o in self._orderbooks[good_id] if o.side == "bid"]
    asks = [o for o in self._orderbooks[good_id] if o.side == "ask"]
    completed_orders = []

    # Initialize tracking for Trade History
    original_quantities = {id(o): o.qty for o in (bids + asks)}
    order_fulfillment = DefaultDict(
        lambda: {"filled_qty": 0, "total_value": 0.0}
    )

    # build & store step-wise supply / demand curves
    price_grid = sorted({o.price for o in self._orderbooks[good_id]})
    if price_grid:  # skip empty books
      supply_curve = [
          sum(o.qty for o in asks if o.price <= p)  # cumulative asks
          for p in price_grid
      ]
      demand_curve = [
          sum(o.qty for o in bids if o.price >= p)  # cumulative bids
          for p in price_grid
      ]

      step_idx = self._state["round"]
      if self.curve_history is not None:
        self.curve_history.setdefault(step_idx, {})[good_id] = {
            "prices": price_grid,
            "supply": supply_curve,
            "demand": demand_curve,
        }

    if not bids or not asks:
      # If there are no bids or no asks, all existing orders for this good fail.
      for order in bids + asks:
        outcome = (
            f"Your {order.side} for {order.qty} {good_id} at"
            f" ${order.price:.2f} did not result in a trade as there were no"
            " counterparties."
        )
        self._agents[order.agent_id].queue.append(outcome)
      # Log failed orders before returning
      self._logtrade_history(
          good_id, bids + asks, original_quantities, order_fulfillment
      )
      if asks:
        lowest_ask_price = min(order.price for order in asks)
        return (lowest_ask_price, completed_orders, False)
      else:
        return float("nan"), completed_orders, False

    bids.sort(key=lambda o: o.price, reverse=True)
    asks.sort(key=lambda o: o.price)

    # ---  Keep track of who successfully trades ---
    successful_traders = set()

    i = j = 0
    trade_price: float | None = None
    while i < len(bids) and j < len(asks) and bids[i].price >= asks[j].price:
      buyer = self._agents[bids[i].agent_id]
      seller = self._agents[asks[j].agent_id]
      seller_stock = seller.inventory.get(good_id, 0)

      trade_price = (bids[i].price + asks[j].price) / 2
      qty = min(bids[i].qty, asks[j].qty, seller_stock)
      trade_value = trade_price * qty

      # Check if buyer can afford the quantity they are trying to buy
      if buyer.cash < trade_value:
        outcome = (
            f"You attempted to BUY {qty} {good_id} at ${trade_price:.2f} each,"
            " but you do not have enough cash. Order failed."
        )
        buyer.queue.append(outcome)
        bids[i].qty = 0  # Mark bid as filled to prevent further attempts
        i += 1
        continue

      if qty == 0:
        if seller_stock == 0:
          j += 1  # Seller is out of stock, move to next seller
        elif bids[i].qty == 0:
          i += 1  # Buyer order filled, move to next buyer
        elif asks[j].qty == 0:
          j += 1  # Seller order filled, move to next seller
        continue
      # 1. Move goods and cash
      buyer.cash -= trade_value
      seller.cash += trade_value
      buyer.inventory[good_id] = buyer.inventory.get(good_id, 0) + qty
      seller.inventory[good_id] -= qty

      # Update order fulfillment tracking
      order_fulfillment[id(bids[i])]["filled_qty"] += qty
      order_fulfillment[id(bids[i])]["total_value"] += trade_value
      order_fulfillment[id(asks[j])]["filled_qty"] += qty
      order_fulfillment[id(asks[j])]["total_value"] += trade_value

      # 2. Create and record SUCCESS outcome observations
      buyer_outcome = (
          f"You successfully BOUGHT {qty} {good_id} at ${trade_price:.2f} each."
      )
      buyer.queue.append(buyer_outcome)
      successful_traders.add(buyer.name)

      seller_outcome = (
          f"You successfully SOLD {qty} {good_id} at ${trade_price:.2f} each."
      )
      seller.queue.append(seller_outcome)
      successful_traders.add(seller.name)

      # Add to completed orders list
      completed_orders.append(
          f"Seller {seller.name} sold {qty} units at {trade_price} of"
          f" {good_id} to Buyer {buyer.name}"
      )

      # 3. Reduce outstanding quantities
      bids[i].qty -= qty
      asks[j].qty -= qty

      # 4. Advance whichever order is fully satisfied
      if bids[i].qty == 0:
        i += 1
      if asks[j].qty == 0:
        j += 1

    # --- Log all orders (successful and unsuccessful) to history ---
    self._logtrade_history(
        good_id, bids + asks, original_quantities, order_fulfillment
    )

    # --- Log outcomes for agents whose orders were NOT filled ---
    all_orders = bids + asks
    for order in all_orders:
      # Check if agent's order had remaining quantity AND they did not succeed
      # in any trade for this good. This handles agents whose prices were not
      # competitive.
      if order.qty > 0 and order.agent_id not in successful_traders:
        outcome = (
            f"Your {order.side} for {order.qty} {good_id} at"
            f" ${order.price:.2f} did not result in a trade."
        )
        self._agents[order.agent_id].queue.append(outcome)

        # Add this agent to the set to prevent duplicate "failure" messages
        # if they had multiple unsuccessful orders for the same good.
        successful_traders.add(order.agent_id)

    if trade_price is not None:
      return (trade_price, completed_orders, True)
    else:
      # trade_price is None, find the lowest ask price
      if asks:  # asks is a list of ask Order objects
        lowest_ask_price = min(order.price for order in asks)
        return (lowest_ask_price, completed_orders, False)
      else:
        # No asks were placed
        raise ValueError("No asks were placed.")
        # return (float("nan"), completed_orders, False)

  def _clear_at_fixed_prices(self, good_id: str) -> tuple[float, list[str]]:
    """Clears the market at a pre-determined fixed price for each good."""
    good_item = self._goods[good_id]
    if good_item.price is None:
      raise ValueError(
          f"Good '{good_id}' has no price for fixed_prices market."
      )

    if good_item.inventory is None:
      return float("nan"), []

    fixed_price = good_item.price
    completed_orders = []

    # Only consider bids from consumers.
    bids = [
        o
        for o in self._orderbooks[good_id]
        if o.side == "bid" and o.price >= fixed_price
    ]

    original_quantities = {id(o): o.qty for o in bids}
    order_fulfillment = DefaultDict(
        lambda: {"filled_qty": 0, "total_value": 0.0}
    )

    if not bids or good_item.inventory == 0:
      for order in bids:
        self._agents[order.agent_id].queue.append(
            f"Your bid for {order.qty} {good_id} did not trade."
        )
      self._logtrade_history(
          good_id, bids, original_quantities, order_fulfillment
      )
      return float("nan"), []

    random.shuffle(bids)  # Randomize who gets to buy if supply is limited

    for bid in bids:
      if good_item.inventory == 0:
        break  # Market is out of stock

      buyer = self._agents[bid.agent_id]
      qty_to_buy = min(bid.qty, good_item.inventory)
      trade_value = fixed_price * qty_to_buy

      if buyer.cash < trade_value:
        buyer.queue.append(
            f"You attempted to BUY {qty_to_buy} {good_id}, but do not have"
            " enough cash."
        )
        continue

      if qty_to_buy > 0:
        # Decrement market inventory and agent cash
        good_item.inventory -= qty_to_buy
        buyer.cash -= trade_value
        buyer.inventory[good_id] = buyer.inventory.get(good_id, 0) + qty_to_buy

        # Log fulfillment
        order_fulfillment[id(bid)]["filled_qty"] += qty_to_buy
        order_fulfillment[id(bid)]["total_value"] += trade_value

        buyer.queue.append(
            f"You successfully BOUGHT {qty_to_buy} {good_id} at the fixed price"
            f" of ${fixed_price:.2f}."
        )
        completed_orders.append(
            f"Buyer {buyer.name} bought {qty_to_buy} of {good_id} from the"
            " market."
        )

    self._logtrade_history(
        good_id, bids, original_quantities, order_fulfillment
    )
    return fixed_price, completed_orders

  def _clear_highest_bid_auction(
      self, good_id: str, price_number: int
  ) -> tuple[float, list[str]]:
    """Placeholder for a first-price sealed-bid auction.

    Price number 0 is the first price, 1 is the second price.

    Args:
      good_id: The id of the good to clear the auction for.
      price_number: The price number to use for the auction.

    Returns:
      A tuple containing the trade price and a list of completed orders.
    """
    self._log_self(
        "placeholder_warning",
        f"First-price auction for {good_id} is not yet implemented.",
    )
    return float("nan"), []

  def _resolve(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Extracts all orders from the concatenated putative_event string."""
    self._log_self("resolve_called", f"action_spec: {action_spec}")

    # --- Isolate the single, massive putative_event string ---
    component_states = "\n".join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    observations = [
        obs.strip()
        for obs in component_states.split("[observation]")
        if obs.strip()
    ]

    putative_event_string = ""
    for obs in reversed(observations):
      if "[putative_event]" in obs:
        # We only care about the most recent putative_event
        putative_event_string = obs
        break

    if not putative_event_string:
      return "Error: No putative event found to resolve."

    # Iterate through agents to find their action in the string.
    for agent_name in self._acting_player_names:
      # Create a regex pattern to find the agent's name, followed by
      # any characters (non-greedy), and then capture the first JSON block.
      # This handles the narrative text between the name and the JSON.
      # pattern = re.compile(rf"{re.escape(agent_name)}.*?(?P<json>\{{.*?\}})")
      pattern = re.compile(
          rf"\b{re.escape(agent_name)}\b.*?(?P<json>\{{.*?\}})", re.DOTALL
      )
      match = pattern.search(putative_event_string)

      if not match:
        self._log_self(
            "parse_warning",
            f"Could not find an action for {agent_name} in the event string.",
        )
        continue

      json_string = match.group("json")

      try:
        action_json = json.loads(json_string)
        good_id, price, qty, side = (
            action_json.get(k) for k in ["good", "price", "qty", "action"]
        )

        if not all([
            good_id,
            good_id in self._goods,
            price is not None,
            qty is not None,
            side,
        ]):
          continue

        order = Order(
            agent_id=agent_name,
            good=self._goods[good_id],
            price=price,
            qty=qty,
            side=side,
            round=self._state["round"],
        )
        self._orderbooks[good_id].append(order)
        self._log_self(
            "order_placed",
            f"Correctly parsed and placed order for {agent_name}:"
            f" {action_json}",
        )
      except json.JSONDecodeError as e:
        self._log_self(
            "resolve_error",
            f"Failed to decode JSON for {agent_name}: {json_string},"
            f" Error: {e}",
        )

    # --- CLEAR THE MARKET ---
    events = ["All agents placed orders"]
    clearing: Dict[str, float] = {}
    all_completed_orders: List[str] = []
    sales_made = False
    for good_id in self._goods:
      if self._market_type == "clearing_house":
        trade_price, completed_orders, trade = self._clear_auction(good_id)
        if trade:
          sales_made = True
      elif self._market_type == "fixed_prices":
        trade_price, completed_orders = self._clear_at_fixed_prices(good_id)
        if completed_orders:
          sales_made = True
      elif self._market_type == "first_price_auction":
        trade_price, completed_orders = self._clear_highest_bid_auction(
            good_id, price_number=0
        )
      elif self._market_type == "second_price_auction":
        trade_price, completed_orders = self._clear_highest_bid_auction(
            good_id, price_number=1
        )
      else:
        self._log_self(
            "clear_warning",
            f"Unknown market type: {self._market_type}",
        )
        continue

      if math.isnan(trade_price) and self.history:
        # use the last known clearing price
        if good_id in self.history[-1]:
          trade_price = self.history[-1][good_id]
        else:
          trade_price = float("nan")
      clearing[good_id] = trade_price
      all_completed_orders.extend(completed_orders)
    self.history.append(clearing)
    if sales_made:
      events.append("Sales executed.")
    else:
      events.append("No sales were made.")
    events.append(f"Day {self._state['round']} prices: {clearing}")
    events.append(f"Completed orders: {all_completed_orders}")
    # 4) Reset orderbooks
    for ob in self._orderbooks.values():
      ob.clear()
    self._state["round"] += 1

    return "\n".join(events)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Prepares the action for the actor based on the current experiment state.

    Args:
        action_spec: The action specification indicating the desired output
          type.

    Returns:
        str:
            - If action_spec.output_type is NEXT_ACTION_SPEC or
            MAKE_OBSERVATION:
              - Returns a formatted string containing the experiment-specific
              output.
            - Otherwise, returns an empty string.
    """
    self._log_self(
        "pre_act_called", f"action_spec.output_type: {action_spec.output_type}"
    )

    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      # get agent name from action spec, taking the longest name if there are
      # multiple matches.
      agent_name = None
      max_len = 0
      for name in self._acting_player_names:
        if name in action_spec.call_to_action:
          if len(name) > max_len:
            agent_name = name
            max_len = len(name)

      # raise error is agent_name is none
      if agent_name is None:
        raise ValueError(f"Agent name not found in action spec: {action_spec}")

      return self._handle_make_observation(agent_name)
    elif action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      # get agent name from action spec
      agent_name = None
      for name in self._acting_player_names:
        if name in action_spec.call_to_action:
          agent_name = name

      # raise error is agent_name is none
      if agent_name is None:
        raise ValueError(f"Agent name not found in action spec: {action_spec}")

      return self._handle_next_action_spec(agent_name)
    elif action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      return self._handle_next_acting()
    elif action_spec.output_type == entity_lib.OutputType.RESOLVE:
      return self._resolve(action_spec)
    else:
      return ""

  def get_state(self) -> entity_component.ComponentState:
    """Returns the serializable state of the component."""
    # Convert agent objects to dictionaries for safe serialization.
    agents_state = {
        name: dataclasses.asdict(agent) for name, agent in self._agents.items()
    }
    # Convert order objects in orderbooks to dictionaries.
    orderbooks_state = {}
    for good_id, orders in self._orderbooks.items():
      orderbooks_state[good_id] = [
          dataclasses.asdict(order) for order in orders
      ]

    return {
        "acting_player_names": self._acting_player_names,
        "current_player_index": self._current_player_index,
        "start_of_round": self._start_of_round,
        "agents": agents_state,
        "state": self._state,
        "history": self.history,
        "curve_history": self.curve_history,
        "trade_history": self.trade_history,
        "processed_actions": list(
            self._processed_actions
        ),  # Sets must be converted to lists for JSON.
        "orderbooks": orderbooks_state,
        # _goods is static, but saving it can prevent issues
        "goods": {
            name: dataclasses.asdict(good) for name, good in self._goods.items()
        },
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component from a saved state."""
    self._acting_player_names = state["acting_player_names"]
    self._current_player_index = state["current_player_index"]
    self._start_of_round = state.get("start_of_round", True)

    # Reconstruct the Good objects first.
    goods_data = state.get("goods", {})
    if isinstance(goods_data, dict):
      self._goods = {}
      for name, good_data in goods_data.items():
        if isinstance(good_data, dict):
          self._goods[name] = Good(**good_data)
        else:
          self._log_self(
              "set_state_warning",
              f"Skipping good {name} due to unexpected data type:"
              f" {type(good_data)}",
          )
    else:
      self._goods = {}

    # Reconstruct the MarketplaceAgent objects.
    agents_data = state.get("agents", {})
    if isinstance(agents_data, dict):
      self._agents = {}
      for name, agent_data in agents_data.items():
        if isinstance(agent_data, dict):
          self._agents[name] = MarketplaceAgent(**agent_data)
        else:
          # Handle cases where agent_data is not a dict, e.g., log error
          self._log_self(
              "set_state_warning",
              f"Skipping agent {name} due to unexpected data type:"
              f" {type(agent_data)}",
          )
    else:
      self._agents = {}

    self._state = state["state"]
    history_data = state.get("history")
    if isinstance(history_data, list):
      self.history = history_data
    else:
      self.history = []  # Ensure it's a list
    curve_history_data = state.get("curve_history")
    if isinstance(curve_history_data, dict):
      self.curve_history = curve_history_data
    else:
      self.curve_history = {}  # Ensure it's a dict
    trade_history_data = state.get("trade_history")
    if isinstance(trade_history_data, list):
      self.trade_history = trade_history_data
    else:
      self.trade_history = []
    self._processed_actions = set(
        state.get("processed_actions", [])
    )  # Convert list back to a set.

    # Reconstruct the Order objects, including the nested Good objects.
    self._orderbooks = {good_id: [] for good_id in self._goods}
    saved_orderbooks = state.get("orderbooks", {})
    if isinstance(saved_orderbooks, dict):
      for good_id, orders_data in saved_orderbooks.items():
        if good_id in self._orderbooks and isinstance(orders_data, list):
          for order_data in orders_data:
            if isinstance(order_data, dict):
              # The 'good' field in the saved data is a dictionary.
              # We need to pop it and reconstruct it as a Good object.
              good_data = order_data.pop("good", None)
              if isinstance(good_data, dict):
                reconstructed_good = Good(**good_data)
                order_data["good"] = reconstructed_good
                self._orderbooks[good_id].append(Order(**order_data))

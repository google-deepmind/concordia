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

"""Food consumption and hunger component for Concordia Island."""

from collections.abc import Mapping
import datetime
import threading
from typing import Any

from absl import logging
from concordia.components import game_master as gm_components
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state
from examples.concordia_island.sim import economic_profile

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _get_datetime_from_clock(clock: Any) -> datetime.datetime | None:
  """Safely extract the current simulated datetime from either clock type."""
  if clock is None:
    return None
  current_dt = getattr(clock, "_current_dt", None)
  if current_dt is not None:
    return current_dt
  if hasattr(clock, "current_tick") and hasattr(clock, "_start_dt"):
    tick = clock.current_tick
    dt = getattr(clock, "_start_dt")
    tick_interval = getattr(clock, "_tick_interval")
    for _ in range(tick):
      dt += tick_interval
      if dt.hour >= getattr(clock, "_waking_end", 23):
        next_day = dt.date() + datetime.timedelta(days=1)
        dt = datetime.datetime(
            next_day.year,
            next_day.month,
            next_day.day,
            getattr(clock, "_waking_start", 7),
            0,
            0,
        )
    return dt
  return None


class FoodConsumptionComponent(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """GM component that tracks daily food consumption and generates hunger observations."""

  def __init__(
      self,
      profiles: dict[str, economic_profile.AgentEconomicProfile],
      clock_key: str = "clock",
      make_observation_key: str = "make_observation",
      pre_act_label: str = "\nFood Status",
      world_state: Any | None = None,
  ):
    super().__init__()
    self._profiles = profiles
    self._clock_key = clock_key
    self._make_observation_key = make_observation_key
    self._pre_act_label = pre_act_label
    self._world_state = world_state
    self._lock = threading.Lock()
    self._last_processed_day: int = -1

  def _get_clock(self):
    try:
      return self.get_entity().get_component(self._clock_key)
    except (AttributeError, KeyError):
      return None

  def _inject_observation(self, agent_name: str, text: str):
    """Queue an observation for the specific agent."""
    try:
      make_obs = self.get_entity().get_component(
          self._make_observation_key,
          type_=gm_components.make_observation.MakeObservation,
      )
      make_obs.add_to_queue(agent_name, text)
    except (AttributeError, KeyError):
      logging.warning(
          "FoodConsumption: Could not queue observation for %s: MakeObservation"
          " component not found.",
          agent_name,
      )
    logging.info("FoodConsumption for %s: %s", agent_name, text[:150])

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Evaluate daily food consumption on day transitions (at 7:00 AM)."""
    # Only fire on RESOLVE or MAKE_OBSERVATION steps
    if action_spec.output_type not in (
        entity_lib.OutputType.RESOLVE,
        entity_lib.OutputType.MAKE_OBSERVATION,
    ):
      return ""

    clock = self._get_clock()
    if clock is None:
      return ""

    dt = _get_datetime_from_clock(clock)
    if dt is None:
      return ""

    with self._lock:
      # Process food decrement at 7:00 AM of each new day
      if dt.hour == 7 and dt.day != self._last_processed_day:
        self._last_processed_day = dt.day
        day_name = DAY_NAMES[dt.weekday()]

        # Refresh profiles from the optional world state, if configured (e.g.
        # food bought at the marketplace).
        if self._world_state is not None:
          try:
            for name, profile in self._profiles.items():
              inv = self._world_state.get_inventory(name)
              if inv:
                if "food_units" in inv:
                  profile.food_balance = inv["food_units"]
                for item_name, qty in inv.items():
                  if item_name != "food_units":
                    profile.inventory[item_name] = qty
              wallet = self._world_state.get_wallet(name)
              if wallet is not None:
                profile.liquid_balance = wallet
            logging.info(
                "FoodConsumption: Refreshed %d economic profiles from world"
                " state.",
                len(self._profiles),
            )
          except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                "FoodConsumption: Failed to refresh from world state: %s", e
            )

        for name, profile in self._profiles.items():
          needed = profile.daily_food_needed
          if profile.food_balance >= needed:
            profile.food_balance -= needed
            obs = (
                f"// home [{day_name}, 7:00 AM]: {name} prepares a meal from"
                " pantry supplies."
                f" Checking account balance: ${profile.liquid_balance:.2f}."
            )
          else:
            profile.food_balance = 0
            obs = (
                f"// home [{day_name}, 7:00 AM]: {name} wakes up feeling weak"
                " and hungry."
                f" Checking account balance: ${profile.liquid_balance:.2f}."
            )
          self._inject_observation(name, obs)

        # Sync updated food balances to the optional world state.
        if self._world_state is not None:
          try:
            self._world_state.upsert_inventory_batch({
                name: {"food_units": max(0, p.food_balance)}
                for name, p in self._profiles.items()
            })
            logging.info(
                "FoodConsumption: Synced %d post-consumption food balances to"
                " world state.",
                len(self._profiles),
            )
          except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                "FoodConsumption: Failed to sync food balances to world state:"
                " %s",
                e,
            )

    return ""

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          "last_processed_day": self._last_processed_day,
          "food_balances": {
              name: p.food_balance for name, p in self._profiles.items()
          },
          "inventories": {
              name: dict(p.inventory) for name, p in self._profiles.items()
          },
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._last_processed_day = component_state.as_int(
          state, "last_processed_day", self._last_processed_day
      )
      saved_balances = state.get("food_balances")
      if isinstance(saved_balances, Mapping):
        for name, bal in saved_balances.items():
          profile = self._profiles.get(str(name))
          if profile is not None and isinstance(bal, (int, float)):
            profile.food_balance = int(bal)
      saved_invs = state.get("inventories")
      if isinstance(saved_invs, Mapping):
        for name, inv in saved_invs.items():
          profile = self._profiles.get(str(name))
          if profile is None or not isinstance(inv, Mapping):
            continue
          profile.inventory = {
              str(item): int(qty)
              for item, qty in inv.items()
              if isinstance(qty, (int, float))
          }

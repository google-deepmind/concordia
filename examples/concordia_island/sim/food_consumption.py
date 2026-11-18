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
from examples.concordia_island.sim import fiscal_scheduler

# Hour of the day from which the day's meal is processed: after the night
# marketplace, which runs between the last waking tick of one day and the
# first tick (7:00 AM) of the next.
DEFAULT_PROCESSING_HOUR = 9

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
      processing_hour: int = DEFAULT_PROCESSING_HOUR,
      fiscal_scheduler_key: str | None = None,
  ):
    """Initializes the component.

    Args:
      profiles: agent name -> economic profile (shared with FiscalScheduler).
      clock_key: Key of the GM clock component.
      make_observation_key: Key of the MakeObservation component.
      pre_act_label: Label for logging.
      world_state: Optional shared world state (see `world_state_protocol`).
      processing_hour: The day's meal is processed at the first step at or after
        this hour. Must be after the nighttime marketplace has finished; see
        `process_due`.
      fiscal_scheduler_key: Key of the GM's FiscalScheduler, if any. Its due
        events (e.g. payroll) are applied before the meal is processed. GM
        components run `pre_act` in parallel, so component order alone does not
        guarantee this.
    """
    super().__init__()
    self._processing_hour = processing_hour
    self._fiscal_scheduler_key = fiscal_scheduler_key
    self._profiles = profiles
    self._clock_key = clock_key
    self._make_observation_key = make_observation_key
    self._pre_act_label = pre_act_label
    self._world_state = world_state
    self._lock = threading.Lock()
    # `datetime.date.toordinal()` of the last calendar day whose meal was
    # processed. Keyed on the full date, not the day of the month, so that the
    # same day of the next month is not mistaken for an already-processed day.
    self._last_processed_day: int = -1
    self._hungry_streak: dict[str, int] = {}

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
    """Processes the day's meal once per day, after the night marketplace."""
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

    self.process_due(dt)
    return ""

  def process_due(self, dt: datetime.datetime) -> None:
    """Processes today's meal if it is due at `dt` and not yet done.

    The meal is processed at most once per agent per calendar day. Idempotent
    and thread-safe.

    Args:
      dt: The current simulated datetime.

    Raises:
      RuntimeError: If consumption is not committed for any agent. Also
        propagates any error from the world state: a failed refresh or write
        is not logged and skipped, because continuing on stale balances would
        tell agents the wrong thing and a lost write silently corrupts the run.
    """
    with self._lock:
      # Process the day's meal once, at the first step at or after
      # `_processing_hour` (9:00 AM by default), not at 7:00 AM. The night
      # marketplace for the previous evening is resolved when the clock
      # already reads the next morning, so a 7:00 AM meal could run before
      # that night's shopping, report a pre-shopping balance and leave food
      # bought at night uneaten until the following day.
      day_key = dt.date().toordinal()
      if dt.hour < self._processing_hour or day_key <= self._last_processed_day:
        return
      day_name = DAY_NAMES[dt.weekday()]

      # Apply today's due fiscal events first, so that a payday's balance and
      # any food credit are visible below. Errors raise.
      if self._fiscal_scheduler_key:
        fiscal = self.get_entity().get_component(
            self._fiscal_scheduler_key,
            type_=fiscal_scheduler.FiscalScheduler,
        )
        fiscal.process_due_events(dt)
      time_label = dt.strftime("%I:%M %p").lstrip("0")
      # Include the calendar date: agent memories de-duplicate identical
      # texts, and an undated line repeated a week later would be dropped.
      stamp = f"{day_name}, {dt.strftime('%B')} {dt.day}, {time_label}"

      if self._world_state is not None:
        # Refresh balances from the shared world state (food bought at the
        # marketplace, money moved by another game master).
        unseeded: dict[str, dict[str, int]] = {}
        for name, profile in self._profiles.items():
          inv = self._world_state.get_inventory(name) or {}
          if "food_units" not in inv:
            # No pantry row yet (first day): the starting pantry exists only
            # in the profile. Write it before eating through deltas, or the
            # delta starts from 0 and the agent is told it is hungry.
            unseeded[name] = {"food_units": max(0, profile.food_balance)}
          else:
            profile.food_balance = inv["food_units"]
          for item_name, qty in inv.items():
            if item_name != "food_units":
              profile.inventory[item_name] = qty
          wallet = self._world_state.get_wallet(name)
          if wallet is not None:
            profile.liquid_balance = wallet
        logging.info(
            "FoodConsumption: Refreshed %d economic profiles from world state.",
            len(self._profiles),
        )
        if unseeded:
          self._world_state.upsert_inventory_batch(unseeded)
          logging.info(
              "FoodConsumption: seeded %d starting pantries.", len(unseeded)
          )
        # Eat via an atomic delta rather than writing back an absolute count
        # computed from the copy above, so a concurrent credit or purchase
        # cannot be overwritten.
        committed = self._world_state.apply_inventory_deltas({
            name: {"food_units": -p.daily_food_needed}
            for name, p in self._profiles.items()
        })
        before_after = {
            name: items["food_units"]
            for name, items in committed.items()
            if "food_units" in items
        }
        missing = set(self._profiles) - set(before_after)
        if missing:
          raise RuntimeError(
              "FoodConsumption: consumption not committed for"
              f" {sorted(missing)}."
          )
        logging.info(
            "FoodConsumption: committed %d consumption deltas to world state.",
            len(before_after),
        )
      else:
        before_after = {}
        for name, p in self._profiles.items():
          before = p.food_balance
          before_after[name] = (before, max(0, before - p.daily_food_needed))

      for name, profile in self._profiles.items():
        needed = profile.daily_food_needed
        before, after = before_after[name]
        profile.food_balance = after
        if before >= needed:
          self._hungry_streak[name] = 0
          obs = (
              f"// home [{stamp}]: {name} prepares a meal from pantry"
              f" supplies ({profile.food_balance} food units left at home)."
              f" Checking account balance: ${profile.liquid_balance:.2f}."
          )
        else:
          streak = self._hungry_streak.get(name, 0) + 1
          self._hungry_streak[name] = streak
          if before > 0:
            # A partial pantry is eaten as a small meal and the agent is told
            # so, rather than being told there is nothing to eat while the
            # remaining units silently disappear.
            hunger = (
                f"there were only {before} of the {needed} food units the"
                " household needs today, so it ate a small meal and the"
                " pantry is now empty"
            )
          else:
            hunger = "the pantry is empty and there is nothing to eat"
          if streak > 1:
            hunger += f" (day {streak} in a row without enough food at home)"
          obs = (
              f"// home [{stamp}]: {name} wakes up feeling weak and hungry;"
              f" {hunger}."
              f" Checking account balance: ${profile.liquid_balance:.2f}."
          )
        self._inject_observation(name, obs)

      # Only now: if anything above raised, the next call retries.
      self._last_processed_day = day_key

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
          "hungry_streak": dict(self._hungry_streak),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      # Never move backwards: restoring a checkpoint written before today's
      # meal must not make the meal due again. (Checkpoints from before the
      # day was keyed on the date ordinal hold a day of the month, which is
      # always smaller than any ordinal.)
      self._last_processed_day = max(
          self._last_processed_day,
          component_state.as_int(
              state, "last_processed_day", self._last_processed_day
          ),
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
      saved_streaks = state.get("hungry_streak")
      if isinstance(saved_streaks, Mapping):
        self._hungry_streak = {
            str(name): int(n)
            for name, n in saved_streaks.items()
            if isinstance(n, (int, float))
        }

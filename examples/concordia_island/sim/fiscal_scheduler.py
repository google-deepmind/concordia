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

"""Generalized fiscal scheduler component for Concordia Island."""

import calendar
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


def _parse_date(iso_date: str) -> datetime.date:
  """Parses a FiscalEvent ISO date bound. Raises ValueError if malformed."""
  return datetime.date.fromisoformat(iso_date)


class FiscalScheduler(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Processes FiscalEvent schedules to apply macroeconomic actions and inject observations."""

  def __init__(
      self,
      fiscal_events: list[economic_profile.FiscalEvent],
      profiles: dict[str, economic_profile.AgentEconomicProfile],
      clock_key: str = "clock",
      make_observation_key: str = "make_observation",
      pre_act_label: str = "\nFiscal Events",
      world_state: Any = None,
  ):
    super().__init__()
    self._events = fiscal_events
    self._profiles = profiles
    self._clock_key = clock_key
    self._make_observation_key = make_observation_key
    self._pre_act_label = pre_act_label
    self._world_state = world_state
    self._lock = threading.Lock()

    self._simulation_day: int = 0
    self._last_seen_calendar_day: int = -1
    self._first_date: datetime.date | None = None
    self._fired_events: set[str] = set()

  def _get_clock(self):
    try:
      return self.get_entity().get_component(self._clock_key)
    except (AttributeError, KeyError):
      return None

  def _inject_observation(self, agent_name: str, text: str):
    """Queue an observation for the specific agent."""
    logging.info("FiscalScheduler for %s: %s", agent_name, text[:150])
    try:
      make_obs = self.get_entity().get_component(
          self._make_observation_key,
          type_=gm_components.make_observation.MakeObservation,
      )
      make_obs.add_to_queue(agent_name, text)
    except (AttributeError, KeyError):
      logging.warning(
          "FiscalScheduler: Could not queue observation for %s: MakeObservation"
          " component not found.",
          agent_name,
      )

  def _condition_met(
      self,
      condition: str | None,
      profile: economic_profile.AgentEconomicProfile,
  ) -> bool:
    if condition is None:
      return True
    if condition == "is_employed":
      return profile.is_employed
    if condition == "not_employed":
      return not profile.is_employed
    if condition == "fjg_enrolled":
      return profile.fjg_enrolled
    return True

  def _resolve_amount(
      self,
      event: economic_profile.FiscalEvent,
      profile: economic_profile.AgentEconomicProfile,
  ) -> float:
    if event.amount_source == "fixed":
      return float(event.amount_value or 0.0)
    if event.amount_source == "wage":
      return float(profile.weekly_wage)
    if event.amount_source == "rent":
      return float(profile.weekly_rent)
    if event.amount_source == "profile_field" and event.profile_field:
      val = getattr(profile, event.profile_field, 0.0)
      return float(val)
    return 0.0

  def _commit_deltas(
      self,
      deltas: dict[str, tuple[float, float]],
      event_name: str,
  ) -> dict[str, tuple[float, float, float, float]]:
    """Commits (checking, savings) deltas and returns before/after balances.

    `self._profiles` is this process's private copy of the world. The nighttime
    marketplace runs in *different* processes (one per shard) and is what
    actually spends the money. Nothing pushes its numbers back here, so this
    component must never compute a new balance from its own copy and write
    that back: doing so discards everything the marketplace did.

    That is not a theoretical concern. In an earlier fiscal run Tara Lee
    moved $1000 of her $2000 into savings at the marketplace, the marketplace
    persisted $1000, and the Friday rent event then debited $550 from its
    stale $2000 and wrote back $1450 -- handing her back the $1000 she had put
    away. Every payday and every rent debit was doing this to every agent.

    Sending a delta instead makes that impossible: the world state re-reads
    and writes atomically, so the movements compose. The committed values are
    copied back onto the local profiles so this process's view cannot drift.

    Args:
      deltas: agent name -> (checking delta, savings delta).
      event_name: The fiscal event being applied, for logging.

    Returns:
      agent name -> (checking before, checking after, savings before,
      savings after). Callers need the "before" values because the deltas are
      floored at zero inside the transaction, so the amount actually taken is
      only knowable from the transaction's own view.
    """
    if not deltas:
      return {}

    before = {
        name: (
            self._profiles[name].liquid_balance,
            self._profiles[name].savings_balance,
        )
        for name in deltas
        if name in self._profiles
    }

    if self._world_state is None:
      # No world state configured: all state is in-process, so there is
      # nothing to race with and the local profile *is* the source of truth.
      applied = {}
      for name, (liquid_delta, savings_delta) in deltas.items():
        profile = self._profiles.get(name)
        if profile is None:
          continue
        before_liquid, before_savings = before[name]
        after_liquid = round(max(0.0, before_liquid + liquid_delta), 2)
        after_savings = round(max(0.0, before_savings + savings_delta), 2)
        profile.liquid_balance = after_liquid
        profile.savings_balance = after_savings
        applied[name] = (
            before_liquid,
            after_liquid,
            before_savings,
            after_savings,
        )
      return applied

    # Let this raise. A fiscal event that silently fails to move money leaves
    # the run's economics wrong in a way no downstream analysis can detect.
    committed = self._world_state.apply_wallet_deltas(deltas)

    applied = {}
    for name, (after_liquid, after_savings) in committed.items():
      profile = self._profiles.get(name)
      if profile is None:
        continue
      before_liquid, before_savings = before.get(
          name, (profile.liquid_balance, profile.savings_balance)
      )
      profile.liquid_balance = after_liquid
      profile.savings_balance = after_savings
      applied[name] = (
          before_liquid,
          after_liquid,
          before_savings,
          after_savings,
      )
    logging.info(
        "FiscalScheduler: committed %d wallet deltas for event %s",
        len(applied),
        event_name,
    )
    return applied

  def _commit_debits_with_savings(
      self,
      debits: dict[str, float],
      event_name: str,
  ) -> dict[str, tuple[float, float, float, float]]:
    """Debits checking first, then savings for any shortfall. Atomic.

    Used for rent when `FiscalEvent.savings_fallback` is set. Taking rent
    from checking only put agents into arrears while they still had enough
    in savings; covering the shortfall from savings means arrears only when
    the household genuinely cannot pay.

    Args:
      debits: agent name -> amount to debit.
      event_name: The fiscal event being applied, for logging.

    Returns:
      agent name -> (checking before, checking after, savings before,
      savings after), as committed.
    """
    if not debits:
      return {}
    if self._world_state is None:
      out = {}
      for name, amount in debits.items():
        p = self._profiles[name]
        cb, sb = p.liquid_balance, p.savings_balance
        from_checking = min(amount, max(0.0, cb))
        from_savings = min(amount - from_checking, max(0.0, sb))
        p.liquid_balance = round(cb - from_checking, 2)
        p.savings_balance = round(sb - from_savings, 2)
        out[name] = (cb, p.liquid_balance, sb, p.savings_balance)
      return out
    # Let this raise: a lost debit is silent data corruption.
    committed = self._world_state.apply_debits_checking_then_savings(debits)
    for name, (_, ca, _, sa) in committed.items():
      p = self._profiles.get(name)
      if p is not None:
        p.liquid_balance = ca
        p.savings_balance = sa
    logging.info(
        "FiscalScheduler: committed %d checking-then-savings debits for %s",
        len(committed),
        event_name,
    )
    return committed

  def _commit_food_deltas(
      self,
      food_deltas: dict[str, int],
      event_name: str,
  ) -> dict[str, int]:
    """Credits food units and returns each agent's committed pantry count.

    With a world state configured, the shared pantry lives there and the
    next morning's FoodConsumption re-reads it, so a credit added only to
    this game master's in-memory profile would be lost before anyone could
    eat it. The credit is an atomic delta on the shared inventory, like
    money.

    Args:
      food_deltas: agent name -> food units to add (may be negative).
      event_name: The fiscal event being applied, for logging.

    Returns:
      agent name -> food units after the credit, as committed.

    Raises:
      RuntimeError: If a food credit is not committed for any agent.
    """
    if not food_deltas:
      return {}
    if self._world_state is None:
      out = {}
      for name, delta in food_deltas.items():
        profile = self._profiles[name]
        profile.food_balance = max(0, profile.food_balance + delta)
        out[name] = profile.food_balance
      return out
    # Let this raise: a lost food credit is the bug this method fixes.
    committed = self._world_state.apply_inventory_deltas(
        {name: {"food_units": delta} for name, delta in food_deltas.items()}
    )
    out = {
        name: items["food_units"][1]
        for name, items in committed.items()
        if "food_units" in items
    }
    missing = set(food_deltas) - set(out)
    if missing:
      raise RuntimeError(
          f"FiscalScheduler: event {event_name} food credit was not committed"
          f" for {sorted(missing)}."
      )
    logging.info(
        "FiscalScheduler: committed %d food credits for event %s",
        len(out),
        event_name,
    )
    return out

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Evaluate scheduled fiscal events for the current tick."""
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

    self.process_due_events(dt)
    return ""

  def process_due_events(self, dt: datetime.datetime) -> None:
    """Fires every event due at `dt` that has not fired yet. Idempotent.

    Game master components run `pre_act` in parallel, so a component that
    needs today's fiscal events applied first (FoodConsumption reports the
    checking balance and eats any food credited today) calls this directly
    instead of relying on component order.

    Args:
      dt: The current simulated datetime.
    """
    with self._lock:
      # Day N is counted from the calendar, not by counting the day changes
      # this process happens to observe: a day on which this game master
      # served nobody used to leave every later `start_day`/`one_time_day`
      # one day late.
      if self._first_date is None:
        self._first_date = dt.date()
      self._last_seen_calendar_day = dt.day
      self._simulation_day = (dt.date() - self._first_date).days + 1

      # 7:00 AM is tick 0, 9:00 AM is tick 1, ..., 21:00 (9PM) is tick 7
      tick_of_day = max(0, min(7, (dt.hour - 7) // 2))
      weekday = dt.weekday()
      day_name = DAY_NAMES[weekday]
      # Dated, so identical texts on different days (e.g. a weekly paycheck)
      # are not de-duplicated away in agent memory.
      date_label = f"{dt.strftime('%B')} {dt.day}"

      for event in self._events:
        event_key = (
            f"{event.name}_day{self._simulation_day}_tick{event.tick_of_day}"
        )
        if event_key in self._fired_events:
          continue

        # Check start_day threshold if specified
        if (
            event.start_day is not None
            and self._simulation_day < event.start_day
        ):
          continue

        # Calendar bounds. These are what the policy configs use, so that a
        # policy starts on a fixed calendar date (e.g. after its announcement)
        # regardless of how days are counted.
        today = dt.date()
        if event.start_date is not None and today < _parse_date(
            event.start_date
        ):
          continue
        if event.end_date is not None and today > _parse_date(event.end_date):
          continue

        # An event is due from its tick onwards, not only *at* its tick. This
        # only runs when the game master resolves a step; matching the tick
        # exactly meant an event was skipped for the whole day if no step
        # happened to fall in that one tick. `_fired_events` still guarantees
        # at most one firing per day.
        if tick_of_day < event.tick_of_day:
          continue

        # Check schedule trigger
        if event.one_time_day is not None:
          if self._simulation_day != event.one_time_day:
            continue
        elif event.monthly:
          if event.start_date is None:
            raise ValueError(
                f"FiscalEvent {event.name!r} is monthly without start_date."
            )
          anchor = _parse_date(event.start_date).day
          last_day = calendar.monthrange(today.year, today.month)[1]
          if today.day != min(anchor, last_day):
            continue
        elif event.interval_days is not None:
          if event.start_date is None:
            raise ValueError(
                f"FiscalEvent {event.name!r} sets interval_days without"
                " start_date."
            )
          elapsed = (today - _parse_date(event.start_date)).days
          if elapsed % event.interval_days != 0:
            continue
        elif event.day_of_week is not None and event.day_of_week != weekday:
          continue

        # Event triggers now!
        self._fired_events.add(event_key)
        logging.info(
            "FiscalScheduler: Firing event %s on day %d (%s), tick %d",
            event.name,
            self._simulation_day,
            day_name,
            tick_of_day,
        )

        # Express the whole event as a set of deltas and commit them in one
        # transaction, rather than mutating this process's copy of the
        # profiles and writing the result back. See `_commit_deltas`.
        liquid_savings_deltas: dict[str, tuple[float, float]] = {}
        food_deltas: dict[str, int] = {}
        fallback_debits: dict[str, float] = {}
        amounts: dict[str, float] = {}

        for name, profile in self._profiles.items():
          if not self._condition_met(event.condition, profile):
            continue

          amount = self._resolve_amount(event, profile)
          if amount <= 0.0 and event.event_type != "debit":
            continue
          amounts[name] = amount

          liquid_delta = 0.0
          savings_delta = 0.0
          if event.event_type == "credit":
            if event.target_account == "liquid":
              liquid_delta = amount
            elif event.target_account == "savings":
              savings_delta = amount
            elif event.target_account == "food":
              food_deltas[name] = int(amount)
          elif event.event_type == "debit":
            if event.target_account == "liquid" and event.savings_fallback:
              fallback_debits[name] = amount
            elif event.target_account == "liquid":
              liquid_delta = -amount
            else:
              # Historically only liquid debits were implemented; anything
              # else fell through and did nothing at all, with no observation
              # and no log line. Keep the behaviour but make it visible.
              logging.warning(
                  "FiscalScheduler: event %s debits unsupported account %r;"
                  " no money moved for %s",
                  event.name,
                  event.target_account,
                  name,
              )
          if liquid_delta or savings_delta:
            liquid_savings_deltas[name] = (liquid_delta, savings_delta)

        applied = self._commit_deltas(liquid_savings_deltas, event.name)
        applied.update(
            self._commit_debits_with_savings(fallback_debits, event.name)
        )
        food_after = self._commit_food_deltas(food_deltas, event.name)

        for name, amount in amounts.items():
          profile = self._profiles[name]

          if name in food_after:
            profile.food_balance = food_after[name]

          before_liquid, after_liquid, before_savings, after_savings = (
              applied.get(
                  name,
                  (
                      profile.liquid_balance,
                      profile.liquid_balance,
                      profile.savings_balance,
                      profile.savings_balance,
                  ),
              )
          )

          if event.event_type == "credit":
            if event.observation_template:
              self._inject_observation(
                  name,
                  event.observation_template.format(
                      name=name,
                      amount=amount,
                      balance=after_liquid,
                      savings=after_savings,
                      rent=profile.weekly_rent,
                      day_name=day_name,
                      date=date_label,
                  ),
              )

          elif event.event_type == "debit":
            if event.target_account != "liquid":
              continue
            # The delta was floored at zero inside the transaction, so the
            # amount actually taken is the difference the transaction saw --
            # not something this process can compute on its own.
            from_savings = round(before_savings - after_savings, 2)
            if not event.savings_fallback:
              from_savings = 0.0
            paid = round(before_liquid - after_liquid + from_savings, 2)
            in_arrears = paid + 1e-9 < amount
            template = event.observation_template
            if in_arrears:
              template = event.arrears_template or event.observation_template
            if template:
              text = template.format(
                  name=name,
                  amount=amount,
                  paid=paid,
                  balance=after_liquid,
                  savings=after_savings,
                  rent=profile.weekly_rent,
                  day_name=day_name,
                  date=date_label,
              )
              if from_savings > 0:
                text += (
                    f" Checking did not cover it, so ${from_savings:.2f} was"
                    f" taken from savings (savings now ${after_savings:.2f})."
                )
              self._inject_observation(name, text)

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          "simulation_day": self._simulation_day,
          "last_seen_calendar_day": self._last_seen_calendar_day,
          "first_date": (
              self._first_date.isoformat() if self._first_date else ""
          ),
          "fired_events": list(self._fired_events),
          "balances": {
              name: {
                  "liquid": p.liquid_balance,
                  "savings": p.savings_balance,
                  "food": p.food_balance,
              }
              for name, p in self._profiles.items()
          },
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._simulation_day = component_state.as_int(
          state, "simulation_day", self._simulation_day
      )
      self._last_seen_calendar_day = component_state.as_int(
          state, "last_seen_calendar_day", self._last_seen_calendar_day
      )
      first_date = state.get("first_date")
      if isinstance(first_date, str) and first_date:
        self._first_date = datetime.date.fromisoformat(first_date)
      self._fired_events |= component_state.as_str_set(state, "fired_events")
      saved_balances = state.get("balances")
      if isinstance(saved_balances, Mapping):
        for name, bals in saved_balances.items():
          profile = self._profiles.get(str(name))
          if profile is None or not isinstance(bals, Mapping):
            continue
          liquid = bals.get("liquid")
          if isinstance(liquid, (int, float)) and not isinstance(liquid, bool):
            profile.liquid_balance = float(liquid)
          savings = bals.get("savings")
          if isinstance(savings, (int, float)) and not isinstance(
              savings, bool
          ):
            profile.savings_balance = float(savings)
          # Food is a count of whole units, not money.
          food = bals.get("food")
          if isinstance(food, (int, float)) and not isinstance(food, bool):
            profile.food_balance = int(food)

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

"""Fixed-interval clock for deterministic time progression.

Unlike GenerativeClock which uses LLM calls to determine time, this clock
advances deterministically at fixed intervals. It works with any engine type
(sequential, simultaneous, async).

For async engines, the clock includes an optional round-based barrier that
prevents time from advancing until all agents have acted for the current tick.
"""

from collections.abc import Sequence
import datetime
import re
import threading
from typing import Any

from absl import logging
from concordia.components.agent import memory as memory_component
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from examples.concordia_island.sim import component_state

PUTATIVE_EVENT_TAG = '[putative_event]'


_ORDINALS = {
    1: '1st',
    2: '2nd',
    3: '3rd',
    21: '21st',
    22: '22nd',
    23: '23rd',
    31: '31st',
}

_DAYS_OF_WEEK = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]

_MONTHS = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
]


def _ordinal(n: int) -> str:
  if n in _ORDINALS:
    return _ORDINALS[n]
  if 4 <= n <= 20:
    return f'{n}th'
  suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
  return f'{n}{suffix}'


def format_sim_time(dt: datetime.datetime) -> str:
  """Format a datetime into simulation time string.

  Args:
    dt: The datetime to format.

  Returns:
    String like "Thursday, January 1st, 8:00 AM"
  """
  day_name = _DAYS_OF_WEEK[dt.weekday()]
  month_name = _MONTHS[dt.month - 1]
  day_ord = _ordinal(dt.day)
  hour = dt.hour % 12
  if hour == 0:
    hour = 12
  minute = f'{dt.minute:02d}'
  ampm = 'AM' if dt.hour < 12 else 'PM'
  return f'{day_name}, {month_name} {day_ord}, {hour}:{minute} {ampm}'


def parse_sim_time(
    time_str: str, default_waking_start: int = 7
) -> datetime.datetime:
  """Parse a time string like 'Thursday, January 1st, 9:00 PM' into datetime."""
  m = re.search(
      r'(\w+)\s+(\d+)(?:st|nd|rd|th)?,\s*(\d+):(\d+)\s*(AM|PM)',
      time_str,
      re.IGNORECASE,
  )
  if m:
    month_name, day, hour_str, minute_str, ampm = m.groups()
    month = 1
    if month_name.capitalize() in _MONTHS:
      month = _MONTHS.index(month_name.capitalize()) + 1
    hour = int(hour_str)
    if ampm.upper() == 'PM' and hour < 12:
      hour += 12
    elif ampm.upper() == 'AM' and hour == 12:
      hour = 0
    return datetime.datetime(2026, month, int(day), hour, int(minute_str))
  return datetime.datetime(2026, 1, 1, default_waking_start, 0, 0)


class FixedIntervalClock(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Deterministic clock that advances at fixed intervals.

  This component provides simulation time without any LLM calls. Time advances
  in fixed increments (default 30 minutes) and can optionally skip non-waking
  hours.

  For async engines, pass player_names to enable round-based advancement:
  the clock only advances after ALL agents have acted for the current tick.
  For sequential/simultaneous engines, omit player_names and the clock
  advances on every post_act(RESOLVE) call.

  Conversation integration: pass conversation_state_key to skip agents that
  are in active conversations. Their speech turns do not count as tick actions.
  When a conversation ends, call mark_agent_acted() for each participant.
  """

  def __init__(
      self,
      start_time: str = 'Thursday, January 1st, 7:00 AM',
      tick_interval_minutes: int = 30,
      waking_hour_start: int = 7,
      waking_hour_end: int = 23,
      player_names: Sequence[str] | None = None,
      conversation_state_key: str | None = None,
      memory_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      max_ticks: int | None = None,
      pre_act_label: str = '\nCurrent time',
      initial_locations: dict[str, str] | None = None,
      locations_key: str = 'locations',
  ):
    self._pre_act_label = pre_act_label
    self._tick_interval = datetime.timedelta(minutes=tick_interval_minutes)
    self._waking_start = waking_hour_start
    self._waking_end = waking_hour_end
    self._start_time_str = start_time
    self._conversation_state_key = conversation_state_key
    self._memory_key = memory_key
    self._max_ticks = max_ticks
    self._initial_locations = dict(initial_locations or {})
    self._locations_key = locations_key

    self._start_dt = parse_sim_time(start_time, waking_hour_start)

    self._current_tick = 0
    self._current_dt = self._start_dt
    self._time = format_sim_time(self._current_dt)
    self._in_resolve_phase: dict[str, bool] = {}
    self._player_names_list = list(player_names) if player_names else []

    self._player_names = set(player_names) if player_names else None
    self._agents_acted_this_tick: set[str] = set()
    self._nighttime_completed_days: set[int] = set()
    # Per-agent record of which agents have been routed into the nighttime
    # GM(s) for the night that precedes a given date (keyed by
    # `date.toordinal()`, so that the key never collides across months).
    self._night_entered: dict[int, set[str]] = {}
    # Per-agent record of which agents have *returned* from the nighttime
    # GM(s) to the island for a given date (same keying as `_night_entered`).
    self._night_finished: dict[int, set[str]] = {}
    # Only enforced once a nighttime GM exists (see `enable_night_gate`).
    self._night_gate_enabled = False
    self._lock = threading.RLock()
    self._condition = threading.Condition(self._lock)
    self._pending_events: dict[str, str] = {}

    ticks_per_day = (
        (waking_hour_end - waking_hour_start) * 60 // tick_interval_minutes
    )
    logging.info(
        'FixedIntervalClock: %d min/tick, waking %d:00-%d:00 (%d ticks/day)',
        tick_interval_minutes,
        waking_hour_start,
        waking_hour_end,
        ticks_per_day,
    )

  def mark_nighttime_completed(self, day: int) -> None:
    """Marks nighttime simulation as completed for the given day number."""
    with self._lock:
      self._nighttime_completed_days.add(day)

  def is_nighttime_completed(self, day: int) -> bool:
    """Returns True if nighttime simulation was completed for the given day number."""
    with self._lock:
      return day in self._nighttime_completed_days

  def current_date_ordinal(self) -> int:
    """Returns `toordinal()` of the current simulated date."""
    with self._lock:
      return self._current_dt.toordinal()

  def is_first_day(self) -> bool:
    """Returns True while the clock is still on the simulation's start date.

    There is no night before the first day, so no agent needs to visit the
    nighttime GM(s) before acting on it.
    """
    with self._lock:
      return self._current_dt.date() == self._start_dt.date()

  def mark_agent_entered_night(
      self, agent_name: str, date_ordinal: int
  ) -> None:
    """Records that `agent_name` was routed to the nighttime GM(s).

    Args:
      agent_name: The player that was switched to the nighttime GM.
      date_ordinal: `toordinal()` of the date the night leads into.
    """
    with self._lock:
      self._night_entered.setdefault(date_ordinal, set()).add(agent_name)

  def has_agent_entered_night(self, agent_name: str, date_ordinal: int) -> bool:
    """Returns True if `agent_name` has entered the night before the date."""
    with self._lock:
      return agent_name in self._night_entered.get(date_ordinal, set())

  def mark_all_agents_entered_night(self, date_ordinal: int) -> None:
    """Records every player as routed to the nighttime GM(s) for the date."""
    with self._lock:
      self._night_entered.setdefault(date_ordinal, set()).update(
          self._player_names_list
      )
      self._night_finished.setdefault(date_ordinal, set()).update(
          self._player_names_list
      )

  def mark_agent_finished_night(
      self, agent_name: str, date_ordinal: int
  ) -> None:
    """Records that `agent_name` is back on the island after the night."""
    with self._lock:
      self._night_finished.setdefault(date_ordinal, set()).add(agent_name)

  def has_agent_finished_night(
      self, agent_name: str, date_ordinal: int
  ) -> bool:
    """Returns True if `agent_name` has completed the night before the date."""
    with self._lock:
      return agent_name in self._night_finished.get(date_ordinal, set())

  def enable_night_gate(self) -> None:
    """Turns on per-agent night gating (call when a nighttime GM exists)."""
    with self._lock:
      self._night_gate_enabled = True

  def is_agent_past_night(self, agent_name: str) -> bool:
    """Whether `agent_name` may act or observe on the island today.

    True when there is no nighttime GM, on the first day (no night precedes
    it), or once the agent has been through tonight's nighttime GM(s) and
    returned to the island. Until then the agent is held: no island action,
    and its 7:00 AM observations stay queued so that, in memory, the night
    precedes the morning.

    Args:
      agent_name: The player to check.

    Returns:
      True if `agent_name` has completed the night for the current date.
    """
    with self._lock:
      if not self._night_gate_enabled:
        return True
      if self._current_dt.date() == self._start_dt.date():
        return True
      return agent_name in self._night_finished.get(
          self._current_dt.toordinal(), set()
      )

  def reached_max_ticks(self) -> bool:
    """Returns True once the run's tick limit (if any) has been reached."""
    with self._lock:
      return (
          self._max_ticks is not None and self._current_tick >= self._max_ticks
      )

  @property
  def current_tick(self) -> int:
    with self._lock:
      return self._current_tick

  def has_agent_acted(self, agent_name: str) -> bool:
    """Returns True if the agent has acted in the current tick."""
    with self._lock:
      return agent_name in self._agents_acted_this_tick

  @property
  def ticks_per_day(self) -> int:
    return (
        (self._waking_end - self._waking_start)
        * 60
        // int(self._tick_interval.total_seconds() // 60)
    )

  def _reset_all_locations_to_home(self) -> None:
    """Reset all agents' physical location in Locations component to their home."""
    if not self._initial_locations:
      return
    try:
      loc_comp = self.get_entity().get_component(self._locations_key)
      if loc_comp is not None:
        state = loc_comp.get_state()
        entity_locs = state.get('entity_locations')
        if isinstance(entity_locs, dict):
          for name, home in self._initial_locations.items():
            if home:
              entity_locs[name] = home
          loc_comp.set_state(state)
          logging.info(
              'FixedIntervalClock: 7:00 AM reset %d agents to home',
              len(self._initial_locations),
          )
    except (AttributeError, KeyError) as e:
      logging.warning(
          'FixedIntervalClock: Failed to reset locations to home: %s', e
      )

  def _advance_to_next_tick(self) -> None:
    # Defensive: end ALL active conversations on every tick advance.
    # When entities run independently without a global tick barrier,
    # conversations can outlive their tick.  This ensures no
    # conversation persists across tick boundaries.
    self._force_end_all_conversations(
        f'Tick boundary ({self._current_tick} -> {self._current_tick + 1})'
    )

    self._current_tick += 1
    self._current_dt += self._tick_interval

    if self._current_dt.hour >= self._waking_end:
      # About to skip overnight.  Force-end any lingering conversations
      # BEFORE the day boundary so the marketplace GM gets a clean slate
      # with no active conversation state.
      self._force_end_all_conversations('End of day (clock overnight skip)')
      next_day = self._current_dt.date() + datetime.timedelta(days=1)
      self._current_dt = datetime.datetime(
          next_day.year,
          next_day.month,
          next_day.day,
          self._waking_start,
          0,
          0,
      )
      self._reset_all_locations_to_home()

    self._time = format_sim_time(self._current_dt)

  def _force_end_all_conversations(self, reason: str) -> None:
    """Force-end all active conversations via the conversation state component.

    This is the authoritative enforcement point.  It fires inside the clock
    at the moment the overnight skip occurs, BEFORE any GM can observe the
    new day.  This guarantees clean phase separation between island daytime
    and marketplace nighttime regardless of GM switching timing.

    Args:
      reason: The reason string for ending conversations.
    """
    if self._conversation_state_key is None:
      return
    try:
      from examples.concordia_island.sim import conversation as async_conv  # pylint: disable=g-import-not-at-top

      conv_state = self.get_entity().get_component(
          self._conversation_state_key,
          type_=async_conv.AsyncConversationState,
      )
      conv_state.end_all_conversations(reason)
      logging.info(
          'FixedIntervalClock: force-ended all conversations (%s)', reason
      )
    except (AttributeError, KeyError, ImportError):
      pass

  def get_pre_act_label(self) -> str:
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    with self._lock:
      return self._time + '\n'

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      with self._lock:
        if (
            self._max_ticks is not None
            and self._current_tick >= self._max_ticks
        ):
          return 'Yes'
      return 'No'

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      entity_name = self._get_resolving_entity()
      if not entity_name:
        # Robust fallback: parse agent name directly from the call to action
        entity_name = self._extract_agent_from_call_to_action(
            action_spec.call_to_action
        )
      if not entity_name:
        try:
          candidate = getattr(self.get_entity(), '_active_capture_key', None)
        except RuntimeError:
          candidate = None
        if self._player_names and candidate in self._player_names:
          entity_name = candidate
      if entity_name:
        with self._lock:
          self._in_resolve_phase[entity_name] = True
      else:
        with self._lock:
          self._in_resolve_phase['__sequential__'] = True
    result = self.get_pre_act_value()
    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
    })
    return result

  def pre_observe(self, observation: str) -> str:
    if PUTATIVE_EVENT_TAG in observation:
      tag_end = observation.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
      raw = observation[tag_end:].strip()
      found_any = False
      for name in self._player_names_list:
        if re.search(rf'(^|\n){re.escape(name)}\b', raw):
          with self._lock:
            self._pending_events[name] = observation
          found_any = True
          logging.info(
              'FixedIntervalClock.pre_observe: registered %s for resolution',
              name,
          )
      if not found_any:
        with self._lock:
          self._pending_events['__unknown__'] = observation
        logging.info('FixedIntervalClock.pre_observe: registered __unknown__')
    return ''

  def _get_resolving_entity(self) -> str | None:
    with self._lock:
      if not self._pending_events:
        return None
      thread_id = threading.current_thread().ident
      active_entity_name = None
      # `_capture_key_by_thread` belongs to the concrete game master entity, not
      # to the `EntityWithComponents` interface, so it has to be read
      # defensively.
      capture_by_thread = getattr(
          self.get_entity(), '_capture_key_by_thread', None
      )
      if isinstance(capture_by_thread, dict):
        active_entity_name = capture_by_thread.get(thread_id)
      if not active_entity_name:
        active_entity_name = getattr(
            self.get_entity(), '_active_capture_key', None
        )
      if (
          active_entity_name
          and self._player_names
          and active_entity_name in self._player_names
          and active_entity_name in self._pending_events
      ):
        self._pending_events.pop(active_entity_name)
        return active_entity_name
    return None

  def _extract_agent_from_call_to_action(
      self, call_to_action: str
  ) -> str | None:
    """Extract player full name from standard call_to_action format."""
    match = re.search(
        r'(?:what will|what should)\s+([^?]+?)\s+do\s+next',
        call_to_action,
        re.IGNORECASE,
    )
    if match:
      name = match.group(1).strip()
      # Verify the parsed name is actually a registered player
      if self._player_names and name in self._player_names:
        return name
    return None

  def _is_in_conversation(self, agent_name: str) -> bool:
    """Returns whether `agent_name` is currently in a conversation.

    Args:
      agent_name: The agent name to check.

    Returns:
      True if the agent is currently in a conversation, False otherwise.
    """
    if self._conversation_state_key is None or not agent_name:
      return False
    try:
      # The conversation-state component supplies `is_in_conversation`; the
      # declared `BaseComponent` return type of `get_component` does not.
      conv_state: Any = self.get_entity().get_component(
          self._conversation_state_key
      )
      return bool(conv_state.is_in_conversation(agent_name))
    except (AttributeError, KeyError):
      return False

  def _extract_agent_from_event(self, event: str) -> str | None:
    if not self._player_names:
      return None
    for name in self._player_names:
      if name in event:
        return name
    return None

  def mark_agent_acted(self, agent_name: str) -> None:
    """Mark an agent as having acted for the current tick.

    Called by conversation components when a conversation ends to count
    both participants as having completed their tick action.

    Args:
      agent_name: The name of the agent to mark.
    """
    with self._condition:
      if self._player_names is None:
        return
      self._agents_acted_this_tick.add(agent_name)
      # Clear resolve-phase flag so post_act doesn't double-count this
      # agent for the same tick (which would bleed into the next tick
      # and cause systematic tick skipping).
      self._in_resolve_phase.pop(agent_name, None)
      if self._agents_acted_this_tick >= self._player_names:
        self._agents_acted_this_tick.clear()
        self._advance_to_next_tick()
        self._condition.notify_all()
        logging.info(
            'FixedIntervalClock: tick %d -> %s (after conversation)',
            self._current_tick,
            self._time,
        )

  def post_act(
      self,
      event: str,
  ) -> str:
    with self._condition:
      if self._player_names is not None:
        thread_id = threading.current_thread().ident
        try:
          gm = self.get_entity()
        except RuntimeError:
          gm = None
        agent = None
        if hasattr(gm, '_capture_key_by_thread'):
          agent = gm._capture_key_by_thread.get(thread_id)  # pylint: disable=protected-access
        if not agent:
          agent = getattr(gm, '_active_capture_key', None)
        if agent and agent not in self._player_names:
          agent = None  # Reject the GM's own name
        if not agent:
          agent = self._extract_agent_from_event(event)
        # Don't advance daytime clock ticks during the nighttime GMs
        # (X social feed and marketplace).
        gm_name = getattr(gm, 'name', '') or ''
        if (
            'marketplace' in gm_name
            or gm_name.startswith('x_rules')
            or 'instagram' in gm_name
            or gm_name.startswith('conversation_rules')
        ):
          return ''
        # Don't count conversation speech turns as tick-level actions.
        if agent and self._is_in_conversation(agent):
          return ''
        is_seq = self._in_resolve_phase.pop('__sequential__', False)
        if is_seq:
          for name in self._player_names_list:
            if not self._is_in_conversation(name):
              self._agents_acted_this_tick.add(name)
          self._pending_events.clear()
        else:
          # No agent could be identified for this event, so there is no
          # per-agent turn to record. Previously this fell through the lookup
          # below -- `_in_resolve_phase` is keyed by agent name, so a `None`
          # key never matched -- and returned here anyway.
          if agent is None:
            return ''
          if not self._in_resolve_phase.pop(agent, False):
            return ''
          # Skip if already registered by mark_agent_acted() (called by
          # TickGatedNextActing.post_act).  Without this guard, the clock
          # and TickGated would both add the agent, causing a double-advance.
          if agent in self._agents_acted_this_tick:
            return ''
          self._agents_acted_this_tick.add(agent)
        logging.info(
            'Clock.post_act: acted at tick %d, acted=%s, needed=%s',
            self._current_tick,
            self._agents_acted_this_tick,
            self._player_names,
        )

        if self._agents_acted_this_tick >= self._player_names:
          self._agents_acted_this_tick.clear()
          self._advance_to_next_tick()
          self._condition.notify_all()
          logging.info(
              'FixedIntervalClock: tick %d -> %s',
              self._current_tick,
              self._time,
          )
      elif self._in_resolve_phase:
        self._in_resolve_phase.clear()
        self._advance_to_next_tick()

    return ''

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'current_tick': self._current_tick,
          'time': self._time,
          'current_dt_iso': self._current_dt.isoformat(),
          'agents_acted': list(self._agents_acted_this_tick),
          'max_ticks': self._max_ticks,
          'night_entered': {
              str(k): sorted(v) for k, v in self._night_entered.items()
          },
          'night_finished': {
              str(k): sorted(v) for k, v in self._night_finished.items()
          },
          'night_gate_enabled': self._night_gate_enabled,
      }

  @staticmethod
  def _parse_night_record(raw: object, label: str) -> dict[int, set[str]]:
    """Parses a `{str(ordinal): [names]}` mapping from a checkpoint."""
    parsed: dict[int, set[str]] = {}
    if not isinstance(raw, dict):
      return parsed
    for k, names in raw.items():
      try:
        ordinal = int(str(k))
      except ValueError:
        logging.warning(
            'FixedIntervalClock.set_state: ignoring %s key %r', label, k
        )
        continue
      if isinstance(names, (list, tuple, set)):
        parsed[ordinal] = {str(n) for n in names}
    return parsed

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._current_tick = component_state.as_int(state, 'current_tick', 0)
      self._time = component_state.as_str(state, 'time', self._time)
      dt_iso = component_state.as_optional_str(state, 'current_dt_iso')
      if dt_iso:
        self._current_dt = datetime.datetime.fromisoformat(dt_iso)
      self._agents_acted_this_tick = component_state.as_str_set(
          state, 'agents_acted'
      )
      raw_entered = state.get('night_entered')
      if isinstance(raw_entered, dict):
        self._night_entered = self._parse_night_record(
            raw_entered, 'night_entered'
        )
      else:
        # Checkpoint predates per-agent night tracking: assume tonight's night
        # (if any) has already been entered by everyone, so that restoring
        # mid-day does not hold agents out of the island GM.
        self._night_entered = {
            self._current_dt.toordinal(): set(self._player_names_list)
        }
      raw_finished = state.get('night_finished')
      if isinstance(raw_finished, dict):
        self._night_finished = self._parse_night_record(
            raw_finished, 'night_finished'
        )
      else:
        # Same legacy fallback: nobody is held back after a restore.
        self._night_finished = {
            self._current_dt.toordinal(): set(self._player_names_list)
        }
      if 'night_gate_enabled' in state:
        self._night_gate_enabled = bool(state.get('night_gate_enabled'))
      # `None` here means "no tick limit", which is distinct from a limit of 0.
      if 'max_ticks' in state:
        self._max_ticks = component_state.as_optional_int(state, 'max_ticks')


class ClockProxy(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component proxy that delegates to a shared FixedIntervalClock.

  This allows secondary Game Masters (e.g. MarketplaceNightGameMaster,
  InstagramGameMaster) to be registered with their own unique Component instance
  satisfying Concordia's EntityAgent requirement (set_entity can only be called
  once)
  while sharing all clock state with the primary IslandGameMaster's clock.
  """

  def __init__(
      self,
      clock: FixedIntervalClock,
      pre_act_label: str = '\nCurrent time',
  ):
    super().__init__()
    self._clock = clock
    self._pre_act_label = pre_act_label

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    if hasattr(self._clock, 'set_entity'):
      try:
        self._clock.set_entity(entity)
      except RuntimeError:
        pass  # Entity already set on wrapped clock (e.g. single-process mode).

  def __getattr__(self, name: str) -> Any:
    return getattr(self._clock, name)

  def get_pre_act_label(self) -> str:
    return self._clock.get_pre_act_label()

  def get_pre_act_value(self) -> str:
    return self._clock.get_pre_act_value()

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      # A nighttime resolution is not a daytime turn. Forwarding it would mark
      # the agent (or, if unattributed, every player via '__sequential__') as
      # in the island resolve phase; the island GM's next post_act for that
      # agent -- e.g. its next-GM query on return -- would then count it as
      # having acted, skipping the morning ticks after each night.
      return self._clock.get_pre_act_value()
    return self._clock.pre_act(action_spec)

  def pre_observe(self, observation: str) -> str:
    # Nighttime GMs (X, marketplace) must never register daytime tick
    # resolutions. The wrapped clock's entity is the island GM, so its own
    # GM-name guard cannot tell these calls apart; drop them here instead.
    del observation
    return ''

  def post_act(self, event: str) -> str:
    # See pre_observe: nighttime actions never advance daytime ticks.
    del event
    return ''

  def get_state(self) -> entity_component.ComponentState:
    return self._clock.get_state()

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._clock.set_state(state)

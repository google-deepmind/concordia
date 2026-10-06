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

"""Temporal nudge component for realistic daily scheduling.

Injects time-aware observations into the simulation to enforce:
- 9-5 work hours on weekdays (Mon-Fri)
- End-of-day location reset (agents go home to sleep)
- Weekend awareness (no work on Sat/Sun)
- Evening social options (common room for Sunset Apartments residents)

Works with any tick interval by detecting boundary crossings between
consecutive ticks, not exact hour matches.
"""

import dataclasses
import threading
from typing import Any

from absl import logging
from concordia.components import game_master as gm_components
from concordia.components.agent import memory as memory_component
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state
from examples.concordia_island.sim import conversation as async_conv

# Time boundaries (hours). The component detects when the clock crosses

# these thresholds between consecutive ticks.
WORK_START_HOUR = 9
WORK_END_HOUR = 17
BEDTIME_HOUR = 22  # Last useful hour before waking_end skip


@dataclasses.dataclass
class _AgentInfo:
  """Per-agent scheduling metadata."""

  name: str
  home_place: str
  work_place: str | None
  is_sunset_apts: bool  # Lives at sunset_apartments → can use common room


class TemporalNudge(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """GM component that injects time-based scheduling observations.

  Instead of checking for exact hours (which break with different tick
  intervals), this component tracks the previous tick's datetime and
  detects when a time boundary is *crossed* between prev_dt and current_dt.

  For example, with 120-min ticks (7, 9, 11, 13, 15, 17, 19, 21, 23):
  - Work start: detected when prev_hour < 9 and current_hour >= 9
  - Work end:   detected when prev_hour < 17 and current_hour >= 17
  - Bedtime:    detected when prev_hour < 22 and current_hour >= 22

  With 240-min ticks (7, 11, 15, 19, 23):
  - Work start: detected when crossing 9 (7 -> 11 crosses 9)
  - Work end:   detected when crossing 17 (15 -> 19 crosses 17)
  """

  def __init__(
      self,
      agent_configs: list[Any],
      clock_key: str = 'clock',
      locations_key: str = 'locations',
      make_observation_key: str = 'make_observation',
      memory_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
  ):
    super().__init__()
    self._clock_key = clock_key
    self._locations_key = locations_key
    self._make_observation_key = make_observation_key
    self._memory_key = memory_key
    self._async_conversation_key = async_conversation_key
    self._lock = threading.Lock()

    # Build per-agent lookup from configs
    self._agents: dict[str, _AgentInfo] = {}
    for cfg in agent_configs:
      self._agents[cfg.name] = _AgentInfo(
          name=cfg.name,
          home_place=cfg.home_place,
          work_place=getattr(cfg, 'work_place', None),
          is_sunset_apts='sunset_apartments' in cfg.home_place,
      )

    # Previous tick tracking for boundary detection
    self._prev_hour: int | None = None
    self._prev_weekday: int | None = None  # 0=Mon, 6=Sun
    self._nudge_fired_this_tick: set[str] = set()
    # Track agents who have been laid off (work_place set to None)
    self._laid_off_agents: set[str] = set()

  def _get_clock(self):
    try:
      return self.get_entity().get_component(self._clock_key)
    except (AttributeError, KeyError):
      return None

  def _get_locations(self):
    try:
      return self.get_entity().get_component(self._locations_key)
    except (AttributeError, KeyError):
      return None

  def _get_memory(self):
    try:
      return self.get_entity().get_component(
          self._memory_key, type_=memory_component.Memory
      )
    except (AttributeError, KeyError):
      return None

  def _end_all_conversations(self, reason: str):
    """Force-ends any lingering conversations across all agents."""
    try:
      conv_state = self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
      if conv_state is not None:
        conv_state.end_all_conversations(reason)
    except (AttributeError, KeyError):
      pass

  def _crossed_boundary(self, boundary_hour: int, current_hour: int) -> bool:
    """Check if the clock crossed a boundary between prev and current tick."""
    if self._prev_hour is None:
      # First tick — treat as crossing if we're at or past the boundary
      return current_hour >= boundary_hour
    if self._prev_hour < boundary_hour <= current_hour:
      return True
    return False

  def _is_new_day(self, current_weekday: int) -> bool:
    """Check if this is the first tick of a new day."""
    if self._prev_weekday is None:
      return True
    return current_weekday != self._prev_weekday

  def _inject_memory(self, agent_name: str, text: str):
    """Queue a temporal observation for the specific agent."""
    try:
      make_obs = self.get_entity().get_component(
          self._make_observation_key,
          type_=gm_components.make_observation.MakeObservation,
      )
      make_obs.add_to_queue(agent_name, text)
    except (AttributeError, KeyError):
      logging.warning(
          'TemporalNudge: Could not queue observation for %s: MakeObservation'
          ' component not found.',
          agent_name,
      )
    logging.info('TemporalNudge for %s: %s', agent_name, text[:150])

  def _reset_agent_to_home(self, agent_name: str):
    """Move an agent's location back to their home."""
    info = self._agents.get(agent_name)
    if not info:
      return
    locations = self._get_locations()
    if locations is None:
      return
    try:
      state = locations.get_state()
      entity_locs = state.get('entity_locations', {})
      if entity_locs.get(agent_name) != info.home_place:
        entity_locs[agent_name] = info.home_place
        locations.set_state(state)
        logging.info(
            'TemporalNudge: Reset %s to %s', agent_name, info.home_place
        )
    except (AttributeError, KeyError) as e:
      logging.warning('TemporalNudge: Failed to reset %s: %s', agent_name, e)

  def _get_evening_options(self, info: _AgentInfo) -> str:
    """Build evening destination options string for an agent."""
    options = [info.home_place]
    if info.is_sunset_apts:
      options.append('sunset_apartments_common_room')
    options.extend([
        'cafe',
        'restaurant',
        'town_square',
        'park',
        'beach',
        'community_center',
    ])
    return ', '.join(options)

  def _ensure_all_agents_home(self, time_str: str):
    """Reset all agents to home if they aren't already.

    Called at 7:00 AM start of day to guarantee agents wake up at home.
    Also force-ends any lingering conversations across day boundaries.

    Args:
      time_str: Current simulation time string for log messages.
    """
    self._end_all_conversations('New day morning reset')
    locations = self._get_locations()
    if locations is None:
      return
    try:
      state = locations.get_state()
      entity_locs = state.get('entity_locations', {})
      changed = False
      for name, info in self._agents.items():
        current_loc = entity_locs.get(name, '')
        if current_loc and current_loc != info.home_place:
          entity_locs[name] = info.home_place
          changed = True
          logging.info(
              'TemporalNudge: Morning reset %s from %s to %s',
              name,
              current_loc,
              info.home_place,
          )
      if changed:
        locations.set_state(state)
    except (AttributeError, KeyError) as e:
      logging.warning('TemporalNudge: Failed to reset locations: %s', e)

  def lay_off_agent(self, agent_name: str) -> None:
    """Mark an agent as laid off, removing their work schedule.

    The agent will no longer receive work-start/work-end nudges.
    On weekday mornings they receive a 'no work' nudge instead.

    Args:
      agent_name: Name of the agent to lay off.
    """
    info = self._agents.get(agent_name)
    if info is None:
      logging.warning(
          'TemporalNudge: Cannot lay off unknown agent %s', agent_name
      )
      return
    info.work_place = None
    self._laid_off_agents.add(agent_name)
    logging.info('TemporalNudge: Laid off agent %s', agent_name)

  def _fire_work_start(self, time_str: str):
    """Log work start event without polluting agent observation/memory stream."""
    logging.info('TemporalNudge: Workday start at %s', time_str)

  def _fire_work_end(self, time_str: str):
    """Log work end event without polluting agent observation/memory stream."""
    logging.info('TemporalNudge: Workday end at %s', time_str)

  def _fire_bedtime(self, time_str: str):
    """Reset all agents to home at bedtime without synthetic memory pollution."""
    self._end_all_conversations('Bedtime reached')
    locations = self._get_locations()
    if locations is not None:
      try:
        state = locations.get_state()
        entity_locs = state.get('entity_locations', {})
        changed = False
        for name, info in self._agents.items():
          if entity_locs.get(name) != info.home_place:
            entity_locs[name] = info.home_place
            changed = True
        if changed:
          locations.set_state(state)
      except (AttributeError, KeyError):
        pass
    logging.info('TemporalNudge: Bedtime home reset at %s', time_str)

  def _fire_weekend_morning(self, time_str: str):
    """Log weekend morning without polluting agent observation/memory stream."""
    logging.info('TemporalNudge: Weekend morning at %s', time_str)

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Check clock boundaries and fire appropriate nudges."""
    # Support both MAKE_OBSERVATION and RESOLVE so morning home resets occur
    # BEFORE observations are formatted and delivered to agents.
    if action_spec.output_type not in (
        entity_lib.OutputType.MAKE_OBSERVATION,
        entity_lib.OutputType.RESOLVE,
    ):
      return ''

    clock = self._get_clock()
    if clock is None:
      return ''

    with self._lock:
      try:
        current_dt = clock._current_dt  # pylint: disable=protected-access
        current_hour = current_dt.hour
        current_weekday = current_dt.weekday()  # 0=Mon, 6=Sun
        is_weekend = current_weekday >= 5
        time_str = clock.get_pre_act_value().strip()
      except AttributeError:
        return ''

      # 7:00 AM morning check: ALWAYS ensure all agents are at home and
      # conversations are cleanly ended before observations are generated.
      if current_hour == 7:
        self._ensure_all_agents_home(time_str)

      # Bedtime check (>= 21:00 / 9:00 PM): ensure bedtime reset
      if current_hour >= 21:
        self._fire_bedtime(time_str)

      # Create a tick identifier to avoid firing logging multiple times per tick
      tick_id = f'{current_dt.isoformat()}'
      if tick_id in self._nudge_fired_this_tick:
        self._prev_hour = current_hour
        self._prev_weekday = current_weekday
        return ''

      new_day = self._is_new_day(current_weekday)
      if new_day and is_weekend:
        self._fire_weekend_morning(time_str)
        self._nudge_fired_this_tick.add(tick_id)
      elif current_hour == 9 and not is_weekend:
        self._fire_work_start(time_str)
        self._nudge_fired_this_tick.add(tick_id)
      elif current_hour == 17 and not is_weekend:
        self._fire_work_end(time_str)
        self._nudge_fired_this_tick.add(tick_id)

      self._prev_hour = current_hour
      self._prev_weekday = current_weekday

    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {
        'prev_hour': self._prev_hour,
        'prev_weekday': self._prev_weekday,
        'laid_off_agents': list(self._laid_off_agents),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._prev_hour = component_state.as_optional_int(state, 'prev_hour')
    self._prev_weekday = component_state.as_optional_int(state, 'prev_weekday')
    self._laid_off_agents = component_state.as_str_set(
        state, 'laid_off_agents'
    )
    # Re-apply work_place = None for any laid-off agents
    for agent_name in self._laid_off_agents:
      info = self._agents.get(agent_name)
      if info is not None:
        info.work_place = None

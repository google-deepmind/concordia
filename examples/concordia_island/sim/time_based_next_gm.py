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

"""Component helping a game master pick which game master to use next based on time."""

from collections.abc import Iterable, Mapping, Sequence
import logging
import threading
from typing import Any

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state
from examples.concordia_island.sim import conversation as async_conv
from examples.concordia_island.sim import marketplace_sharding
from examples.concordia_island.sim import social_scheduler as social_scheduler_lib


class TimeBasedNextGM(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that switches the next Game Master based on time.

  Checks the FixedIntervalClock. When the day changes (indicating the
  nighttime gap was crossed), it switches to the nighttime GM (marketplace
  or instagram). Before switching, it ends all active conversations to
  prevent conversation observations from leaking into the nighttime GM's
  action pipeline.
  """

  def __init__(
      self,
      island_gm_name: str = 'island rules',
      nighttime_gm_name: str = 'marketplace_rules',
      clock_key: str = 'clock',
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
      social_scheduler_key: str = (
          social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY
      ),
      pre_act_label: str = '\nNext Game Master',
      enable_conversation_shards: bool = False,
      num_marketplace_gms: int = 1,
      player_names: Sequence[str] = (),
      per_agent_switching: bool = True,
      **kwargs,
  ):
    """Initializes the component.

    Args:
      island_gm_name: Name of the main island GM.
      nighttime_gm_name: Name of the nighttime GM (marketplace or instagram).
        When num_marketplace_gms > 1 this is the prefix 'marketplace_rules'.
      clock_key: Key of the clock component to check time.
      async_conversation_key: Key for the AsyncConversationState component.
      social_scheduler_key: Key for the SocialScheduler component.
      pre_act_label: Label for logging.
      enable_conversation_shards: If True, routes conversing agents to shards.
      num_marketplace_gms: Number of marketplace GM shards. When > 1, entities
        are routed to 'marketplace_rules N' via deterministic round-robin.
      player_names: Ordered list of all player names for shard assignment.
      per_agent_switching: True for the async engine, where every entity runs
        its own loop and must be routed into the nighttime GM(s) individually.
        False for the sequential/simultaneous engines, which switch all
        entities at once (a leaked per-thread capture key must not be
        mistaken for a per-entity query there).
      **kwargs: For backward compatibility (e.g. instagram_gm_name).
    """
    super().__init__()
    if 'instagram_gm_name' in kwargs:
      nighttime_gm_name = kwargs['instagram_gm_name']
    self._island_gm_name = island_gm_name
    self._nighttime_gm_name = nighttime_gm_name
    self._clock_key = clock_key
    self._async_conversation_key = async_conversation_key
    self._pre_act_label = pre_act_label
    self._enable_conversation_shards = enable_conversation_shards
    self._social_scheduler_key = social_scheduler_key
    self._num_marketplace_gms = num_marketplace_gms
    self._per_agent_switching = per_agent_switching
    # Build entity -> shard mapping for marketplace routing. This MUST come
    # from marketplace_sharding so that it is identical to the slice each
    # marketplace worker computes for itself in a different process; that
    # module imposes a canonical ordering precisely so the two cannot drift.
    self._entity_to_marketplace_gm: dict[str, str] = dict(
        marketplace_sharding.assign_shard_map(
            player_names, num_marketplace_gms
        )
    )
    self._lock = threading.Lock()
    self._last_seen_day = -1
    self._switched_entities_by_day: dict[int, set[str]] = {}

  def _end_all_conversations(self, reason: str) -> None:
    """End all active conversations before switching GMs."""
    try:
      conv_state = self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
      conv_state.end_all_conversations(reason)
    except (AttributeError, KeyError):
      # No conversation state component — nothing to clean up.
      pass

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      try:
        # Declared as BaseComponent; the concrete clock type is resolved
        # dynamically, hence the explicit Any.
        clock: Any = self.get_entity().get_component(self._clock_key)
        if hasattr(clock, 'current_tick'):
          _ = clock.current_tick
        elif hasattr(clock, 'get_pre_act_value'):
          clock.get_pre_act_value()
        current_day = clock._current_dt.day  # pylint: disable=protected-access
      except (AttributeError, KeyError):
        return self._island_gm_name

      if self._social_scheduler_key:
        try:
          social_scheduler = self.get_entity().get_component(
              self._social_scheduler_key,
              type_=social_scheduler_lib.SocialScheduler,
          )
          if social_scheduler is not None and hasattr(
              social_scheduler, 'fire_pending_events_for_current_tick'
          ):
            social_scheduler.fire_pending_events_for_current_tick()
        except (AttributeError, KeyError):
          pass

      # Identify the requesting entity.
      requesting_entity = ''
      tag = getattr(action_spec, 'tag', '') or ''
      if isinstance(tag, str) and ':' in tag:
        requesting_entity = tag.split(':', 1)[1]
      if not requesting_entity:
        try:
          gm = self.get_entity()
          thread_id = __import__('threading').current_thread().ident
          if hasattr(gm, 'get_capture_key_for_thread'):
            key = gm.get_capture_key_for_thread(thread_id)
            if isinstance(key, str) and key and key != gm.name:
              requesting_entity = key
          if not requesting_entity and hasattr(gm, '_capture_key_by_thread'):
            key = gm._capture_key_by_thread.get(thread_id)  # pylint: disable=protected-access
            if isinstance(key, str) and key and key != gm.name:
              requesting_entity = key
          if not requesting_entity and hasattr(gm, '_active_capture_key'):
            key = gm._active_capture_key  # pylint: disable=protected-access
            if isinstance(key, str) and key and key != gm.name:
              requesting_entity = key
        except Exception:  # pylint: disable=broad-except
          pass

      with self._lock:
        per_agent_clock = (
            self._per_agent_switching
            and requesting_entity
            and hasattr(clock, 'mark_agent_entered_night')
        )
        if per_agent_clock:
          # Per-agent path (async engine). Every agent is routed into the
          # nighttime GM exactly once per night, *before* its first action of
          # the new day. The record lives on the shared clock so that
          # TickGatedNextActing can hold the agent out of the island GM until
          # it has been through the night (otherwise a race lets an agent
          # take its 7 AM island action first). Keyed by date ordinal, since
          # day-of-month collides after a month.
          if self._last_seen_day == -1:
            self._last_seen_day = current_day
          ordinal = clock.current_date_ordinal()
          already_entered = clock.has_agent_entered_night(
              requesting_entity, ordinal
          )
          if (
              clock.is_first_day()
              or already_entered
              or (
                  hasattr(clock, 'reached_max_ticks')
                  and clock.reached_max_ticks()
              )
          ):
            # No night before day one, one night per agent per date, and no
            # night once the run has reached its tick limit (the island GM's
            # terminator ends the run instead of starting a partial night).
            result = self._island_gm_name
            if not clock.is_first_day() and hasattr(
                clock, 'mark_agent_finished_night'
            ):
              # Either the nighttime GM(s) handed this agent back (the island
              # GM is asked again only after the night) or there is no night
              # to run. Release the island's per-agent gates (action + queued
              # 7:00 AM observations) so the morning lands *after* the night
              # in the agent's memory and nobody is ever held with no night.
              clock.mark_agent_finished_night(requesting_entity, ordinal)
          else:
            clock.mark_agent_entered_night(requesting_entity, ordinal)
            self._switched_entities_by_day.setdefault(ordinal, set()).add(
                requesting_entity
            )
            target_gm = self._nighttime_gm_name
            if (
                target_gm.startswith(marketplace_sharding.MARKETPLACE_GM_PREFIX)
                and requesting_entity in self._entity_to_marketplace_gm
            ):
              target_gm = self._entity_to_marketplace_gm[requesting_entity]
            self._end_all_conversations(f'GM switch to {target_gm}')
            logging.info(
                'TimeBasedNextGM: switching %s to %s (day %d -> %d)',
                requesting_entity,
                target_gm,
                self._last_seen_day,
                current_day,
            )
            self._last_seen_day = current_day
            result = target_gm
        elif self._last_seen_day == -1:
          # First call: record the starting day and stay on island GM.
          # Do NOT switch to marketplace on startup.
          self._last_seen_day = current_day
          result = self._island_gm_name
        elif current_day != self._last_seen_day:
          # Legacy path for unattributed (engine-level) queries. Day changed —
          # the 11 PM tick completed and clock advanced.
          nighttime_done = False
          if hasattr(clock, 'is_nighttime_completed'):
            nighttime_done = clock.is_nighttime_completed(
                self._last_seen_day
            ) or clock.is_nighttime_completed(current_day)
          if nighttime_done:
            # Nighttime already ran for this night — stay on island GM.
            self._last_seen_day = current_day
            result = self._island_gm_name
          else:
            self._end_all_conversations(
                f'GM switch to {self._nighttime_gm_name}'
            )
            if hasattr(clock, 'mark_all_agents_entered_night'):
              # An engine-level switch routes every player at once.
              clock.mark_all_agents_entered_night(clock.current_date_ordinal())
            logging.info(
                'TimeBasedNextGM: switching %s to %s (day %d -> %d)',
                requesting_entity or 'entity',
                self._nighttime_gm_name,
                self._last_seen_day,
                current_day,
            )
            result = self._nighttime_gm_name
        else:
          # Same day — stay on island GM.
          result = self._island_gm_name

        # If not switching to nighttime GM, check if entity is in an active
        # conversation that should route to a dedicated conversation GM shard.
        if result == self._island_gm_name and self._enable_conversation_shards:
          if requesting_entity:
            if self._social_scheduler_key:
              try:
                social_sched = self.get_entity().get_component(
                    self._social_scheduler_key,
                    type_=social_scheduler_lib.SocialScheduler,
                )
                if social_sched is not None and hasattr(
                    social_sched, 'wait_for_in_flight_event'
                ):
                  social_sched.wait_for_in_flight_event(requesting_entity)
              except (AttributeError, KeyError):
                pass

            try:
              conv_state = self.get_entity().get_component(
                  self._async_conversation_key,
                  type_=async_conv.AsyncConversationState,
              )
              if conv_state is not None:
                if hasattr(conv_state, '_get_active_conv_for_player'):
                  # pylint: disable=protected-access
                  active_conv = conv_state._get_active_conv_for_player(
                      requesting_entity
                  )
                  # pylint: enable=protected-access
                  if active_conv and active_conv.get('assigned_gm'):
                    result = active_conv['assigned_gm']
                    logging.info(
                        'TimeBasedNextGM: routing %s to conversation shard %s',
                        requesting_entity,
                        result,
                    )
                elif hasattr(conv_state, 'get_conversation_for'):
                  c = conv_state.get_conversation_for(requesting_entity)
                  if (
                      c
                      and getattr(c, 'active', False)
                      and getattr(c, 'assigned_gm', '')
                  ):
                    result = c.assigned_gm
                    logging.info(
                        'TimeBasedNextGM: routing %s to conversation shard %s',
                        requesting_entity,
                        result,
                    )
            except Exception as e:  # pylint: disable=broad-except
              logging.warning(
                  'TimeBasedNextGM: error checking active conv for %s: %s',
                  requesting_entity,
                  e,
              )

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'last_seen_day': self._last_seen_day,
          'switched_entities_by_day': {
              d: list(s) for d, s in self._switched_entities_by_day.items()
          },
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._last_seen_day = component_state.as_int(state, 'last_seen_day', -1)
      raw_switched = state.get('switched_entities_by_day')
      self._switched_entities_by_day = {}
      if isinstance(raw_switched, Mapping):
        for day, names in raw_switched.items():
          if not isinstance(names, Iterable) or isinstance(
              names, (str, bytes, Mapping)
          ):
            continue
          # Written as an int, but a checkpoint that round-tripped through
          # JSON comes back with the day as a string.
          try:
            day_index = int(str(day))
          except ValueError:
            logging.warning(
                'TimeBasedNextGameMaster.set_state: ignoring unparseable day '
                'key %r in switched_entities_by_day',
                day,
            )
            continue
          self._switched_entities_by_day[day_index] = {
              str(name) for name in names
          }

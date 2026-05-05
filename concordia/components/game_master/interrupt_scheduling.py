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

"""Data types and scheduler for interrupt-driven game master orchestration.

This module provides the core types for an interrupt-driven simulation model
where entities control their attention via interrupt masks and timers:

- Event: A timestamped occurrence in the shared environment.
- InterruptMask: A prefix-based filter for which events reach an entity.
- Timer: A scheduled non-maskable interrupt for a specific entity.
- EntityScheduler: A game master component that tracks per-entity masks,
  timers, pending observations, and a shared event queue.
"""

import bisect
from collections.abc import Sequence
import dataclasses
from typing import Any

from concordia.components.game_master import interrupt_time_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, order=True)
class Event:
  """A timestamped occurrence in the shared environment.

  Events are totally ordered by (timestamp, tag, source, description).
  This makes the event queue deterministic.

  The timestamp type is determined by the ``TimeModel`` in use. It must
  support Python comparison operators (``<``, ``<=``, ``==``, ``>=``, ``>``).

  Attributes:
    timestamp: When the event occurs in simulated time. Type depends on the
      active ``TimeModel`` (e.g. ``datetime.datetime`` for
      ``DatetimeTimeModel``, ``int`` for ``GenerativeTimeModel``).
    tag: Category string for interrupt mask matching, e.g. "chat.direct",
      "experiment.completed". Timer events use "timer.expired.<entity_name>".
    source: Name of the entity that caused the event, or "__external__" for
      injected events.
    description: Human-readable description of what happened.
  """

  timestamp: Any
  tag: str
  source: str
  description: str


@dataclasses.dataclass(frozen=True)
class InterruptMask:
  """Specifies which event tags reach an entity using prefix matching.

  An event matches if any prefix string in ``prefixes`` is a prefix of the
  event's tag. The empty string ``""`` is a prefix of everything, so
  ``InterruptMask(prefixes=("",))`` matches all events (this is the default).
  An empty tuple matches nothing (but timer expiry is still non-maskable).

  Examples:
    ``InterruptMask()``                              — match all events
    ``InterruptMask(prefixes=("chat.",))``            — match all chat events
    ``InterruptMask(prefixes=("chat.direct",))``      — match direct chats only
    ``InterruptMask(prefixes=("chat.", "alarm."))``   — match chat or alarm
    ``InterruptMask(prefixes=())``                    — match nothing

  Attributes:
    prefixes: Tuple of tag prefix strings. An event matches if its tag starts
      with any of these prefixes.
  """

  prefixes: tuple[str, ...] = ('',)

  def matches(self, event: Event) -> bool:
    """Returns True if this mask matches the given event's tag."""
    return any(event.tag.startswith(p) for p in self.prefixes)


# Convenience constants.
MATCH_ALL = InterruptMask()  # prefixes=("",)
MATCH_NONE = InterruptMask(prefixes=())


def mask_from_prefixes(
    prefixes: Sequence[str],
) -> InterruptMask:
  """Creates an InterruptMask from a list of prefix strings.

  Args:
    prefixes: Event-tag prefixes to match.  An empty list means match nothing
      (MATCH_NONE).  A list containing the empty string ``[""]`` matches
      everything (MATCH_ALL).

  Returns:
    The corresponding InterruptMask.
  """
  if not prefixes:
    return MATCH_NONE
  prefix_tuple = tuple(prefixes)
  if '' in prefix_tuple:
    return MATCH_ALL
  return InterruptMask(prefixes=prefix_tuple)


# Tag prefix for timer expiry events.
TIMER_EXPIRED_TAG_PREFIX = 'timer.expired.'


@dataclasses.dataclass(frozen=True)
class Timer:
  """A scheduled timer interrupt for a specific entity.

  Timer events are non-maskable: they always reach the owning entity,
  regardless of its current interrupt mask. Timer events are also private:
  they only poll the owning entity, not any other entity.

  Each entity may have at most one pending timer. Setting a new timer
  replaces any existing one.

  Attributes:
    expiry: When the timer fires in simulated time. Type depends on the
      active ``TimeModel``.
    entity_name: The entity that scheduled this timer.
    description: Human-readable description of why the timer was set.
  """

  expiry: Any
  entity_name: str
  description: str


# ──────────────────────────────────────────────────────────────────────────────
# Entity Scheduler
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SCHEDULER_COMPONENT_KEY = 'entity_scheduler'


class EntityScheduler(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Manages per-entity interrupt masks, timers, and pending observations.

  This is the heart of the interrupt-driven game master. It maintains:

  - **Interrupt masks**: Per-entity prefix-based filters that determine which
    events the entity responds to. Default is MATCH_ALL.
  - **Timers**: At most one per entity. Non-maskable and private.
  - **Pending observations**: Events that didn't match an entity's mask,
    accumulated for delivery when the entity is next polled.
  - **Event queue**: A sorted list of ``Event`` objects representing pending
    occurrences in the shared environment.
  - **Simulated clock**: The current simulated time, advanced by the
    next-acting component.

  Key invariant: every entity response must provide a new mask and a new
  timer. Since timer expiry always produces a new entity response (and hence
  a new timer), the event queue is never empty as long as there are
  participants.

  Other components interact with the scheduler as follows:

  - ``_InterruptNextActing`` queries the scheduler to find the next event
    or timer, advances time, determines which entities to poll, and calls
    ``poll_entity`` to clear their mask and timer.
  - ``_InterruptMakeObservation`` drains pending observations for each
    polled entity.
  - ``_InterruptResolution`` updates masks and timers based on entity
    responses, and injects new events.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      time_model: interrupt_time_model.TimeModel,
      max_ticks: int | None = None,
  ):
    """Initialises the entity scheduler.

    Args:
      player_names: Names of all participant entities.
      time_model: The time model to use for timestamp creation,
        formatting, and serialisation.
      max_ticks: Optional maximum number of ticks (event processing steps)
        before the simulation should terminate. None means no limit.
    """
    super().__init__()
    self._time_model = time_model

    self._player_names = list(player_names)
    self._current_time = self._time_model.initial_time()
    self._max_ticks = max_ticks
    self._tick_count = 0

    # Per-entity state.
    self._masks: dict[str, InterruptMask] = {
        name: MATCH_ALL for name in player_names
    }
    self._timers: dict[str, Timer | None] = {
        name: Timer(
            expiry=self._current_time,
            entity_name=name,
            description='initial timer',
        )
        for name in player_names
    }
    self._pending: dict[str, list[Event]] = {name: [] for name in player_names}

    # Shared event queue, sorted by Event's natural ordering.
    self._event_queue: list[Event] = []

    # The event currently being processed (set by next_acting, read by others).
    self._current_event: Event | None = None

    # Entities being polled in the current step (set by next_acting).
    self._current_polled_entities: list[str] = []

  @property
  def player_names(self) -> list[str]:
    """Returns the list of player names."""
    return list(self._player_names)

  @property
  def time_model(self) -> interrupt_time_model.TimeModel:
    """Returns the time model used by this scheduler."""
    return self._time_model

  # ── Time ──────────────────────────────────────────────────────────────────

  def get_current_time(self) -> Any:
    """Returns the current simulated time."""
    return self._current_time

  def advance_time(self, new_time: Any) -> None:
    """Advances the simulated clock.

    Args:
      new_time: The new simulated time. Must not be before current time.

    Raises:
      ValueError: If new_time is before the current time.
    """
    if new_time < self._current_time:
      raise ValueError(
          f'Cannot move time backwards: {new_time} < {self._current_time}'
      )
    self._current_time = new_time

  def format_time(self) -> str:
    """Returns the current simulated time as a human-readable string."""
    return self._time_model.format_time(self._current_time)

  # ── Event queue ───────────────────────────────────────────────────────────

  def inject_event(self, event: Event) -> None:
    """Adds an event to the queue, maintaining sorted order.

    Args:
      event: The event to inject.

    Raises:
      ValueError: If the event's timestamp is before the current simulated
        time.
    """
    if event.timestamp < self._current_time:
      raise ValueError(
          'Cannot inject event in the past: event timestamp'
          f' {event.timestamp} < current time {self._current_time}'
      )
    bisect.insort(self._event_queue, event)

  def pop_next_event(self) -> Event | None:
    """Removes and returns the next event, or None if empty."""
    return self._event_queue.pop(0) if self._event_queue else None

  def peek_event_queue(self) -> list[Event]:
    """Returns a copy of the current event queue (sorted by Event order)."""
    return list(self._event_queue)

  # ── Current event (for inter-component communication) ─────────────────────

  def get_current_event(self) -> Event | None:
    """Returns the event currently being processed."""
    return self._current_event

  def set_current_event(self, event: Event | None) -> None:
    """Sets the event currently being processed."""
    self._current_event = event

  def get_current_polled_entities(self) -> list[str]:
    """Returns the entities being polled in the current step."""
    return list(self._current_polled_entities)

  def set_current_polled_entities(self, names: list[str]) -> None:
    """Sets the entities being polled in the current step."""
    self._current_polled_entities = list(names)

  # ── Masks ─────────────────────────────────────────────────────────────────

  def get_mask(self, entity_name: str) -> InterruptMask:
    """Returns the current interrupt mask for an entity."""
    return self._masks[entity_name]

  def set_mask(self, entity_name: str, mask: InterruptMask) -> None:
    """Sets the interrupt mask for an entity."""
    self._masks[entity_name] = mask

  def clear_mask(self, entity_name: str) -> None:
    """Resets an entity's mask to MATCH_ALL."""
    self._masks[entity_name] = MATCH_ALL

  # ── Timers ────────────────────────────────────────────────────────────────

  def get_timer(self, entity_name: str) -> Timer | None:
    """Returns the pending timer for an entity, or None."""
    return self._timers[entity_name]

  def set_timer(self, entity_name: str, timer: Timer) -> None:
    """Sets (or replaces) the pending timer for an entity.

    Args:
      entity_name: The entity whose timer to set.
      timer: The new timer. Its entity_name field should match.
    """
    self._timers[entity_name] = timer

  def clear_timer(self, entity_name: str) -> None:
    """Clears the pending timer for an entity."""
    self._timers[entity_name] = None

  def get_next_timer(self) -> Timer | None:
    """Returns the earliest pending timer across all entities, or None."""
    earliest: Timer | None = None
    for timer in self._timers.values():
      if timer is not None:
        if earliest is None or timer.expiry < earliest.expiry:
          earliest = timer
    return earliest

  # ── Pending observations ──────────────────────────────────────────────────

  def add_pending(self, entity_name: str, event: Event) -> None:
    """Adds an event to an entity's pending observation queue."""
    self._pending[entity_name].append(event)

  def drain_pending(self, entity_name: str) -> list[Event]:
    """Returns and clears accumulated pending observations for an entity."""
    events = list(self._pending[entity_name])
    self._pending[entity_name] = []
    return events

  # ── Polling ───────────────────────────────────────────────────────────────

  def poll_entity(self, entity_name: str) -> None:
    """Clears both the mask and timer for an entity being polled.

    Called when an entity is about to receive an interrupt (either from a
    matching event or from a timer expiry). This resets the entity to a
    clean state; the entity's response must provide a new mask and timer.

    Args:
      entity_name: The entity being polled.
    """
    self.clear_mask(entity_name)
    self.clear_timer(entity_name)

  def get_entities_to_poll(self, event: Event) -> list[str]:
    """Determines which entities should be polled for an event.

    For timer events (tag starts with "timer.expired."): only the owning
    entity is polled.

    For all other events: any entity whose mask matches the event is polled,
    plus any entity whose timer has expired at or before the event's
    timestamp (since timer expiry is non-maskable).  The entity that
    *sourced* the event is excluded from mask-matching so that an
    entity's own action does not trigger itself.

    Args:
      event: The event to check against entity masks and timers.

    Returns:
      List of entity names to poll.
    """
    if event.tag.startswith(TIMER_EXPIRED_TAG_PREFIX):
      # Timer events are private: only poll the owning entity.
      entity_name = event.tag[len(TIMER_EXPIRED_TAG_PREFIX) :]
      if entity_name in self._masks:
        return [entity_name]
      return []

    # Non-timer event: poll mask-matched entities, excluding the source.
    to_poll = set()
    for name in self._player_names:
      if name != event.source and self._masks[name].matches(event):
        to_poll.add(name)

    # Also poll any entity whose timer has expired at or before this event.
    for name in self._player_names:
      timer = self._timers[name]
      if timer is not None and timer.expiry <= event.timestamp:
        to_poll.add(name)

    return list(to_poll)

  # ── Termination ───────────────────────────────────────────────────────────

  def increment_tick(self) -> None:
    """Increments the tick counter."""
    self._tick_count += 1

  def get_tick_count(self) -> int:
    """Returns the current tick count."""
    return self._tick_count

  def should_terminate(self) -> bool:
    """Returns True if the maximum tick count has been reached."""
    if self._max_ticks is not None and self._tick_count >= self._max_ticks:
      return True
    return False

  # ── ContextComponent interface ────────────────────────────────────────────

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """Returns empty string; scheduler state is accessed via methods."""
    del action_spec
    return ''

  # ── State serialisation ───────────────────────────────────────────────────

  def get_state(self) -> entity_component.ComponentState:
    """Returns the serialisable state of the scheduler."""
    timers_state = {}
    for name, timer in self._timers.items():
      if timer is not None:
        timers_state[name] = {
            'expiry': self._time_model.serialize_time(timer.expiry),
            'entity_name': timer.entity_name,
            'description': timer.description,
        }
      else:
        timers_state[name] = None

    masks_state = {
        name: list(mask.prefixes) for name, mask in self._masks.items()
    }

    pending_state = {}
    for name, events in self._pending.items():
      pending_state[name] = [
          {
              'timestamp': self._time_model.serialize_time(e.timestamp),
              'tag': e.tag,
              'source': e.source,
              'description': e.description,
          }
          for e in events
      ]

    event_queue_state = [
        {
            'timestamp': self._time_model.serialize_time(e.timestamp),
            'tag': e.tag,
            'source': e.source,
            'description': e.description,
        }
        for e in self._event_queue
    ]

    current_event_state = None
    if self._current_event is not None:
      current_event_state = {
          'timestamp': self._time_model.serialize_time(
              self._current_event.timestamp
          ),
          'tag': self._current_event.tag,
          'source': self._current_event.source,
          'description': self._current_event.description,
      }

    return {
        'current_time': self._time_model.serialize_time(self._current_time),
        'tick_count': self._tick_count,
        'masks': masks_state,
        'timers': timers_state,
        'pending': pending_state,
        'event_queue': event_queue_state,
        'current_event': current_event_state,
        'current_polled_entities': self._current_polled_entities,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores the scheduler state from a serialised form."""
    self._current_time = self._time_model.deserialize_time(
        state['current_time']
    )
    self._tick_count = int(state['tick_count'])

    # Restore masks.
    masks_state = state.get('masks', {})
    for name in self._player_names:
      if name in masks_state:
        self._masks[name] = InterruptMask(prefixes=tuple(masks_state[name]))

    # Restore timers.
    timers_state = state.get('timers', {})
    for name in self._player_names:
      if name in timers_state and timers_state[name] is not None:
        t = timers_state[name]
        self._timers[name] = Timer(
            expiry=self._time_model.deserialize_time(t['expiry']),
            entity_name=t['entity_name'],
            description=t['description'],
        )
      else:
        self._timers[name] = None

    # Restore pending observations.
    pending_state = state.get('pending', {})
    for name in self._player_names:
      if name in pending_state:
        self._pending[name] = [
            Event(
                timestamp=self._time_model.deserialize_time(e['timestamp']),
                tag=e['tag'],
                source=e['source'],
                description=e['description'],
            )
            for e in pending_state[name]
        ]

    # Restore event queue.
    self._event_queue = [
        Event(
            timestamp=self._time_model.deserialize_time(e['timestamp']),
            tag=e['tag'],
            source=e['source'],
            description=e['description'],
        )
        for e in state.get('event_queue', [])
    ]

    # Restore current event.
    ce = state.get('current_event')
    if ce is not None:
      self._current_event = Event(
          timestamp=self._time_model.deserialize_time(ce['timestamp']),
          tag=ce['tag'],
          source=ce['source'],
          description=ce['description'],
      )
    else:
      self._current_event = None

    self._current_polled_entities = [
        str(x) for x in state.get('current_polled_entities', [])
    ]

# Copyright 2024 DeepMind Technologies Limited.
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

"""Game master component for tracking schedules and timed events."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
from typing import Any

from concordia.typing import entity_component


@dataclasses.dataclass(frozen=True)
class ScheduledEvent:
  """Represents a scheduled event in the simulation.

  Attributes:
    name: The name/description of the event.
    scheduled_time: When the event is scheduled to occur.
    participants: List of entity names involved in the event (optional).
    location: Where the event takes place (optional).
    duration_minutes: How long the event lasts (optional).
    recurring: Whether this is a recurring event (daily, weekly, etc.).
    priority: Priority level of the event (1-10, where 10 is highest).
    completed: Whether the event has been completed.
    details: Additional details about the event.
  """
  name: str
  scheduled_time: datetime.datetime
  _: dataclasses.KW_ONLY
  participants: tuple[str, ...] = ()
  location: str = ''
  duration_minutes: int = 0
  recurring: str = ''  # e.g., 'daily', 'weekly', 'none'
  priority: int = 5
  completed: bool = False
  details: str = ''

  def to_dict(self) -> dict[str, Any]:
    """Converts the event to a dictionary."""
    return {
        'name': self.name,
        'scheduled_time': self.scheduled_time.isoformat(),
        'participants': list(self.participants),
        'location': self.location,
        'duration_minutes': self.duration_minutes,
        'recurring': self.recurring,
        'priority': self.priority,
        'completed': self.completed,
        'details': self.details,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ScheduledEvent':
    """Creates an event from a dictionary."""
    return cls(
        name=data['name'],
        scheduled_time=datetime.datetime.fromisoformat(data['scheduled_time']),
        participants=tuple(data.get('participants', [])),
        location=data.get('location', ''),
        duration_minutes=data.get('duration_minutes', 0),
        recurring=data.get('recurring', ''),
        priority=data.get('priority', 5),
        completed=data.get('completed', False),
        details=data.get('details', ''),
    )


class ScheduleTracker(entity_component.ContextComponent):
  """Tracks scheduled events and provides time-aware context for simulations.

  This component maintains a schedule of events, tracks their completion,
  and provides utilities for querying upcoming and past events. It's useful
  for simulations that involve appointments, meetings, deadlines, or any
  time-dependent activities.
  """

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      pre_act_label: str = 'Scheduled events',
      verbose: bool = True,
  ):
    """Initializes the ScheduleTracker component.

    Args:
      clock_now: A callable that returns the current simulation time.
      pre_act_label: Label to use when outputting schedule information.
      verbose: Whether to include detailed information in pre_act output.
    """
    self._clock_now = clock_now
    self._pre_act_label = pre_act_label
    self._verbose = verbose
    self._events: list[ScheduledEvent] = []
    self._event_history: list[ScheduledEvent] = []

  def add_event(
      self,
      name: str,
      scheduled_time: datetime.datetime,
      participants: Sequence[str] = (),
      location: str = '',
      duration_minutes: int = 0,
      recurring: str = '',
      priority: int = 5,
      details: str = '',
  ) -> None:
    """Adds a new event to the schedule.

    Args:
      name: The name/description of the event.
      scheduled_time: When the event is scheduled to occur.
      participants: List of entity names involved in the event.
      location: Where the event takes place.
      duration_minutes: How long the event lasts.
      recurring: Whether this is a recurring event.
      priority: Priority level (1-10, where 10 is highest).
      details: Additional details about the event.
    """
    event = ScheduledEvent(
        name=name,
        scheduled_time=scheduled_time,
        participants=tuple(participants),
        location=location,
        duration_minutes=duration_minutes,
        recurring=recurring,
        priority=priority,
        completed=False,
        details=details,
    )
    self._events.append(event)
    # Keep events sorted by time
    self._events.sort(key=lambda e: (e.scheduled_time, -e.priority))

  def complete_event(self, event_name: str) -> bool:
    """Marks an event as completed.

    Args:
      event_name: The name of the event to complete.

    Returns:
      True if the event was found and marked complete, False otherwise.
    """
    for i, event in enumerate(self._events):
      if event.name == event_name and not event.completed:
        # Create a new event with completed=True (since events are frozen)
        completed_event = ScheduledEvent(
            name=event.name,
            scheduled_time=event.scheduled_time,
            participants=event.participants,
            location=event.location,
            duration_minutes=event.duration_minutes,
            recurring=event.recurring,
            priority=event.priority,
            completed=True,
            details=event.details,
        )
        self._events[i] = completed_event
        self._event_history.append(completed_event)
        return True
    return False

  def get_upcoming_events(
      self,
      limit: int = 10,
      time_window_minutes: int | None = None,
      participant: str | None = None,
  ) -> Sequence[ScheduledEvent]:
    """Returns upcoming events.

    Args:
      limit: Maximum number of events to return.
      time_window_minutes: Only return events within this many minutes.
      participant: If specified, only return events with this participant.

    Returns:
      A list of upcoming events, sorted by time then priority.
    """
    now = self._clock_now()
    upcoming = []

    for event in self._events:
      if event.completed:
        continue
      if event.scheduled_time < now:
        continue
      if time_window_minutes and (
          event.scheduled_time - now
      ).total_seconds() > time_window_minutes * 60:
        continue
      if participant and participant not in event.participants:
        continue

      upcoming.append(event)
      if len(upcoming) >= limit:
        break

    return upcoming

  def get_overdue_events(self) -> Sequence[ScheduledEvent]:
    """Returns events that are past their scheduled time but not completed.

    Returns:
      A list of overdue events.
    """
    now = self._clock_now()
    return [
        event for event in self._events
        if event.scheduled_time < now and not event.completed
    ]

  def get_current_events(self) -> Sequence[ScheduledEvent]:
    """Returns events that are currently happening.

    Returns:
      A list of events whose scheduled time is now and duration hasn't elapsed.
    """
    now = self._clock_now()
    current = []

    for event in self._events:
      if event.completed:
        continue
      if event.scheduled_time > now:
        continue
      if event.duration_minutes == 0:
        continue

      end_time = event.scheduled_time + datetime.timedelta(
          minutes=event.duration_minutes
      )
      if end_time > now:
        current.append(event)

    return current

  def get_events_by_location(self, location: str) -> Sequence[ScheduledEvent]:
    """Returns all events at a specific location.

    Args:
      location: The location to filter by.

    Returns:
      A list of events at the specified location.
    """
    return [
        event for event in self._events
        if event.location == location and not event.completed
    ]

  def pre_act(
      self,
      action_spec: entity_component.ActionSpec,
  ) -> str:
    """Returns schedule information for the current time.

    Args:
      action_spec: The action specification (unused).

    Returns:
      A formatted string with schedule information.
    """
    del action_spec  # Unused

    now = self._clock_now()
    output_parts = [f'{self._pre_act_label}:']

    # Current events
    current = self.get_current_events()
    if current:
      output_parts.append('\nCurrently happening:')
      for event in current:
        if self._verbose:
          output_parts.append(
              f'  - {event.name} at {event.location} '
              f'(ends at {event.scheduled_time + datetime.timedelta(minutes=event.duration_minutes):%H:%M})'
          )
          if event.participants:
            output_parts.append(
                f'    Participants: {", ".join(event.participants)}'
            )
        else:
          output_parts.append(f'  - {event.name}')

    # Overdue events
    overdue = self.get_overdue_events()
    if overdue:
      output_parts.append('\nOverdue events:')
      for event in overdue[:5]:  # Limit to avoid clutter
        output_parts.append(f'  - {event.name} (was at {event.scheduled_time:%H:%M})')

    # Upcoming events
    upcoming = self.get_upcoming_events(limit=5, time_window_minutes=180)
    if upcoming:
      output_parts.append('\nUpcoming events:')
      for event in upcoming:
        time_until = event.scheduled_time - now
        hours = int(time_until.total_seconds() // 3600)
        minutes = int((time_until.total_seconds() % 3600) // 60)
        
        if self._verbose:
          time_str = f'{event.scheduled_time:%H:%M}'
          if hours > 0:
            time_str += f' (in {hours}h {minutes}m)'
          else:
            time_str += f' (in {minutes}m)'
          
          output_parts.append(f'  - {event.name} at {time_str}')
          if event.location:
            output_parts.append(f'    Location: {event.location}')
          if event.participants:
            output_parts.append(
                f'    Participants: {", ".join(event.participants)}'
            )
        else:
          output_parts.append(
              f'  - {event.name} at {event.scheduled_time:%H:%M}'
          )

    if len(output_parts) == 1:
      output_parts.append('  No scheduled events at this time.')

    return '\n'.join(output_parts)

  def update(self) -> None:
    """Updates the schedule, handling recurring events."""
    now = self._clock_now()
    
    # Process recurring events that have passed
    new_recurring_events = []
    for event in self._events:
      if not event.recurring or event.recurring == 'none':
        continue
      if event.scheduled_time >= now:
        continue
      if event.completed:
        continue

      # Auto-complete old instance and create new one
      self.complete_event(event.name)

      # Calculate next occurrence
      if event.recurring == 'daily':
        next_time = event.scheduled_time + datetime.timedelta(days=1)
      elif event.recurring == 'weekly':
        next_time = event.scheduled_time + datetime.timedelta(weeks=1)
      elif event.recurring == 'monthly':
        # Approximate monthly as 30 days
        next_time = event.scheduled_time + datetime.timedelta(days=30)
      else:
        continue

      # Only add if the next occurrence is in the future
      if next_time > now:
        new_recurring_events.append(ScheduledEvent(
            name=event.name,
            scheduled_time=next_time,
            participants=event.participants,
            location=event.location,
            duration_minutes=event.duration_minutes,
            recurring=event.recurring,
            priority=event.priority,
            completed=False,
            details=event.details,
        ))

    self._events.extend(new_recurring_events)
    self._events.sort(key=lambda e: (e.scheduled_time, -e.priority))

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'events': [event.to_dict() for event in self._events],
        'event_history': [event.to_dict() for event in self._event_history],
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    if 'events' in state:
      self._events = [
          ScheduledEvent.from_dict(event_dict)
          for event_dict in state['events']
      ]
    if 'event_history' in state:
      self._event_history = [
          ScheduledEvent.from_dict(event_dict)
          for event_dict in state['event_history']
      ]

  def get_pre_act_label(self) -> str:
    """Returns the label to use in pre_act output."""
    return self._pre_act_label

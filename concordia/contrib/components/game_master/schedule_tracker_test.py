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

"""Tests for schedule_tracker component."""

import datetime
import unittest
from unittest import mock

from concordia.contrib.components.game_master import schedule_tracker


class ScheduledEventTest(unittest.TestCase):
  """Tests for the ScheduledEvent dataclass."""

  def test_event_creation(self):
    """Tests that events can be created."""
    event_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
    event = schedule_tracker.ScheduledEvent(
        name='Team Meeting',
        scheduled_time=event_time,
        participants=('Alice', 'Bob'),
        location='Conference Room A',
        duration_minutes=60,
        priority=8,
    )
    
    self.assertEqual(event.name, 'Team Meeting')
    self.assertEqual(event.scheduled_time, event_time)
    self.assertEqual(event.participants, ('Alice', 'Bob'))
    self.assertEqual(event.location, 'Conference Room A')
    self.assertEqual(event.duration_minutes, 60)
    self.assertEqual(event.priority, 8)
    self.assertFalse(event.completed)

  def test_event_to_dict(self):
    """Tests event serialization to dictionary."""
    event_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
    event = schedule_tracker.ScheduledEvent(
        name='Lunch',
        scheduled_time=event_time,
        location='Cafeteria',
    )
    
    event_dict = event.to_dict()
    self.assertEqual(event_dict['name'], 'Lunch')
    self.assertEqual(event_dict['location'], 'Cafeteria')
    self.assertIn('scheduled_time', event_dict)

  def test_event_from_dict(self):
    """Tests event deserialization from dictionary."""
    event_time = datetime.datetime(2024, 1, 1, 14, 30, 0)
    event_dict = {
        'name': 'Project Review',
        'scheduled_time': event_time.isoformat(),
        'participants': ['Charlie', 'Diana'],
        'location': 'Office 101',
        'duration_minutes': 45,
        'priority': 7,
        'completed': False,
        'details': 'Quarterly review',
        'recurring': 'weekly',
    }
    
    event = schedule_tracker.ScheduledEvent.from_dict(event_dict)
    self.assertEqual(event.name, 'Project Review')
    self.assertEqual(event.scheduled_time, event_time)
    self.assertEqual(event.participants, ('Charlie', 'Diana'))
    self.assertEqual(event.location, 'Office 101')
    self.assertEqual(event.duration_minutes, 45)
    self.assertEqual(event.priority, 7)
    self.assertEqual(event.recurring, 'weekly')


class ScheduleTrackerTest(unittest.TestCase):
  """Tests for the ScheduleTracker component."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.current_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
    self.clock = mock.Mock(return_value=self.current_time)
    self.tracker = schedule_tracker.ScheduleTracker(
        clock_now=self.clock,
        verbose=True,
    )

  def test_initialization(self):
    """Tests that tracker can be initialized."""
    self.assertIsNotNone(self.tracker)
    self.assertEqual(self.tracker.get_pre_act_label(), 'Scheduled events')

  def test_add_event(self):
    """Tests adding an event to the schedule."""
    event_time = datetime.datetime(2024, 1, 1, 15, 0, 0)
    self.tracker.add_event(
        name='Doctor Appointment',
        scheduled_time=event_time,
        location='Clinic',
        duration_minutes=30,
    )
    
    upcoming = self.tracker.get_upcoming_events()
    self.assertEqual(len(upcoming), 1)
    self.assertEqual(upcoming[0].name, 'Doctor Appointment')

  def test_get_upcoming_events(self):
    """Tests retrieving upcoming events."""
    # Add several events
    self.tracker.add_event(
        name='Event 1',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
        priority=5,
    )
    self.tracker.add_event(
        name='Event 2',
        scheduled_time=datetime.datetime(2024, 1, 1, 14, 0, 0),
        priority=3,
    )
    self.tracker.add_event(
        name='Event 3',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
        priority=8,  # Same time as Event 1, but higher priority
    )
    
    upcoming = self.tracker.get_upcoming_events(limit=10)
    self.assertEqual(len(upcoming), 3)
    # Event 3 should come before Event 1 due to higher priority at same time
    self.assertEqual(upcoming[0].name, 'Event 3')
    self.assertEqual(upcoming[1].name, 'Event 1')
    self.assertEqual(upcoming[2].name, 'Event 2')

  def test_get_upcoming_events_with_time_window(self):
    """Tests filtering upcoming events by time window."""
    # Event within window
    self.tracker.add_event(
        name='Near Event',
        scheduled_time=datetime.datetime(2024, 1, 1, 12, 30, 0),
    )
    # Event outside window
    self.tracker.add_event(
        name='Far Event',
        scheduled_time=datetime.datetime(2024, 1, 1, 16, 0, 0),
    )
    
    upcoming = self.tracker.get_upcoming_events(
        time_window_minutes=60
    )
    self.assertEqual(len(upcoming), 1)
    self.assertEqual(upcoming[0].name, 'Near Event')

  def test_get_upcoming_events_by_participant(self):
    """Tests filtering upcoming events by participant."""
    self.tracker.add_event(
        name='Alice Meeting',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
        participants=['Alice', 'Bob'],
    )
    self.tracker.add_event(
        name='Charlie Meeting',
        scheduled_time=datetime.datetime(2024, 1, 1, 14, 0, 0),
        participants=['Charlie'],
    )
    
    alice_events = self.tracker.get_upcoming_events(participant='Alice')
    self.assertEqual(len(alice_events), 1)
    self.assertEqual(alice_events[0].name, 'Alice Meeting')
    
    charlie_events = self.tracker.get_upcoming_events(participant='Charlie')
    self.assertEqual(len(charlie_events), 1)
    self.assertEqual(charlie_events[0].name, 'Charlie Meeting')

  def test_complete_event(self):
    """Tests completing an event."""
    self.tracker.add_event(
        name='Task to Complete',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
    )
    
    # Event should be in upcoming
    self.assertEqual(len(self.tracker.get_upcoming_events()), 1)
    
    # Complete the event
    result = self.tracker.complete_event('Task to Complete')
    self.assertTrue(result)
    
    # Event should no longer appear in upcoming
    # (because it's marked completed, not because it's past)
    upcoming = self.tracker.get_upcoming_events()
    self.assertEqual(len(upcoming), 0)

  def test_get_overdue_events(self):
    """Tests retrieving overdue events."""
    # Add a past event
    self.tracker.add_event(
        name='Overdue Task',
        scheduled_time=datetime.datetime(2024, 1, 1, 10, 0, 0),
    )
    # Add a future event
    self.tracker.add_event(
        name='Future Task',
        scheduled_time=datetime.datetime(2024, 1, 1, 14, 0, 0),
    )
    
    overdue = self.tracker.get_overdue_events()
    self.assertEqual(len(overdue), 1)
    self.assertEqual(overdue[0].name, 'Overdue Task')

  def test_get_current_events(self):
    """Tests retrieving events that are currently happening."""
    # Event that started and is still ongoing
    self.tracker.add_event(
        name='Current Meeting',
        scheduled_time=datetime.datetime(2024, 1, 1, 11, 30, 0),
        duration_minutes=60,  # Ends at 12:30, after current time
    )
    # Event that already ended
    self.tracker.add_event(
        name='Past Meeting',
        scheduled_time=datetime.datetime(2024, 1, 1, 10, 0, 0),
        duration_minutes=30,  # Ended at 10:30
    )
    # Event with no duration (not current)
    self.tracker.add_event(
        name='Instant Event',
        scheduled_time=datetime.datetime(2024, 1, 1, 11, 0, 0),
        duration_minutes=0,
    )
    
    current = self.tracker.get_current_events()
    self.assertEqual(len(current), 1)
    self.assertEqual(current[0].name, 'Current Meeting')

  def test_get_events_by_location(self):
    """Tests retrieving events by location."""
    self.tracker.add_event(
        name='Meeting A',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
        location='Room 101',
    )
    self.tracker.add_event(
        name='Meeting B',
        scheduled_time=datetime.datetime(2024, 1, 1, 14, 0, 0),
        location='Room 101',
    )
    self.tracker.add_event(
        name='Meeting C',
        scheduled_time=datetime.datetime(2024, 1, 1, 15, 0, 0),
        location='Room 202',
    )
    
    room_101_events = self.tracker.get_events_by_location('Room 101')
    self.assertEqual(len(room_101_events), 2)

  def test_recurring_events(self):
    """Tests handling of recurring events."""
    # Add a daily recurring event that's in the past
    self.tracker.add_event(
        name='Daily Standup',
        scheduled_time=datetime.datetime(2024, 1, 1, 9, 0, 0),
        duration_minutes=15,
        recurring='daily',
    )
    
    # Call update to process recurring events
    self.tracker.update()
    
    # The original event should be marked complete
    overdue = self.tracker.get_overdue_events()
    self.assertEqual(len(overdue), 0)  # Should be auto-completed
    
    # A new instance should be created for tomorrow
    upcoming = self.tracker.get_upcoming_events()
    self.assertTrue(any(
        event.name == 'Daily Standup' and 
        event.scheduled_time.day == 2
        for event in upcoming
    ))

  def test_pre_act_output(self):
    """Tests the pre_act output formatting."""
    # Add various events
    self.tracker.add_event(
        name='Upcoming Meeting',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
        location='Conference Room',
        participants=['Alice', 'Bob'],
    )
    self.tracker.add_event(
        name='Overdue Task',
        scheduled_time=datetime.datetime(2024, 1, 1, 10, 0, 0),
    )
    self.tracker.add_event(
        name='Current Event',
        scheduled_time=datetime.datetime(2024, 1, 1, 11, 0, 0),
        duration_minutes=90,  # Still ongoing
    )
    
    output = self.tracker.pre_act(action_spec=None)
    
    self.assertIn('Scheduled events', output)
    self.assertIn('Upcoming Meeting', output)
    self.assertIn('Overdue Task', output)
    self.assertIn('Current Event', output)

  def test_state_persistence(self):
    """Tests that state can be saved and restored."""
    # Add some events
    self.tracker.add_event(
        name='Event 1',
        scheduled_time=datetime.datetime(2024, 1, 1, 13, 0, 0),
    )
    self.tracker.add_event(
        name='Event 2',
        scheduled_time=datetime.datetime(2024, 1, 1, 14, 0, 0),
    )
    self.tracker.complete_event('Event 1')
    
    # Save state
    state = self.tracker.get_state()
    
    # Create new tracker and restore state
    new_tracker = schedule_tracker.ScheduleTracker(
        clock_now=self.clock,
    )
    new_tracker.set_state(state)
    
    # Verify state was restored
    upcoming = new_tracker.get_upcoming_events()
    self.assertEqual(len(upcoming), 1)
    self.assertEqual(upcoming[0].name, 'Event 2')
    
    # Event 1 should be in history
    history = new_tracker.get_state()['event_history']
    self.assertEqual(len(history), 1)


if __name__ == '__main__':
  unittest.main()

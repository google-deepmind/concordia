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

"""Unit tests for the InterruptNextActing component."""

import datetime
from unittest import mock

from absl.testing import absltest
from concordia.components.game_master import interrupt_next_acting
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model
from concordia.typing import entity as entity_lib


_START = datetime.datetime(2026, 1, 1, 9, 0)

_NEXT_ACTING_SPEC = entity_lib.ActionSpec(
    call_to_action='',
    output_type=entity_lib.OutputType.NEXT_ACTING,
    options=('Alice', 'Bob'),
)

_WRONG_SPEC = entity_lib.ActionSpec(
    call_to_action='',
    output_type=entity_lib.OutputType.RESOLVE,
)


def _make_scheduler(
    names: list[str],
    start: datetime.datetime = _START,
    max_ticks: int | None = None,
) -> interrupt_scheduling.EntityScheduler:
  return interrupt_scheduling.EntityScheduler(
      player_names=names,
      time_model=interrupt_time_model.DatetimeTimeModel(start),
      max_ticks=max_ticks,
  )


def _wire(component, scheduler):
  """Wires a component to a mock entity carrying the given scheduler."""
  components = {
      interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY: scheduler,
  }
  mock_entity = mock.MagicMock()
  mock_entity.get_component.side_effect = lambda key, type_=None: components[
      key
  ]
  component.set_entity(mock_entity)
  # Silence the logging channel.
  component._logging_channel = mock.MagicMock()


class WrongOutputTypeTest(absltest.TestCase):

  def test_returns_empty_for_non_next_acting(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_WRONG_SPEC)
    self.assertEqual(result, '')


class TimerVsEventPriorityTest(absltest.TestCase):
  """Tests that the correct firing order is chosen."""

  def test_timer_fires_before_event(self):
    scheduler = _make_scheduler(['Alice'])
    # Alice's initial timer is at _START.  Inject an event at _START+10m.
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=10),
            tag='alert.test',
            source='__external__',
            description='An alert.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(result, 'Alice')
    # Time should advance to the timer's expiry (_START), not the event.
    self.assertEqual(scheduler.get_current_time(), _START)

  def test_event_fires_before_timer(self):
    scheduler = _make_scheduler(['Alice'])
    # Replace Alice's initial timer with one far in the future.
    scheduler.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=_START + datetime.timedelta(hours=2),
            entity_name='Alice',
            description='distant timer',
        ),
    )
    # Inject an event at _START+5m.
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=5),
            tag='chat.hello',
            source='__external__',
            description='Someone says hello.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(result, 'Alice')
    # Time should advance to the event's timestamp.
    self.assertEqual(
        scheduler.get_current_time(),
        _START + datetime.timedelta(minutes=5),
    )

  def test_timer_wins_ties(self):
    scheduler = _make_scheduler(['Alice'])
    # Inject an event at _START (same time as initial timer).
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START,
            tag='alert.test',
            source='__external__',
            description='Simultaneous alert.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(result, 'Alice')
    # The timer event was consumed; the external event should still be queued.
    queue = scheduler.peek_event_queue()
    self.assertLen(queue, 1)
    self.assertEqual(queue[0].tag, 'alert.test')


class NoEventsOrTimersTest(absltest.TestCase):

  def test_returns_empty_when_nothing_pending(self):
    scheduler = _make_scheduler(['Alice'])
    # Clear Alice's initial timer.
    scheduler.clear_timer('Alice')
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(result, '')


class SkipLoopTest(absltest.TestCase):
  """Tests the loop that skips events no entity reacts to."""

  def test_skips_event_with_no_matching_entity(self):
    scheduler = _make_scheduler(['Alice'])
    # Alice only listens for 'emergency.' events.
    scheduler.set_mask(
        'Alice',
        interrupt_scheduling.InterruptMask(prefixes=('emergency.',)),
    )
    # Clear Alice's timer so only queued events matter.
    scheduler.clear_timer('Alice')

    # Inject two events: first is unmatched, second is matched.
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=1),
            tag='chat.casual',
            source='__external__',
            description='Casual chat.',
        )
    )
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=5),
            tag='emergency.fire',
            source='__external__',
            description='Fire alarm.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(result, 'Alice')
    # Time should have advanced to the matched event.
    self.assertEqual(
        scheduler.get_current_time(),
        _START + datetime.timedelta(minutes=5),
    )


class MultipleEntitiesTest(absltest.TestCase):

  def test_polled_and_pending(self):
    """Tests that polled entities are returned and others get pending."""
    scheduler = _make_scheduler(['Alice', 'Bob'])
    # Alice: MATCH_NONE mask, distant timer.
    scheduler.set_mask('Alice', interrupt_scheduling.MATCH_NONE)
    scheduler.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=_START + datetime.timedelta(hours=2),
            entity_name='Alice',
            description='focus',
        ),
    )
    # Bob: MATCH_ALL mask, distant timer.
    scheduler.set_mask('Bob', interrupt_scheduling.MATCH_ALL)
    scheduler.set_timer(
        'Bob',
        interrupt_scheduling.Timer(
            expiry=_START + datetime.timedelta(hours=2),
            entity_name='Bob',
            description='patrol',
        ),
    )
    # Inject an event.
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=10),
            tag='announcement.daily',
            source='__external__',
            description='Daily announcement.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    result = component.pre_act(_NEXT_ACTING_SPEC)
    # Only Bob should be polled (Alice has MATCH_NONE).
    self.assertEqual(result, 'Bob')
    # Alice should have the event in her pending queue.
    pending = scheduler.drain_pending('Alice')
    self.assertLen(pending, 1)
    self.assertEqual(pending[0].tag, 'announcement.daily')


class TickIncrementTest(absltest.TestCase):

  def test_tick_increments_on_successful_poll(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(scheduler._tick_count, 1)

  def test_tick_increments_on_skipped_event_too(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_mask(
        'Alice',
        interrupt_scheduling.InterruptMask(prefixes=('emergency.',)),
    )
    scheduler.clear_timer('Alice')

    # Inject unmatched then matched event.
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=1),
            tag='chat.casual',
            source='__external__',
            description='Chat.',
        )
    )
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=2),
            tag='emergency.fire',
            source='__external__',
            description='Fire.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    component.pre_act(_NEXT_ACTING_SPEC)
    # One tick for the skipped event + one for the matched event.
    self.assertEqual(scheduler._tick_count, 2)


class InitialStateSnapshotTest(absltest.TestCase):
  """Tests that the initial state is logged exactly once."""

  def test_initial_snapshot_logged_on_first_call(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.inject_event(
        interrupt_scheduling.Event(
            timestamp=_START + datetime.timedelta(minutes=10),
            tag='alert.test',
            source='__external__',
            description='Test alert.',
        )
    )
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    component.pre_act(_NEXT_ACTING_SPEC)
    log_call = component._logging_channel.call_args
    log_data = log_call[0][0]
    self.assertIn('Initial Events', log_data)
    self.assertIn('Initial Entity States', log_data)

  def test_no_snapshot_on_second_call(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_next_acting.InterruptNextActing()
    _wire(component, scheduler)

    # First call consumes Alice's initial timer.
    component.pre_act(_NEXT_ACTING_SPEC)
    # Set a new timer so a second call has something to process.
    scheduler.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=_START + datetime.timedelta(minutes=30),
            entity_name='Alice',
            description='second timer',
        ),
    )
    component.pre_act(_NEXT_ACTING_SPEC)

    # The second log call should NOT contain the snapshot keys.
    log_call = component._logging_channel.call_args
    log_data = log_call[0][0]
    self.assertNotIn('Initial Events', log_data)
    self.assertNotIn('Initial Entity States', log_data)


if __name__ == '__main__':
  absltest.main()

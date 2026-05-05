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

"""Unit tests for the InterruptResolution component."""

import datetime
from unittest import mock

from absl.testing import absltest
from concordia.components import agent as actor_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import interrupt_resolution
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model
from concordia.typing import entity as entity_lib


_START = datetime.datetime(2026, 1, 1, 9, 0)

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG

_RESOLVE_SPEC = entity_lib.ActionSpec(
    call_to_action='',
    output_type=entity_lib.OutputType.RESOLVE,
)

_WRONG_SPEC = entity_lib.ActionSpec(
    call_to_action='',
    output_type=entity_lib.OutputType.NEXT_ACTING,
    options=('Alice',),
)


def _make_scheduler(
    names: list[str],
    start: datetime.datetime = _START,
) -> interrupt_scheduling.EntityScheduler:
  return interrupt_scheduling.EntityScheduler(
      player_names=names,
      time_model=interrupt_time_model.DatetimeTimeModel(start),
  )


def _wire(component, scheduler, memory_contents=None):
  """Wires a component to a mock entity with scheduler and mock memory."""
  mock_memory = mock.MagicMock(spec=actor_components.memory.Memory)
  mock_memory.scan.return_value = memory_contents or []

  components = {
      interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY: scheduler,
      actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: mock_memory,
  }
  mock_entity = mock.MagicMock()
  mock_entity.get_component.side_effect = lambda key, type_=None: components[
      key
  ]
  component.set_entity(mock_entity)
  component._logging_channel = mock.MagicMock()
  return mock_memory


class WrongOutputTypeTest(absltest.TestCase):

  def test_returns_empty_for_non_resolve(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    _wire(component, scheduler)

    result = component.pre_act(_WRONG_SPEC)
    self.assertEqual(result, '')


class NoPutativeEventsTest(absltest.TestCase):

  def test_returns_empty_when_no_suggestions(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    _wire(component, scheduler, memory_contents=[])

    result = component.pre_act(_RESOLVE_SPEC)
    self.assertEqual(result, '')


class MaskUpdateTest(absltest.TestCase):
  """Tests that entity masks are updated from parsed responses."""

  def test_mask_set_from_response(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I focus deeply.\n'
        '{"mask": ["emergency."], "timer":'
        ' {"time": "2h", "reason": "deep work"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    mask = scheduler.get_mask('Alice')
    self.assertEqual(mask.prefixes, ('emergency.',))


class TimerUpdateTest(absltest.TestCase):
  """Tests that entity timers are set from parsed responses."""

  def test_timer_set_from_response(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I look around.\n'
        '{"mask": [""], "timer":'
        ' {"time": "30m", "reason": "check again"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    timer = scheduler.get_timer('Alice')
    self.assertIsNotNone(timer)
    self.assertEqual(
        timer.expiry,
        _START + datetime.timedelta(minutes=30),
    )
    self.assertEqual(timer.description, 'check again')


class ActionEventInjectionTest(absltest.TestCase):
  """Tests that action events are injected into the event queue."""

  def test_action_event_injected(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
        event_tag_for_actions='action',
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I wave hello.\n'
        '{"mask": [""], "timer":'
        ' {"time": "1h", "reason": "idle"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    queue = scheduler.peek_event_queue()
    self.assertLen(queue, 1)
    self.assertEqual(queue[0].tag, 'action.Alice')
    self.assertEqual(queue[0].source, 'Alice')
    self.assertIn('I wave hello', queue[0].description)

  def test_no_event_for_empty_action(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: '
        '{"mask": [""], "timer":'
        ' {"time": "1h", "reason": "idle"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    queue = scheduler.peek_event_queue()
    self.assertEmpty(queue)


class TaggedEventInjectionTest(absltest.TestCase):
  """Tests that extra tagged events are injected from parsed responses."""

  def test_extra_tags_injected(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
        event_tag_for_actions='action',
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: Fire alarm!\n'
        '{"mask": [""], "tags": ["alert.fire", "emergency.evac"],'
        ' "timer": {"time": "0m", "reason": "evacuate"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    queue = scheduler.peek_event_queue()
    tags = [e.tag for e in queue]
    # Should have: action.Alice, alert.fire, emergency.evac.
    self.assertIn('action.Alice', tags)
    self.assertIn('alert.fire', tags)
    self.assertIn('emergency.evac', tags)
    self.assertLen(queue, 3)


class NonPolledEntityIgnoredTest(absltest.TestCase):
  """Tests that non-polled entities' responses are skipped."""

  def test_non_polled_entity_skipped(self):
    scheduler = _make_scheduler(['Alice', 'Bob'])
    # Only Bob is polled.
    scheduler.set_current_polled_entities(['Bob'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice', 'Bob'],
        event_tag_for_actions='action',
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I study quietly.\n'
        '{"mask": [""], "timer": {"time": "1h", "reason": "study"}}'
        ' Bob: I watch the room.\n'
        '{"mask": [""], "timer": {"time": "15m", "reason": "patrol"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    # Only Bob's action should be injected.
    queue = scheduler.peek_event_queue()
    tags = [e.tag for e in queue]
    self.assertIn('action.Bob', tags)
    self.assertNotIn('action.Alice', tags)


class MultiEntityResolutionTest(absltest.TestCase):
  """Tests resolution with multiple polled entities."""

  def test_both_entities_resolved(self):
    scheduler = _make_scheduler(['Alice', 'Bob'])
    scheduler.set_current_polled_entities(['Alice', 'Bob'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice', 'Bob'],
        event_tag_for_actions='action',
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I say hello.\n'
        '{"mask": ["chat."], "timer": {"time": "30m", "reason": "chat"}}'
        ' Bob: I nod.\n'
        '{"mask": [""], "timer": {"time": "1h", "reason": "idle"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    result = component.pre_act(_RESOLVE_SPEC)

    # Both entities should appear in the result.
    self.assertIn('Alice', result)
    self.assertIn('Bob', result)

    # Both masks should be updated.
    self.assertEqual(scheduler.get_mask('Alice').prefixes, ('chat.',))
    self.assertEqual(scheduler.get_mask('Bob'), interrupt_scheduling.MATCH_ALL)

    # Both timers should be set.
    self.assertIsNotNone(scheduler.get_timer('Alice'))
    self.assertIsNotNone(scheduler.get_timer('Bob'))


class ActiveEntityTrackingTest(absltest.TestCase):
  """Tests get_active_entity_name and get_putative_action."""

  def test_tracks_last_resolved_entity(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I wave.\n'
        '{"mask": [""], "timer": {"time": "1h", "reason": "idle"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    self.assertEqual(component.get_active_entity_name(), 'Alice')
    self.assertEqual(component.get_putative_action(), 'I wave.')


class TimerAbsoluteTest(absltest.TestCase):
  """Tests that timer_absolute is used over timer_duration_str."""

  def test_timer_absolute_used_over_duration(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    # Both "time" and "until" present — "until" should win.
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I sleep.\n'
        '{"mask": [], "timer":'
        ' {"time": "1h", "until": "14:00", "reason": "nap"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    timer = scheduler.get_timer('Alice')
    self.assertIsNotNone(timer)
    # _START is 09:00 on 2026-01-01; "14:00" should be same-day 14:00.
    self.assertEqual(
        timer.expiry,
        datetime.datetime(2026, 1, 1, 14, 0),
    )

  def test_timer_absolute_fallback_on_bad_parse(self):
    scheduler = _make_scheduler(['Alice'])
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_resolution.InterruptResolution(
        player_names=['Alice'],
    )
    # "until" is garbage, "time" is valid — should fall back to duration.
    putative = (
        f'{PUTATIVE_EVENT_TAG} Alice: I rest.\n'
        '{"mask": [""], "timer":'
        ' {"time": "1h", "until": "garbage", "reason": "rest"}}'
    )
    _wire(component, scheduler, memory_contents=[putative])

    component.pre_act(_RESOLVE_SPEC)

    timer = scheduler.get_timer('Alice')
    self.assertIsNotNone(timer)
    # Fallback: _START + 1h = 10:00.
    self.assertEqual(
        timer.expiry,
        _START + datetime.timedelta(hours=1),
    )


if __name__ == '__main__':
  absltest.main()

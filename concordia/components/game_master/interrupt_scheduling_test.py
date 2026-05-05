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

"""Tests for interrupt_scheduling module."""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model


def _event(
    hour: int = 9,
    minute: int = 0,
    tag: str = 'test',
    source: str = '__external__',
    description: str = 'test event',
) -> interrupt_scheduling.Event:
  """Creates a test event at a given hour:minute on 2026-01-01."""
  return interrupt_scheduling.Event(
      timestamp=datetime.datetime(2026, 1, 1, hour, minute),
      tag=tag,
      source=source,
      description=description,
  )


class MaskFromPrefixesTest(parameterized.TestCase):

  def test_empty_string_is_match_all(self):
    self.assertEqual(
        interrupt_scheduling.mask_from_prefixes(['']),
        interrupt_scheduling.MATCH_ALL,
    )

  def test_empty_list_is_match_none(self):
    self.assertEqual(
        interrupt_scheduling.mask_from_prefixes([]),
        interrupt_scheduling.MATCH_NONE,
    )

  def test_prefixes(self):
    result = interrupt_scheduling.mask_from_prefixes(['chat.direct', 'alarm.'])
    self.assertEqual(result.prefixes, ('chat.direct', 'alarm.'))


class InterruptMaskTest(parameterized.TestCase):
  """Tests for InterruptMask prefix matching."""

  def test_match_all_matches_everything(self):
    mask = interrupt_scheduling.MATCH_ALL
    self.assertTrue(mask.matches(_event(tag='anything')))
    self.assertTrue(mask.matches(_event(tag='')))
    self.assertTrue(mask.matches(_event(tag='chat.direct')))

  def test_match_none_matches_nothing(self):
    mask = interrupt_scheduling.MATCH_NONE
    self.assertFalse(mask.matches(_event(tag='anything')))
    self.assertFalse(mask.matches(_event(tag='')))
    self.assertFalse(mask.matches(_event(tag='chat.direct')))

  def test_single_prefix(self):
    mask = interrupt_scheduling.InterruptMask(prefixes=('chat.',))
    self.assertTrue(mask.matches(_event(tag='chat.direct')))
    self.assertTrue(mask.matches(_event(tag='chat.group')))
    self.assertFalse(mask.matches(_event(tag='alarm.fire')))

  def test_multiple_prefixes(self):
    mask = interrupt_scheduling.InterruptMask(prefixes=('chat.', 'alarm.'))
    self.assertTrue(mask.matches(_event(tag='chat.direct')))
    self.assertTrue(mask.matches(_event(tag='alarm.fire')))
    self.assertFalse(mask.matches(_event(tag='action.move')))

  def test_exact_match(self):
    mask = interrupt_scheduling.InterruptMask(prefixes=('alarm.fire',))
    self.assertTrue(mask.matches(_event(tag='alarm.fire')))
    self.assertTrue(mask.matches(_event(tag='alarm.fire.kitchen')))
    self.assertFalse(mask.matches(_event(tag='alarm.flood')))

  def test_empty_string_prefix_matches_all(self):
    mask = interrupt_scheduling.InterruptMask(prefixes=('',))
    self.assertTrue(mask.matches(_event(tag='anything')))


class EventOrderingTest(absltest.TestCase):
  """Tests for Event natural ordering."""

  def test_events_ordered_by_timestamp(self):
    early = _event(hour=9)
    late = _event(hour=10)
    self.assertLess(early, late)

  def test_events_same_time_ordered_by_tag(self):
    a = _event(tag='aaa')
    b = _event(tag='zzz')
    self.assertLess(a, b)

  def test_frozen(self):
    e = _event()
    with self.assertRaises(AttributeError):
      e.tag = 'modified'  # type: ignore


class EntitySchedulerTest(absltest.TestCase):
  """Tests for EntityScheduler."""

  def _make_scheduler(
      self,
      names: list[str] | None = None,
      hour: int = 9,
  ) -> interrupt_scheduling.EntityScheduler:
    names = names or ['Alice', 'Bob']
    return interrupt_scheduling.EntityScheduler(
        player_names=names,
        time_model=interrupt_time_model.DatetimeTimeModel(
            datetime.datetime(2026, 1, 1, hour, 0)
        ),
    )

  # ── Time management ────────────────────────────────────────────────

  def test_advance_time(self):
    s = self._make_scheduler()
    new_time = datetime.datetime(2026, 1, 1, 10, 0)
    s.advance_time(new_time)
    self.assertEqual(s.get_current_time(), new_time)

  def test_advance_time_backwards_raises(self):
    s = self._make_scheduler()
    with self.assertRaises(ValueError):
      s.advance_time(datetime.datetime(2026, 1, 1, 8, 0))

  def test_advance_time_same_is_ok(self):
    s = self._make_scheduler()
    s.advance_time(s.get_current_time())  # Should not raise.

  # ── Event queue ────────────────────────────────────────────────────

  def test_inject_and_pop_events_in_order(self):
    s = self._make_scheduler()
    e1 = _event(hour=10, tag='first')
    e2 = _event(hour=11, tag='second')
    e3 = _event(hour=10, minute=30, tag='middle')
    s.inject_event(e2)
    s.inject_event(e1)
    s.inject_event(e3)

    self.assertEqual(s.pop_next_event(), e1)
    self.assertEqual(s.pop_next_event(), e3)
    self.assertEqual(s.pop_next_event(), e2)

  def test_peek_does_not_remove(self):
    s = self._make_scheduler()
    e = _event(hour=10)
    s.inject_event(e)
    self.assertEqual(s.peek_event_queue(), [e])
    self.assertEqual(s.peek_event_queue(), [e])

  def test_pop_empty_returns_none(self):
    s = self._make_scheduler()
    self.assertIsNone(s.pop_next_event())

  def test_inject_past_event_raises(self):
    s = self._make_scheduler(hour=10)
    with self.assertRaises(ValueError):
      s.inject_event(_event(hour=9))

  def test_inject_at_current_time_ok(self):
    s = self._make_scheduler(hour=9)
    s.inject_event(_event(hour=9))  # Should not raise.

  # ── Masks ──────────────────────────────────────────────────────────

  def test_default_masks_are_match_all(self):
    s = self._make_scheduler()
    self.assertEqual(s.get_mask('Alice'), interrupt_scheduling.MATCH_ALL)

  def test_set_and_get_mask(self):
    s = self._make_scheduler()
    custom = interrupt_scheduling.InterruptMask(prefixes=('chat.',))
    s.set_mask('Alice', custom)
    self.assertEqual(s.get_mask('Alice'), custom)

  def test_clear_mask_resets_to_match_all(self):
    s = self._make_scheduler()
    s.set_mask(
        'Alice',
        interrupt_scheduling.MATCH_NONE,
    )
    s.clear_mask('Alice')
    self.assertEqual(s.get_mask('Alice'), interrupt_scheduling.MATCH_ALL)

  # ── Timers ─────────────────────────────────────────────────────────

  def test_initial_timers_expire_at_start(self):
    s = self._make_scheduler(hour=9)
    timer = s.get_timer('Alice')
    self.assertIsNotNone(timer)
    self.assertEqual(timer.expiry, datetime.datetime(2026, 1, 1, 9, 0))

  def test_set_timer_replaces(self):
    s = self._make_scheduler()
    new_timer = interrupt_scheduling.Timer(
        expiry=datetime.datetime(2026, 1, 1, 11, 0),
        entity_name='Alice',
        description='new timer',
    )
    s.set_timer('Alice', new_timer)
    self.assertEqual(s.get_timer('Alice'), new_timer)

  def test_clear_timer_sets_none(self):
    s = self._make_scheduler()
    s.clear_timer('Alice')
    self.assertIsNone(s.get_timer('Alice'))

  def test_get_next_timer_returns_earliest(self):
    s = self._make_scheduler()
    s.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=datetime.datetime(2026, 1, 1, 12, 0),
            entity_name='Alice',
            description='later',
        ),
    )
    s.set_timer(
        'Bob',
        interrupt_scheduling.Timer(
            expiry=datetime.datetime(2026, 1, 1, 10, 0),
            entity_name='Bob',
            description='sooner',
        ),
    )
    next_timer = s.get_next_timer()
    self.assertIsNotNone(next_timer)
    self.assertEqual(next_timer.entity_name, 'Bob')

  def test_get_next_timer_skips_none(self):
    s = self._make_scheduler()
    s.clear_timer('Alice')
    s.set_timer(
        'Bob',
        interrupt_scheduling.Timer(
            expiry=datetime.datetime(2026, 1, 1, 10, 0),
            entity_name='Bob',
            description='only',
        ),
    )
    next_timer = s.get_next_timer()
    self.assertIsNotNone(next_timer)
    self.assertEqual(next_timer.entity_name, 'Bob')

  def test_get_next_timer_all_none_returns_none(self):
    s = self._make_scheduler()
    s.clear_timer('Alice')
    s.clear_timer('Bob')
    self.assertIsNone(s.get_next_timer())

  # ── Pending ────────────────────────────────────────────────────────

  def test_pending_accumulates_and_drains(self):
    s = self._make_scheduler()
    e1 = _event(hour=10, tag='first')
    e2 = _event(hour=11, tag='second')
    s.add_pending('Alice', e1)
    s.add_pending('Alice', e2)
    pending = s.drain_pending('Alice')
    self.assertEqual(pending, [e1, e2])
    # Second drain should be empty.
    self.assertEqual(s.drain_pending('Alice'), [])

  # ── Polling ────────────────────────────────────────────────────────

  def test_poll_clears_mask_and_timer(self):
    s = self._make_scheduler()
    s.set_mask(
        'Alice',
        interrupt_scheduling.MATCH_NONE,
    )
    s.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=datetime.datetime(2026, 1, 1, 11, 0),
            entity_name='Alice',
            description='test',
        ),
    )
    s.poll_entity('Alice')
    # Mask resets to MATCH_ALL, timer clears.
    self.assertEqual(s.get_mask('Alice'), interrupt_scheduling.MATCH_ALL)
    self.assertIsNone(s.get_timer('Alice'))

  def test_get_entities_to_poll_mask_match(self):
    s = self._make_scheduler()
    # Alice listens to chat; Bob listens to nothing.
    s.set_mask(
        'Alice',
        interrupt_scheduling.InterruptMask(prefixes=('chat.',)),
    )
    s.set_mask('Bob', interrupt_scheduling.MATCH_NONE)
    # Clear timers so they don't confuse the test.
    s.clear_timer('Alice')
    s.clear_timer('Bob')

    event = _event(tag='chat.direct')
    polled = s.get_entities_to_poll(event)
    self.assertEqual(polled, ['Alice'])

  def test_get_entities_to_poll_timer_expiry_private(self):
    s = self._make_scheduler()
    # Clear default timers.
    s.clear_timer('Alice')
    s.clear_timer('Bob')
    # Timer event for Alice only polls Alice.
    tag = f'{interrupt_scheduling.TIMER_EXPIRED_TAG_PREFIX}Alice'
    event = _event(tag=tag)
    polled = s.get_entities_to_poll(event)
    self.assertEqual(polled, ['Alice'])

  def test_get_entities_to_poll_includes_expired_timers(self):
    s = self._make_scheduler()
    # Alice has a timer expiring at 10:00.
    s.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=datetime.datetime(2026, 1, 1, 10, 0),
            entity_name='Alice',
            description='test',
        ),
    )
    # Bob has no timer and MATCH_NONE mask.
    s.clear_timer('Bob')
    s.set_mask('Bob', interrupt_scheduling.MATCH_NONE)

    # An event at 10:30 with a tag Alice doesn't match.
    event = _event(hour=10, minute=30, tag='unrelated')
    s.set_mask(
        'Alice',
        interrupt_scheduling.MATCH_NONE,
    )
    polled = s.get_entities_to_poll(event)
    # Alice should be polled due to expired timer.
    self.assertIn('Alice', polled)
    # Bob should NOT be polled.
    self.assertNotIn('Bob', polled)

  # ── Tick / termination ─────────────────────────────────────────────

  def test_tick_counting(self):
    s = self._make_scheduler()
    self.assertEqual(s.get_tick_count(), 0)
    s.increment_tick()
    s.increment_tick()
    self.assertEqual(s.get_tick_count(), 2)

  def test_termination_with_max_ticks(self):
    s = interrupt_scheduling.EntityScheduler(
        player_names=['Alice'],
        time_model=interrupt_time_model.DatetimeTimeModel(
            datetime.datetime(2026, 1, 1, 9, 0)
        ),
        max_ticks=3,
    )
    self.assertFalse(s.should_terminate())
    s.increment_tick()
    s.increment_tick()
    self.assertFalse(s.should_terminate())
    s.increment_tick()
    self.assertTrue(s.should_terminate())

  def test_termination_without_max_ticks(self):
    s = self._make_scheduler()
    for _ in range(100):
      s.increment_tick()
    self.assertFalse(s.should_terminate())

  # ── State serialisation ────────────────────────────────────────────

  def test_get_set_state_roundtrip(self):
    s = self._make_scheduler()
    # Set up some state.
    s.set_mask(
        'Alice',
        interrupt_scheduling.InterruptMask(prefixes=('chat.',)),
    )
    s.set_timer(
        'Alice',
        interrupt_scheduling.Timer(
            expiry=datetime.datetime(2026, 1, 1, 11, 0),
            entity_name='Alice',
            description='test timer',
        ),
    )
    s.inject_event(_event(hour=10, tag='saved_event'))
    s.add_pending('Bob', _event(hour=9, tag='pending'))
    s.increment_tick()

    state = s.get_state()

    # Restore into a new scheduler.
    s2 = self._make_scheduler()
    s2.set_state(state)

    self.assertEqual(
        s2.get_mask('Alice'),
        interrupt_scheduling.InterruptMask(prefixes=('chat.',)),
    )
    alice_timer = s2.get_timer('Alice')
    self.assertIsNotNone(alice_timer)
    self.assertEqual(alice_timer.description, 'test timer')
    queue = s2.peek_event_queue()
    self.assertLen(queue, 1)
    self.assertEqual(queue[0].tag, 'saved_event')
    self.assertEqual(s2.get_tick_count(), 1)


if __name__ == '__main__':
  absltest.main()

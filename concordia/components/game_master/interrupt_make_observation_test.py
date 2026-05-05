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

"""Unit tests for the InterruptMakeObservation component."""

import datetime
from unittest import mock

from absl.testing import absltest
from concordia.components.game_master import interrupt_make_observation
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model
from concordia.typing import entity as entity_lib


_START = datetime.datetime(2026, 1, 1, 9, 0)

_MAKE_OBS_SPEC = entity_lib.ActionSpec(
    call_to_action=(
        'What is the current situation faced by Alice?'
        ' What do they now observe?'
        ' Only include information of which they are aware.'
    ),
    output_type=entity_lib.OutputType.MAKE_OBSERVATION,
)

_WRONG_SPEC = entity_lib.ActionSpec(
    call_to_action='',
    output_type=entity_lib.OutputType.RESOLVE,
)


def _make_scheduler(
    names: list[str],
    start: datetime.datetime = _START,
) -> interrupt_scheduling.EntityScheduler:
  return interrupt_scheduling.EntityScheduler(
      player_names=names,
      time_model=interrupt_time_model.DatetimeTimeModel(start),
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
  component._logging_channel = mock.MagicMock()


class WrongOutputTypeTest(absltest.TestCase):

  def test_returns_empty_for_non_make_observation(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_make_observation.InterruptMakeObservation()
    _wire(component, scheduler)

    result = component.pre_act(_WRONG_SPEC)
    self.assertEqual(result, '')


class NoCurrentEventTest(absltest.TestCase):

  def test_returns_empty_when_no_event(self):
    scheduler = _make_scheduler(['Alice'])
    component = interrupt_make_observation.InterruptMakeObservation()
    _wire(component, scheduler)

    result = component.pre_act(_MAKE_OBS_SPEC)
    self.assertEqual(result, '')


class NotPolledTest(absltest.TestCase):

  def test_returns_empty_when_entity_not_polled(self):
    scheduler = _make_scheduler(['Alice', 'Bob'])
    event = interrupt_scheduling.Event(
        timestamp=_START,
        tag='chat.hello',
        source='__external__',
        description='Hello everyone.',
    )
    scheduler.set_current_event(event)
    # Only Bob is polled, not Alice.
    scheduler.set_current_polled_entities(['Bob'])

    component = interrupt_make_observation.InterruptMakeObservation()
    _wire(component, scheduler)

    result = component.pre_act(_MAKE_OBS_SPEC)
    self.assertEqual(result, '')


class EntityNameExtractionTest(absltest.TestCase):
  """Tests _get_entity_name with various call_to_action formats."""

  def test_standard_format(self):
    component = interrupt_make_observation.InterruptMakeObservation()
    name = component._get_entity_name(
        'What is the current situation faced by Alice?'
        ' What do they now observe?'
        ' Only include information of which they are aware.'
    )
    self.assertEqual(name, 'Alice')

  def test_multi_word_name(self):
    component = interrupt_make_observation.InterruptMakeObservation()
    name = component._get_entity_name(
        'What is the current situation faced by Bob The Builder?'
        ' What do they now observe?'
        ' Only include information of which they are aware.'
    )
    self.assertEqual(name, 'Bob The Builder')

  def test_fallback_to_full_string(self):
    component = interrupt_make_observation.InterruptMakeObservation()
    name = component._get_entity_name('Unrecognised format')
    self.assertEqual(name, 'Unrecognised format')


class BasicObservationTest(absltest.TestCase):
  """Tests the observation content for a polled entity."""

  def test_includes_time_and_event(self):
    scheduler = _make_scheduler(['Alice'])
    event = interrupt_scheduling.Event(
        timestamp=_START,
        tag='chat.hello',
        source='Bob',
        description='Bob says hello.',
    )
    scheduler.set_current_event(event)
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_make_observation.InterruptMakeObservation()
    _wire(component, scheduler)

    result = component.pre_act(_MAKE_OBS_SPEC)
    self.assertIn('[Current time:', result)
    self.assertIn('Bob says hello.', result)


class PendingEventsTest(absltest.TestCase):
  """Tests that pending events are included as a catch-up summary."""

  def test_pending_events_in_observation(self):
    scheduler = _make_scheduler(['Alice'])
    # Add a pending event.
    pending_event = interrupt_scheduling.Event(
        timestamp=_START - datetime.timedelta(minutes=10),
        tag='announcement.daily',
        source='__external__',
        description='The morning announcement.',
    )
    scheduler.add_pending('Alice', pending_event)

    # Set current event and polled entities.
    current_event = interrupt_scheduling.Event(
        timestamp=_START,
        tag='chat.hello',
        source='Bob',
        description='Bob says hello.',
    )
    scheduler.set_current_event(current_event)
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_make_observation.InterruptMakeObservation()
    _wire(component, scheduler)

    result = component.pre_act(_MAKE_OBS_SPEC)
    self.assertIn('Events missed while focused', result)
    self.assertIn('The morning announcement.', result)
    self.assertIn('Bob says hello.', result)

  def test_no_pending_events_no_catch_up(self):
    scheduler = _make_scheduler(['Alice'])
    event = interrupt_scheduling.Event(
        timestamp=_START,
        tag='chat.hello',
        source='Bob',
        description='Bob says hello.',
    )
    scheduler.set_current_event(event)
    scheduler.set_current_polled_entities(['Alice'])

    component = interrupt_make_observation.InterruptMakeObservation()
    _wire(component, scheduler)

    result = component.pre_act(_MAKE_OBS_SPEC)
    self.assertNotIn('Events missed while focused', result)


if __name__ == '__main__':
  absltest.main()

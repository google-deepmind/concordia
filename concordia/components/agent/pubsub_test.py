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

"""Tests for ReactiveX Publisher and Subscriber agent components."""

import threading
from typing import override

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import constant
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.components.agent import pubsub
from concordia.components.agent import report_function
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from reactivex import subject as reactivex_subject


class _ContextEchoActComponent(entity_component.ActingComponent):
  """Acting component that returns the non-empty component contexts joined."""

  @override
  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    return '\n'.join(val for val in context.values() if val)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class _SlowDependentComponent(action_spec_ignored.ActionSpecIgnored):
  """Component that reads a sibling component and blocks on an event."""

  def __init__(
      self,
      *,
      dependency_key: str,
      started_event: threading.Event,
      release_event: threading.Event,
      pre_act_label: str = 'Dependent',
  ):
    super().__init__(pre_act_label)
    self._dependency_key = dependency_key
    self._started_event = started_event
    self._release_event = release_event
    self.observations: list[str] = []

  @override
  def pre_observe(self, observation: str) -> str:
    self.observations.append(observation)
    return ''

  @override
  def _make_pre_act_value(self) -> str:
    self._started_event.set()
    self._release_event.wait(timeout=5.0)
    dep_value = self.get_named_component_pre_act_value(self._dependency_key)
    return f'derived from {dep_value}'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'observations': list(self.observations)}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    observations = state.get('observations')
    if isinstance(observations, list):
      self.observations = [str(item) for item in observations]


class PubSubTest(parameterized.TestCase):

  def test_entity_acts_with_previous_value_until_slow_computation_finishes(
      self,
  ):
    started_events = [threading.Event(), threading.Event()]
    release_events = [threading.Event(), threading.Event()]
    call_count = 0
    call_lock = threading.Lock()

    def slow_computation() -> str:
      nonlocal call_count
      with call_lock:
        idx = call_count
        call_count += 1
      if idx < len(started_events):
        started_events[idx].set()
        release_events[idx].wait(timeout=5.0)
      return f'slow_value_{idx + 1}'

    publisher = pubsub.Publisher(
        function=slow_computation,
        skip_if_busy=True,
    )
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__',
        initial_value='initial_value',
        pre_act_label='Slow Insight',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Alice',
        act_component=_ContextEchoActComponent(),
        context_components={
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      # First act: triggers slow_computation #1, but acts immediately with
      # 'initial_value'.
      action_1 = agent.act()
      self.assertEqual(action_1, 'Slow Insight:\ninitial_value\n')
      self.assertTrue(started_events[0].wait(timeout=5.0))
      self.assertTrue(publisher.is_busy())

      # Second act while computation #1 is still running: acts immediately
      # using 'initial_value' and skips triggering a duplicate computation.
      action_2 = agent.act()
      self.assertEqual(action_2, 'Slow Insight:\ninitial_value\n')
      self.assertEqual(call_count, 1)

      # Allow computation #1 to complete and wait for publication.
      release_events[0].set()
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertFalse(publisher.is_busy())
      self.assertEqual(subscriber.get_current_value(), 'slow_value_1')

      # Third act: uses 'slow_value_1' and triggers computation #2.
      action_3 = agent.act()
      self.assertEqual(action_3, 'Slow Insight:\nslow_value_1\n')
      self.assertTrue(started_events[1].wait(timeout=5.0))

      # Allow computation #2 to complete; fourth act uses 'slow_value_2'.
      release_events[1].set()
      self.assertTrue(publisher.wait(timeout=5.0))
      release_events.append(threading.Event())
      release_events[2].set()
      started_events.append(threading.Event())
      action_4 = agent.act()
      self.assertEqual(action_4, 'Slow Insight:\nslow_value_2\n')
      self.assertTrue(publisher.wait(timeout=5.0))
    finally:
      for ev in release_events:
        ev.set()
      subscriber.close()
      publisher.close()

  def test_skip_if_busy_false_runs_every_trigger_and_keeps_latest_sequence(
      self,
  ):
    started_1 = threading.Event()
    release_1 = threading.Event()
    started_2 = threading.Event()
    release_2 = threading.Event()
    call_count = 0
    call_lock = threading.Lock()

    def variable_speed_computation() -> str:
      nonlocal call_count
      with call_lock:
        call_count += 1
        current_call = call_count
      if current_call == 1:
        started_1.set()
        release_1.wait(timeout=5.0)
        return 'stale_value_from_call_1'
      if current_call == 2:
        started_2.set()
        release_2.wait(timeout=5.0)
        return 'fresh_value_from_call_2'
      return f'value_from_call_{current_call}'

    publisher = pubsub.Publisher(
        function=variable_speed_computation,
        skip_if_busy=False,
    )
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__',
        initial_value='init',
        pre_act_label='Async Thought',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Bob',
        act_component=_ContextEchoActComponent(),
        context_components={
            'subscriber': subscriber,
            '__publisher__': publisher,
        },
    )

    try:
      # Act 1 starts call 1 (which is slow).
      self.assertEqual(agent.act(), 'Async Thought:\ninit\n')
      self.assertTrue(started_1.wait(timeout=5.0))

      # Act 2 starts call 2 concurrently even though call 1 is still busy.
      self.assertEqual(agent.act(), 'Async Thought:\ninit\n')
      self.assertTrue(started_2.wait(timeout=5.0))
      self.assertEqual(call_count, 2)

      # Call 2 finishes first!
      release_2.set()
      # Wait until subscriber receives call 2's value, then release call 1.
      release_1.set()
      self.assertTrue(publisher.wait(timeout=5.0))

      # Even if call 1 finishes after call 2, its older sequence ID must not
      # overwrite call 2's newer value.
      self.assertEqual(
          subscriber.get_current_value(), 'fresh_value_from_call_2'
      )
      self.assertEqual(agent.act(), 'Async Thought:\nfresh_value_from_call_2\n')
      self.assertTrue(publisher.wait(timeout=5.0))
    finally:
      release_1.set()
      release_2.set()
      subscriber.close()
      publisher.close()

  def test_stale_out_of_order_completion_is_dropped(self):
    started_1 = threading.Event()
    release_1 = threading.Event()
    started_2 = threading.Event()
    call_2_published = threading.Event()
    call_count = 0
    call_lock = threading.Lock()

    def out_of_order_computation() -> str:
      nonlocal call_count
      with call_lock:
        call_count += 1
        idx = call_count
      if idx == 1:
        started_1.set()
        release_1.wait(timeout=5.0)
        return 'older_result_1'
      started_2.set()
      return 'newer_result_2'

    publisher = pubsub.Publisher(
        function=out_of_order_computation,
        skip_if_busy=False,
    )
    subscriber = pubsub.Subscriber(
        publisher=publisher,
        initial_value='start',
    )
    publisher.observable.subscribe(
        on_next=lambda val: (
            call_2_published.set() if val == 'newer_result_2' else None
        )
    )
    agent = entity_agent.EntityAgent(
        agent_name='Carol',
        act_component=_ContextEchoActComponent(),
        context_components={
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      # Trigger #1 (blocked on release_1).
      agent.act()
      self.assertTrue(started_1.wait(timeout=5.0))

      # Trigger #2 (completes immediately while #1 is still blocked).
      agent.act()
      self.assertTrue(started_2.wait(timeout=5.0))
      self.assertTrue(call_2_published.wait(timeout=5.0))
      self.assertEqual(subscriber.get_current_value(), 'newer_result_2')

      # Now release #1; it completes after #2, so its result must be ignored.
      release_1.set()
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertEqual(subscriber.get_current_value(), 'newer_result_2')
    finally:
      release_1.set()
      subscriber.close()
      publisher.close()

  def test_publisher_wrapping_component_with_sibling_dependency(self):
    started_event = threading.Event()
    release_event = threading.Event()
    goal_component = constant.Constant(
        state='build a shelter', pre_act_label='Goal'
    )
    slow_component = _SlowDependentComponent(
        dependency_key='goal',
        started_event=started_event,
        release_event=release_event,
    )
    publisher = pubsub.Publisher(component=slow_component)
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__',
        initial_value='no derived plan yet',
        pre_act_label='Derived Plan',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Dave',
        act_component=_ContextEchoActComponent(),
        context_components={
            'goal': goal_component,
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      # Observations are forwarded to the wrapped component.
      agent.observe('It is starting to rain.')
      self.assertEqual(slow_component.observations, ['It is starting to rain.'])

      # First act returns immediately while slow_component is still blocked.
      # By the time slow_component unblocks, `agent` is back in Phase.READY,
      # yet slow_component can still read the snapshotted 'goal' pre_act value.
      action_1 = agent.act()
      self.assertIn('Goal:\nbuild a shelter\n', action_1)
      self.assertIn('Derived Plan:\nno derived plan yet\n', action_1)
      self.assertTrue(started_event.wait(timeout=5.0))
      self.assertEqual(agent.get_phase(), entity_component.Phase.READY)

      release_event.set()
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertEqual(
          subscriber.get_current_value(), 'derived from build a shelter'
      )

      action_2 = agent.act()
      self.assertIn('Derived Plan:\nderived from build a shelter\n', action_2)
      self.assertTrue(publisher.wait(timeout=5.0))
    finally:
      release_event.set()
      subscriber.close()
      publisher.close()

  def test_shared_reactivex_subject_wiring(self):
    shared_subject = reactivex_subject.Subject()
    publisher = pubsub.Publisher(
        report_function.ReportFunction(function=lambda: 'via_shared_subject'),
        subject=shared_subject,
    )
    subscriber = pubsub.Subscriber(
        publisher_component_key=None,
        subject=shared_subject,
        initial_value='before_publish',
        pre_act_label='Shared',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Eve',
        act_component=_ContextEchoActComponent(),
        context_components={
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      # First act latches 'before_publish' and triggers the publisher.
      self.assertEqual(agent.act(), 'Shared:\nbefore_publish\n')
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertEqual(subscriber.get_current_value(), 'via_shared_subject')
      self.assertEqual(
          subscriber.get_dynamic_state(), {'value': 'via_shared_subject'}
      )

      # Second act uses the value published over the shared subject.
      self.assertEqual(agent.act(), 'Shared:\nvia_shared_subject\n')
      self.assertTrue(publisher.wait(timeout=5.0))
    finally:
      subscriber.close()
      publisher.close()

  def test_error_in_slow_computation_does_not_break_pipeline(self):
    should_fail = True

    def flaky_computation() -> str:
      if should_fail:
        raise RuntimeError('transient failure')
      return 'recovered_value'

    publisher = pubsub.Publisher(function=flaky_computation)
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__',
        initial_value='fallback_value',
        pre_act_label='Flaky',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Frank',
        act_component=_ContextEchoActComponent(),
        context_components={
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      # First act triggers failing computation; wait() still completes cleanly
      # and subscriber keeps 'fallback_value'.
      self.assertEqual(agent.act(), 'Flaky:\nfallback_value\n')
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertFalse(publisher.is_busy())
      self.assertEqual(subscriber.get_current_value(), 'fallback_value')

      # Next act succeeds and updates the subscriber.
      should_fail = False
      self.assertEqual(agent.act(), 'Flaky:\nfallback_value\n')
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertEqual(subscriber.get_current_value(), 'recovered_value')
      self.assertEqual(agent.act(), 'Flaky:\nrecovered_value\n')
      self.assertTrue(publisher.wait(timeout=5.0))
    finally:
      subscriber.close()
      publisher.close()

  def test_invalid_publisher_init_arguments_raise_value_error(self):
    with self.assertRaises(ValueError):
      pubsub.Publisher()

    with self.assertRaises(ValueError):
      pubsub.Publisher(
          function=lambda: 'a',
          component=constant.Constant(state='b'),
      )

    with self.assertRaises(ValueError):
      pubsub.Publisher(
          constant.Constant(state='a'),
          component=constant.Constant(state='b'),
      )

  def test_trigger_on_observe_starts_background_work_during_observe(self):
    started_event = threading.Event()
    release_event = threading.Event()
    call_count = 0
    call_lock = threading.Lock()

    def slow_computation() -> str:
      nonlocal call_count
      with call_lock:
        call_count += 1
        idx = call_count
      if idx == 1:
        started_event.set()
        release_event.wait(timeout=5.0)
      return f'computed_{idx}'

    publisher = pubsub.Publisher(
        function=slow_computation,
        skip_if_busy=True,
        trigger_on_observe=True,
    )
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__',
        initial_value='initial',
        pre_act_label='Insight',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Grace',
        act_component=_ContextEchoActComponent(),
        context_components={
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      self.assertTrue(publisher.trigger_on_observe)
      # Calling observe() immediately triggers background work before act()!
      agent.observe('First observation.')
      self.assertTrue(started_event.wait(timeout=5.0))
      self.assertTrue(publisher.is_busy())
      self.assertEqual(call_count, 1)

      # Let the observe-triggered computation finish before act() is called.
      release_event.set()
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertEqual(subscriber.get_current_value(), 'computed_1')
      self.assertEqual(publisher.compute_count, 1)

      # Now the very first act() already has 'computed_1', AND also triggers #2!
      action_1 = agent.act()
      self.assertEqual(action_1, 'Insight:\ncomputed_1\n')
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertEqual(subscriber.get_current_value(), 'computed_2')
      self.assertEqual(publisher.compute_count, 2)
    finally:
      release_event.set()
      subscriber.close()
      publisher.close()

  def test_trigger_on_observe_reads_flushed_observations_and_memory(self):
    memory = memory_component.ListMemory(memory_bank=[])
    obs_to_mem = observation_component.ObservationToMemory()
    last_obs = observation_component.LastNObservations(history_length=10)
    started_event = threading.Event()
    release_event = threading.Event()
    release_event.set()
    slow_component = _SlowDependentComponent(
        dependency_key=observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY,
        started_event=started_event,
        release_event=release_event,
    )
    publisher = pubsub.Publisher(
        component=slow_component,
        trigger_on_observe=True,
    )
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__',
        initial_value='none',
        pre_act_label='Derived',
    )
    agent = entity_agent.EntityAgent(
        agent_name='Heidi',
        act_component=_ContextEchoActComponent(),
        context_components={
            memory_component.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            'obs_to_mem': obs_to_mem,
            observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY: last_obs,
            '__publisher__': publisher,
            'subscriber': subscriber,
        },
    )

    try:
      agent.observe('Alice arrived at the coffee shop.')
      self.assertTrue(publisher.wait(timeout=5.0))
      self.assertIn(
          'Alice arrived at the coffee shop.',
          subscriber.get_current_value(),
      )
    finally:
      subscriber.close()
      publisher.close()


if __name__ == '__main__':
  absltest.main()

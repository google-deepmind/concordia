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

"""Tests for how EntityAgent recovers when a step fails."""

import threading

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component


class _StatelessMixin:

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class _EchoActComponent(_StatelessMixin, entity_component.ActingComponent):
  """Acts by returning the pre-act context of the `cached` component."""

  def get_action_attempt(self, context, action_spec) -> str:
    del action_spec
    return context['cached'].strip()


class _CachedValue(action_spec_ignored.ActionSpecIgnored):
  """Caches the current step number as its pre-act value."""

  def __init__(self):
    super().__init__('cached')
    self.step = 0
    self.computed = threading.Event()

  def _make_pre_act_value(self) -> str:
    value = f'step {self.step}'
    self.computed.set()
    return value

  def cache_lock(self) -> threading.Lock:
    return self._lock


class _FailingContext(_StatelessMixin, entity_component.ContextComponent):
  """Raises from the selected hook while `fail` is set."""

  def __init__(self, hook: str, error: Exception, wait_for=None):
    super().__init__()
    self.fail = True
    self.hook = hook
    self._error = error
    self._wait_for = wait_for

  def _maybe_fail(self, hook: str) -> None:
    if self.fail and hook == self.hook:
      if self._wait_for is not None:
        # Fail only after the other component has cached its value, so the
        # test does not depend on thread scheduling.
        self._wait_for.wait(timeout=5)
      raise self._error

  def pre_act(self, action_spec) -> str:
    self._maybe_fail('pre_act')
    return ''

  def pre_observe(self, observation: str) -> str:
    self._maybe_fail('pre_observe')
    return ''


class _PhaseRecorder(_StatelessMixin, entity_component.ContextComponent):
  """Records the entity phase every time `update` is called."""

  def __init__(self, raise_in_update: bool = False):
    super().__init__()
    self.update_phases = []
    self._raise_in_update = raise_in_update

  def update(self) -> None:
    self.update_phases.append(self.get_entity().get_phase())
    if self._raise_in_update:
      raise RuntimeError('update failed')


class EntityAgentFailedStepTest(absltest.TestCase):

  def test_failed_act_does_not_leak_cached_value_into_next_act(self):
    cached = _CachedValue()
    failing = _FailingContext(
        'pre_act', RuntimeError('model call failed'), cached.computed
    )
    agent = entity_agent.EntityAgent(
        'Alice', _EchoActComponent(), {'cached': cached, 'failing': failing}
    )

    with self.assertRaises(RuntimeError):
      agent.act()

    cached.step = 1
    failing.fail = False
    self.assertEqual(agent.act(), 'cached:\nstep 1')

  def test_call_waiting_on_cache_lock_does_not_cache_after_step_ends(self):
    # A component's pre_act can still be running when the step fails, because
    # the agent does not wait for it. If it is waiting for the cache lock while
    # the step is cleaned up, it must not cache a value once it gets the lock.
    cached = _CachedValue()
    agent = entity_agent.EntityAgent(
        'Alice', _EchoActComponent(), {'cached': cached}
    )
    agent.set_phase(entity_component.Phase.PRE_ACT)
    errors = []

    def read_value():
      try:
        cached.get_pre_act_value()
      except ValueError as error:
        errors.append(error)

    with cached.cache_lock():
      reader = threading.Thread(target=read_value)
      reader.start()
      reader.join(timeout=0.1)  # Let the reader block on the lock.
      agent.set_phase(entity_component.Phase.READY)
    reader.join(timeout=5)

    self.assertLen(errors, 1)
    self.assertFalse(cached.computed.is_set())

  def test_failed_act_reraises_the_original_exception(self):
    error = RuntimeError('model call failed')
    agent = entity_agent.EntityAgent(
        'Alice',
        _EchoActComponent(),
        {
            'cached': _CachedValue(),
            'failing': _FailingContext('pre_act', error),
        },
    )

    with self.assertRaises(RuntimeError) as context:
      agent.act()

    self.assertIs(context.exception, error)

  def test_failed_observe_reraises_the_original_exception(self):
    error = RuntimeError('memory write failed')
    agent = entity_agent.EntityAgent(
        'Alice',
        _EchoActComponent(),
        {'failing': _FailingContext('pre_observe', error)},
    )

    with self.assertRaises(RuntimeError) as context:
      agent.observe('Bob waves.')

    self.assertIs(context.exception, error)

  def test_failed_steps_run_update_in_update_phase_then_return_to_ready(self):
    recorder = _PhaseRecorder()
    failing = _FailingContext('pre_act', RuntimeError('act failed'))
    agent = entity_agent.EntityAgent(
        'Alice',
        _EchoActComponent(),
        {'cached': _CachedValue(), 'failing': failing, 'recorder': recorder},
    )

    with self.assertRaises(RuntimeError):
      agent.act()
    self.assertEqual(recorder.update_phases, [entity_component.Phase.UPDATE])
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)

    failing.hook = 'pre_observe'
    with self.assertRaises(RuntimeError):
      agent.observe('Bob waves.')
    self.assertLen(recorder.update_phases, 2)
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)

  def test_error_in_update_during_recovery_does_not_hide_original_error(self):
    error = ValueError('the real problem')
    agent = entity_agent.EntityAgent(
        'Alice',
        _EchoActComponent(),
        {
            'cached': _CachedValue(),
            'failing': _FailingContext('pre_act', error),
            'recorder': _PhaseRecorder(raise_in_update=True),
        },
    )

    with self.assertRaises(ValueError) as context:
      agent.act()

    self.assertIs(context.exception, error)
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)

  def test_agent_can_act_again_after_a_failed_act(self):
    failing = _FailingContext('pre_act', RuntimeError('act failed'))
    agent = entity_agent.EntityAgent(
        'Alice',
        _EchoActComponent(),
        {'cached': _CachedValue(), 'failing': failing},
    )

    with self.assertRaises(RuntimeError):
      agent.act()
    failing.fail = False

    self.assertEqual(agent.act(), 'cached:\nstep 0')
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)


if __name__ == '__main__':
  absltest.main()

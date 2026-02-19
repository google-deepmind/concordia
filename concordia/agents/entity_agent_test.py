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

"""Tests for EntityAgent."""

from unittest import mock

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.components.agent import no_op_context_processor
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class _StubActComponent(entity_component.ActingComponent):
  """Simple acting component used for tests."""

  def __init__(self, action_attempt: str) -> None:
    self._action_attempt = action_attempt

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del context, action_spec
    return self._action_attempt

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class EntityAgentTest(absltest.TestCase):

  def _make_agent(self, action_attempt: str = 'expected action') -> (
      entity_agent.EntityAgent
  ):
    return entity_agent.EntityAgent(
        agent_name='test-agent',
        act_component=_StubActComponent(action_attempt),
        context_processor=no_op_context_processor.NoOpContextProcessor(),
    )

  def test_stateless_act_requires_pre_act_phase(self):
    agent = self._make_agent()

    with self.assertRaisesRegex(RuntimeError, 'PRE_ACT'):
      agent.stateless_act(entity_lib.DEFAULT_ACTION_SPEC)

  def test_stateless_act_success(self):
    agent = self._make_agent(action_attempt='test action')
    agent.set_phase(entity_component.Phase.PRE_ACT)

    action = agent.stateless_act(entity_lib.DEFAULT_ACTION_SPEC)

    self.assertEqual(action, 'test action')

  def test_stateless_act_propagates_error_and_exits_executor_context(self):
    agent = self._make_agent()
    agent.set_phase(entity_component.Phase.PRE_ACT)

    class _TrackingExecutor:
      """ThreadPoolExecutor test double that records context usage."""

      instances = []

      def __init__(self, *args, **kwargs):
        del args, kwargs
        self.entered = False
        self.exited = False
        type(self).instances.append(self)

      def __enter__(self):
        self.entered = True
        return self

      def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.exited = True
        return False

    def _raise_from_parallel_call(*args, **kwargs):
      del args
      self.assertIn('executor', kwargs)
      raise ValueError('test failure')

    with mock.patch.object(
        entity_agent.futures, 'ThreadPoolExecutor', _TrackingExecutor
    ):
      with mock.patch.object(
          agent, '_parallel_call_', side_effect=_raise_from_parallel_call
      ):
        with self.assertRaisesRegex(ValueError, 'test failure'):
          agent.stateless_act(entity_lib.DEFAULT_ACTION_SPEC)

    self.assertLen(_TrackingExecutor.instances, 1)
    self.assertTrue(_TrackingExecutor.instances[0].entered)
    self.assertTrue(_TrackingExecutor.instances[0].exited)


if __name__ == '__main__':
  absltest.main()

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

"""Tests EntityAgent dispatching component instances independently of equality."""

import dataclasses
import unittest

from concordia.agents import entity_agent
from concordia.typing import entity
from concordia.typing import entity_component


@dataclasses.dataclass(unsafe_hash=True)
class _RecordingContext(entity_component.ContextComponent):
  equality_key: str = 'same'
  label: str = dataclasses.field(default='context', compare=False)
  calls: list[str] = dataclasses.field(default_factory=list, compare=False)

  def pre_act(self, action_spec: entity.ActionSpec) -> str:
    del action_spec
    self.calls.append('pre_act')
    return self.label

  def post_act(self, action_attempt: str) -> str:
    self.calls.append(f'post_act:{action_attempt}')
    return ''

  def pre_observe(self, observation: str) -> str:
    self.calls.append(f'pre_observe:{observation}')
    return ''

  def post_observe(self) -> str:
    self.calls.append('post_observe')
    return ''

  def update(self) -> None:
    self.calls.append('update')

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


@dataclasses.dataclass
class _UnhashableContext(_RecordingContext):
  pass


class _ContextAct(entity_component.ActingComponent):

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity.ActionSpec,
  ) -> str:
    del action_spec
    return '|'.join(f'{key}={value}' for key, value in context.items())

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class EntityAgentTest(unittest.TestCase):

  def test_act_with_unhashable_component(self):
    context = _UnhashableContext()
    agent = entity_agent.EntityAgent('Alice', _ContextAct(), {'one': context})
    self.assertEqual(agent.act(), 'one=context')
    self.assertEqual(
        context.calls, ['pre_act', 'post_act:one=context', 'update']
    )
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)

  def test_observe_with_unhashable_component(self):
    context = _UnhashableContext()
    agent = entity_agent.EntityAgent('Alice', _ContextAct(), {'one': context})
    agent.observe('news')
    self.assertEqual(
        context.calls, ['pre_observe:news', 'post_observe', 'update']
    )
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)

  def test_act_keeps_distinct_equal_components(self):
    first = _RecordingContext(label='first')
    second = _RecordingContext(label='second')
    self.assertEqual(first, second)
    agent = entity_agent.EntityAgent(
        'Alice', _ContextAct(), {'one': first, 'two': second}
    )
    self.assertEqual(agent.act(), 'one=first|two=second')
    expected = ['pre_act', 'post_act:one=first|two=second', 'update']
    self.assertEqual(first.calls, expected)
    self.assertEqual(second.calls, expected)

  def test_observe_keeps_distinct_equal_components(self):
    first = _RecordingContext(label='first')
    second = _RecordingContext(label='second')
    self.assertEqual(first, second)
    agent = entity_agent.EntityAgent(
        'Alice', _ContextAct(), {'one': first, 'two': second}
    )
    agent.observe('news')
    expected = ['pre_observe:news', 'post_observe', 'update']
    self.assertEqual(first.calls, expected)
    self.assertEqual(second.calls, expected)

  def test_act_calls_aliased_instance_once_and_maps_every_name(self):
    context = _RecordingContext()
    agent = entity_agent.EntityAgent(
        'Alice', _ContextAct(), {'one': context, 'two': context}
    )
    self.assertEqual(agent.act(), 'one=context|two=context')
    self.assertEqual(
        context.calls,
        ['pre_act', 'post_act:one=context|two=context', 'update'],
    )

  def test_observe_calls_aliased_instance_once(self):
    context = _RecordingContext()
    agent = entity_agent.EntityAgent(
        'Alice', _ContextAct(), {'one': context, 'two': context}
    )
    agent.observe('news')
    self.assertEqual(
        context.calls, ['pre_observe:news', 'post_observe', 'update']
    )

  def test_no_context_components(self):
    agent = entity_agent.EntityAgent('Alice', _ContextAct())
    self.assertEqual(agent.act(), '')
    agent.observe('news')
    self.assertEqual(agent.get_phase(), entity_component.Phase.READY)


if __name__ == '__main__':
  unittest.main()

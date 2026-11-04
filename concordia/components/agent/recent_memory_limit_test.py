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

"""Public-agent regression tests for recent-memory backend parity."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory
from concordia.components.agent import pubsub
from concordia.typing import entity_component
import numpy as np


class _Recent(action_spec_ignored.ActionSpecIgnored):

  def __init__(self, limit):
    super().__init__('Recent', prefix_pre_act=False)
    self._limit = limit

  def _make_pre_act_value(self):
    bank = self.get_entity().get_component('memory', type_=memory.Memory)
    return '|'.join(bank.retrieve_recent(self._limit))


class _Act(entity_component.ActingComponent):

  def get_action_attempt(self, contexts, action_spec):
    del action_spec
    return contexts['recent']

  def get_state(self):
    return {}

  def set_state(self, state):
    del state


def _memory(backend):
  texts = ['first note', 'second note', 'third note']
  if backend == 'list':
    return memory.ListMemory(texts)
  bank = basic_associative_memory.AssociativeMemoryBank(lambda _: np.ones(2))
  bank.extend(texts)
  return memory.AssociativeMemory(bank)


def _agent(bank, recent):
  return entity_agent.EntityAgent(
      'Alice', _Act(), {'memory': bank, 'recent': recent}
  )


class RecentMemoryLimitTest(parameterized.TestCase):

  @parameterized.product(backend=('list', 'associative'), limit=(0, -1))
  def test_public_agent_rejects_nonpositive_limits(self, backend, limit):
    agent = _agent(_memory(backend), _Recent(limit))
    with self.assertRaisesRegex(ValueError, 'Limit must be positive.'):
      agent.act()

  @parameterized.product(backend=('list', 'associative'), limit=(1, 2, 5))
  def test_positive_limits_keep_newest_append_order(self, backend, limit):
    agent = _agent(_memory(backend), _Recent(limit))
    expected = '|'.join(['first note', 'second note', 'third note'][-limit:])
    self.assertEqual(agent.act(), expected)

  @parameterized.product(backend=('list', 'associative'), limit=(0, -1, 2))
  def test_publisher_delegation_obeys_same_retrieval_contract(
      self, backend, limit
  ):
    publisher = pubsub.Publisher(component=_Recent(limit))
    received = []
    subscription = publisher.observable.subscribe(received.append)
    agent = _agent(_memory(backend), publisher)
    try:
      agent.act()
      self.assertTrue(publisher.wait(timeout=2.0))
      self.assertFalse(publisher.is_busy())
      if limit <= 0:
        self.assertEqual(publisher.compute_count, 0)
        self.assertEmpty(received)
      else:
        self.assertEqual(publisher.compute_count, 1)
        self.assertEqual(received, ['second note|third note'])
    finally:
      subscription.dispose()
      publisher.close()


if __name__ == '__main__':
  absltest.main()

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

"""Regression tests for observation memory watermarks."""

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import memory
from concordia.components.agent import observation
from concordia.typing import entity_component
import numpy as np


class _RecordingAct(entity_component.ActingComponent):

  def __init__(self):
    super().__init__()
    self.contexts = []

  def get_action_attempt(self, contexts, action_spec):
    del action_spec
    self.contexts.append(contexts['observations'])
    return 'waits'

  def get_state(self):
    return {}

  def set_state(self, state):
    del state


def _agent(bank, reader):
  act = _RecordingAct()
  agent = entity_agent.EntityAgent(
      'Alice',
      act,
      {
          '__memory__': bank,
          'to_memory': observation.ObservationToMemory(),
          'observations': reader,
      },
  )
  return agent, act


class ObservationWatermarkTest(absltest.TestCase):

  def test_observation_after_old_retrieval_cap_is_delivered_once(self):
    bank = ['non-observation filler'] * 1_000_000
    reader = observation.ObservationsSinceLastPreAct()
    agent, act = _agent(memory.ListMemory(bank), reader)
    agent.act()
    self.assertEqual(reader.get_state()['last_seen_index'], 1_000_000)
    agent.observe('Fresh observation beyond old cap')
    agent.act()
    self.assertIn(
        '[observation] Fresh observation beyond old cap', act.contexts[-1]
    )
    self.assertEqual(reader.get_state()['last_seen_index'], 1_000_001)
    agent.act()
    self.assertNotIn('Fresh observation beyond old cap', act.contexts[-1])

  def test_multiple_observations_keep_order_and_resume_without_replay(self):
    bank = ['non-observation filler']
    reader = observation.ObservationsSinceLastPreAct()
    agent, act = _agent(memory.ListMemory(bank), reader)
    agent.act()
    agent.observe('First fresh observation')
    agent.observe('Second fresh observation')
    agent.act()
    self.assertIn(
        '[observation] First fresh observation\n'
        '[observation] Second fresh observation\n',
        act.contexts[-1],
    )
    state = reader.get_state()
    restored = observation.ObservationsSinceLastPreAct()
    restored.set_state(state)
    resumed, resumed_act = _agent(memory.ListMemory(bank), restored)
    resumed.act()
    self.assertNotIn('fresh observation', resumed_act.contexts[-1])
    resumed.observe('Observation after restore')
    resumed.act()
    self.assertIn(
        '[observation] Observation after restore', resumed_act.contexts[-1]
    )
    self.assertNotIn('First fresh', resumed_act.contexts[-1])

  def test_associative_memory_retains_append_order_and_once_only_delivery(self):
    bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=lambda _: np.ones(2)
    )
    bank.add('non-observation filler')
    reader = observation.ObservationsSinceLastPreAct()
    agent, act = _agent(memory.AssociativeMemory(bank), reader)
    agent.act()
    agent.observe('First associative observation')
    agent.observe('Second associative observation')
    agent.act()
    self.assertIn(
        '[observation] First associative observation\n'
        '[observation] Second associative observation\n',
        act.contexts[-1],
    )
    self.assertEqual(reader.get_state()['last_seen_index'], 3)
    agent.act()
    self.assertNotIn('associative observation', act.contexts[-1])


if __name__ == '__main__':
  absltest.main()

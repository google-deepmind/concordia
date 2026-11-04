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

"""Tests that ListMemory checkpoints retain pending, uncommitted additions."""

import json
import threading
import unittest

from concordia.agents import entity_agent
from concordia.components.agent import memory as memory_component
from concordia.components.agent import pubsub
from concordia.typing import entity
from concordia.typing import entity_component


class _Act(entity_component.ActingComponent):

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity.ActionSpec,
  ) -> str:
    del context, action_spec
    return 'act'

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _agent(memory: memory_component.ListMemory) -> entity_agent.EntityAgent:
  return entity_agent.EntityAgent('Alice', _Act(), {'memory': memory})


class ListMemoryStateTest(unittest.TestCase):

  def test_ready_checkpoint_keeps_completed_background_memory(self):
    memory = memory_component.ListMemory([])
    started = threading.Event()
    release = threading.Event()

    def remember():
      started.set()
      if not release.wait(timeout=3):
        raise TimeoutError('The test did not release its background work.')
      memory.add('completed asynchronous reflection')
      return 'reflection done'

    publisher = pubsub.Publisher(function=remember)
    agent = entity_agent.EntityAgent(
        'Alice', _Act(), {'memory': memory, 'publisher': publisher}
    )
    restored_publisher = pubsub.Publisher(function=lambda: '')
    try:
      self.assertEqual(agent.act(), 'act')
      self.assertTrue(started.wait(timeout=3))
      self.assertEqual(agent.get_phase(), entity_component.Phase.READY)
      release.set()
      self.assertTrue(publisher.wait(timeout=3))
      checkpoint = json.loads(json.dumps(agent.get_state()))
      self.assertEqual(agent.get_phase(), entity_component.Phase.READY)
      memory.update()
      expected = list(memory.retrieve_recent(10))
      self.assertEqual(expected, ['completed asynchronous reflection'])

      restored = memory_component.ListMemory([])
      restored_agent = entity_agent.EntityAgent(
          'Alice',
          _Act(),
          {'memory': restored, 'publisher': restored_publisher},
      )
      restored_agent.set_state(checkpoint)
      self.assertEqual(list(restored.retrieve_recent(10)), [])
      restored.update()
      self.assertEqual(list(restored.retrieve_recent(10)), expected)
    finally:
      release.set()
      publisher.close()
      restored_publisher.close()

  def test_pending_additions_remain_pending_after_json_roundtrip(self):
    memory = memory_component.ListMemory(['committed'])
    agent = _agent(memory)
    memory.add('pending')
    checkpoint = json.loads(json.dumps(agent.get_state()))
    restored = memory_component.ListMemory(['committed'])
    _agent(restored).set_state(checkpoint)
    self.assertEqual(list(restored.retrieve_recent(10)), ['committed'])
    restored.update()
    self.assertEqual(
        list(restored.retrieve_recent(10)), ['committed', 'pending']
    )

  def test_snapshot_buffer_does_not_change_after_more_additions(self):
    memory = memory_component.ListMemory([])
    _agent(memory)
    memory.add('saved')
    checkpoint = memory.get_state()
    memory.add('later')
    restored = memory_component.ListMemory([])
    _agent(restored)
    restored.set_state(checkpoint)
    restored.update()
    self.assertEqual(list(restored.retrieve_recent(10)), ['saved'])

  def test_restore_owns_pending_buffer(self):
    checkpoint = {'memory_bank': '[]', 'buffer': ['saved']}
    memory = memory_component.ListMemory([])
    _agent(memory)
    memory.set_state(checkpoint)
    memory.add('later')
    self.assertEqual(checkpoint['buffer'], ['saved'])
    memory.update()
    self.assertEqual(list(memory.retrieve_recent(10)), ['saved', 'later'])

  def test_legacy_json_bank_clears_pending_additions(self):
    memory = memory_component.ListMemory([])
    _agent(memory)
    memory.add('not in checkpoint')
    memory.set_state({'memory_bank': json.dumps(['saved'])})
    memory.update()
    self.assertEqual(list(memory.retrieve_recent(10)), ['saved'])

  def test_legacy_list_bank_clears_pending_additions(self):
    memory = memory_component.ListMemory([])
    _agent(memory)
    memory.add('not in checkpoint')
    memory.set_state({'memory_bank': ['saved']})
    memory.update()
    self.assertEqual(list(memory.retrieve_recent(10)), ['saved'])

  def test_committed_only_legacy_checkpoint(self):
    memory = memory_component.ListMemory([])
    _agent(memory)
    memory.set_state({'memory_bank': json.dumps(['saved'])})
    self.assertEqual(list(memory.retrieve_recent(10)), ['saved'])
    memory.update()
    self.assertEqual(list(memory.retrieve_recent(10)), ['saved'])

  def test_empty_checkpoint(self):
    memory = memory_component.ListMemory([])
    _agent(memory)
    memory.set_state({})
    memory.update()
    self.assertEqual(list(memory.retrieve_recent(10)), [])


if __name__ == '__main__':
  unittest.main()

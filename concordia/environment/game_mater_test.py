# Copyright 2023 DeepMind Technologies Limited.
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

"""Test the sequence of calls made by the game master to the components."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.environment import game_master
from concordia.tests import mock_model
from concordia.typing import component
import numpy as np


def embedder(text: str):
  del text
  return np.random.rand(16)


class CallTrackingComponent(component.Component):
  """A mock component that records the sequence of calls."""

  def __init__(self):
    self.calls_sequence = []

  def name(
      self,
  ) -> str:
    return 'Mock Component'

  def state(
      self,
  ) -> str | None:
    self.calls_sequence.append('state')
    return 'Mock state'

  def partial_state(
      self,
      player_name: str,
  ) -> str | None:
    self.calls_sequence.append('partial_state ' + player_name)
    return 'Mock partial state for ' + player_name

  def observe(
      self,
      observation: str,
  ) -> None:
    self.calls_sequence.append('observe')

  def update(
      self,
  ) -> None:
    self.calls_sequence.append('update')

  def update_before_event(
      self,
      cause_statement: str,
  ) -> None:
    self.calls_sequence.append('update_before_event')

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    self.calls_sequence.append('update_after_event')

  def terminate_episode(self) -> bool:
    self.calls_sequence.append('terminate_episode')
    return False


class GameMaterTest(parameterized.TestCase):

  def test_calls_sequence(self):
    gm_call_tracker = CallTrackingComponent()

    model = mock_model.MockModel()

    importance_model = importance_function.ConstantImportanceModel()

    clock = game_clock.FixedIntervalClock()
    mem_factory = blank_memories.MemoryFactory(
        model=model,
        embedder=embedder,
        importance=importance_model.importance,
        clock_now=clock.now,
    )

    mem = mem_factory.make_blank_memory()

    alice_call_tracker = CallTrackingComponent()

    alice = basic_agent.BasicAgent(
        model,
        mem,
        'Alice',
        clock,
        [alice_call_tracker],
        verbose=False,
    )

    bob_call_tracker = CallTrackingComponent()

    bob = basic_agent.BasicAgent(
        model,
        mem,
        'Bob',
        clock,
        [bob_call_tracker],
        verbose=False,
    )

    game_master_memory = associative_memory.AssociativeMemory(
        embedder, importance_model.importance, clock=clock.now
    )

    env = game_master.GameMaster(
        model=model,
        memory=game_master_memory,
        clock=clock,
        players=[alice, bob],
        components=[gm_call_tracker],
        randomise_initiative=False,
        player_observes_event=False,
        verbose=False,
    )
    env.run_episode(1)

    expected = [
        'update',
        'partial_state Alice',
        'update_before_event',
        'state',
        'update_after_event',
        'update',
        'partial_state Bob',
        'update_before_event',
        'state',
        'update_after_event',
        'terminate_episode',
    ]
    self.assertEqual(gm_call_tracker.calls_sequence, expected)

    alice_expected_calls = [
        'update',
        'observe',
        'state',
        'state',
    ]
    self.assertEqual(alice_call_tracker.calls_sequence, alice_expected_calls)

    bob_expected_calls = [
        'update',
        'observe',
        'state',
        'state',
    ]
    self.assertEqual(bob_call_tracker.calls_sequence, bob_expected_calls)


if __name__ == '__main__':
  absltest.main()

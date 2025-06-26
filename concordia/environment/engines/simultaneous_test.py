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

"""Tests for simultaneous simulation.
"""

import functools

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.environment.engines import simultaneous
from concordia.typing import entity as entity_lib
from typing_extensions import override


_ENTITY_NAMES = ('entity_0', 'entity_1')


class MockEntity(entity_agent_with_logging.EntityAgentWithLogging):
  """Mock entity."""

  def __init__(self, name: str) -> None:
    self._name = name

  @functools.cached_property
  @override
  def name(self) -> str:
    """The name of the entity."""
    return self._name

  @override
  def observe(self, observation: str) -> None:
    pass

  @override
  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    """Always return the first entity name."""
    if action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
      return _ENTITY_NAMES[0]
    elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      return action_spec.options[0]
    else:
      raise ValueError(f'Unsupported output type: {action_spec.output_type}')


class SimultaneousTest(absltest.TestCase):

  def test_run_loop(self):
    env = simultaneous.Simultaneous()
    game_master = MockEntity(name='game_master')
    entities = [
        MockEntity(name=_ENTITY_NAMES[0]),
        MockEntity(name=_ENTITY_NAMES[1]),
    ]
    env.run_loop(
        game_masters=[game_master],
        entities=entities,
        max_steps=2,
    )


if __name__ == '__main__':
  absltest.main()

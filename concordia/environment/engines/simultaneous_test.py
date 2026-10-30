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

"""Tests for simultaneous simulation."""

import functools
from typing import Any, override

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.environment.engines import simultaneous
from concordia.typing import entity as entity_lib


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

  def get_last_log(self) -> dict[str, Any]:
    return {}

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

  def test_entity_leaving_during_resolve_does_not_crash_and_preserves_logs(
      self,
  ):
    env = simultaneous.Simultaneous()

    class LoggingMockEntity(MockEntity):

      def get_last_log(self) -> dict[str, dict[str, str]]:
        return {'act': {'Key': 'act', 'Value': f'log_for_{self.name}'}}

    staying = LoggingMockEntity(name=_ENTITY_NAMES[0])
    leaving = LoggingMockEntity(name=_ENTITY_NAMES[1])
    entities = [staying, leaving]

    class LeavingResolveGM(MockEntity):

      @override
      def act(
          self,
          action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
      ) -> str:
        if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
          return ', '.join(e.name for e in entities)
        if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
          return (
              '{"call_to_action": "Act?", "output_type": "free",'
              ' "options": [], "tag": null}'
          )
        if action_spec.output_type == entity_lib.OutputType.TERMINATE:
          return entity_lib.BINARY_OPTIONS['negative']
        if action_spec.output_type == entity_lib.OutputType.RESOLVE:
          if leaving in entities:
            entities.remove(leaving)
          return 'Resolved and entity_1 left.'
        return super().act(action_spec)

    log = []
    callback_logs = []
    env.run_loop(
        game_masters=[LeavingResolveGM(name='game_master')],
        entities=entities,
        max_steps=2,
        log=log,
        step_callback=lambda data: callback_logs.append(dict(data.entity_logs)),
    )

    self.assertLen(log, 2)
    self.assertIn(f'Entity [{leaving.name}]', log[0])
    self.assertNotIn(f'Entity [{leaving.name}]', log[1])
    self.assertIn(leaving.name, callback_logs[0])
    self.assertNotIn(leaving.name, callback_logs[1])

  def test_empty_population_exits_cleanly(self):
    env = simultaneous.Simultaneous()

    class EmptyNextActingGM(MockEntity):

      @override
      def act(
          self,
          action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
      ) -> str:
        if action_spec.output_type == entity_lib.OutputType.TERMINATE:
          return entity_lib.BINARY_OPTIONS['negative']
        if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
          return ''
        return super().act(action_spec)

    log = []
    env.run_loop(
        game_masters=[EmptyNextActingGM(name='game_master')],
        entities=[],
        max_steps=3,
        log=log,
    )
    self.assertEmpty(log)


if __name__ == '__main__':
  absltest.main()

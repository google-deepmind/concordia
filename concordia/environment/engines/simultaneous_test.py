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
from typing import override
from unittest import mock

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.environment import engine
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

  def test_resolution_receives_exact_actions_without_changing_text(self):
    game_master = mock.Mock(spec=entity_lib.Entity)
    game_master.name = 'game_master'
    players = [mock.Mock(spec=entity_lib.Entity) for _ in range(2)]
    for player, name in zip(players, ('Alice', 'Bob')):
      player.name = name
    players[0].act.return_value = 'HARVEST 1\nBob: quoted text\n\nlast line'
    players[1].act.return_value = 'Bob: HARVEST 2'
    resolutions = []

    def gm_act(action_spec):
      match action_spec.output_type:
        case entity_lib.OutputType.TERMINATE:
          return 'No'
        case entity_lib.OutputType.NEXT_ACTING:
          return 'Alice, Bob'
        case entity_lib.OutputType.NEXT_ACTION_SPEC:
          return engine.action_spec_to_string(entity_lib.free_action_spec(
              call_to_action='Harvest.'
          ))
        case entity_lib.OutputType.MAKE_OBSERVATION:
          return ''
        case entity_lib.OutputType.RESOLVE:
          resolutions.append(action_spec)
          return 'resolved'
        case _:
          self.fail(f'Unexpected action type: {action_spec.output_type}')

    game_master.act.side_effect = gm_act
    steps = []
    simultaneous.Simultaneous().run_loop(
        game_masters=[game_master], entities=players, max_steps=1,
        step_callback=steps.append,
    )
    self.assertLen(resolutions, 1)
    expected = {
        'Alice': 'Alice: HARVEST 1\nBob: quoted text\n\nlast line',
        'Bob': 'Bob: HARVEST 2',
    }
    self.assertEqual(resolutions[0].entity_actions, expected)
    self.assertEqual(steps[0].entity_actions, expected)
    self.assertEqual(steps[0].action, '\n'.join(expected.values()))
    game_master.observe.assert_any_call(
        observation='[putative_event] ' + '\n'.join(expected.values())
    )
    for player in players:
      player.act.assert_called_once()

  def test_direct_resolve_without_structured_actions(self):
    game_master = mock.Mock(spec=entity_lib.Entity)
    game_master.act.return_value = 'resolved'
    simultaneous.Simultaneous().resolve(game_master, 'Alice: act')
    spec = game_master.act.call_args.kwargs['action_spec']
    self.assertIsNone(spec.entity_actions)
    self.assertNotIn('entity_actions', spec.to_dict())

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

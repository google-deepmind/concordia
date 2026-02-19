# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for asynchronous engine."""

import functools
import json
import threading
from typing import override
from unittest import mock

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.environment.engines import asynchronous
from concordia.typing import entity as entity_lib


_ENTITY_NAMES = ('entity_0', 'entity_1')

_DEFAULT_ACTION_SPEC_JSON = json.dumps({
    'call_to_action': 'What do you do?',
    'output_type': 'free',
    'options': [],
    'tag': None,
})


class MockEntity(entity_agent_with_logging.EntityAgentWithLogging):

  def __init__(self, name: str) -> None:
    self._name = name
    self._observations = []
    self._act_count = 0

  @functools.cached_property
  @override
  def name(self) -> str:
    return self._name

  @override
  def observe(self, observation: str) -> None:
    self._observations.append(observation)

  def get_last_log(self) -> dict[str, str | dict[str, str]]:
    return {'LastNObservations': {'Summary': str(self._observations[-1:])}}

  @override
  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    self._act_count += 1
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      return _DEFAULT_ACTION_SPEC_JSON
    elif action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return entity_lib.BINARY_OPTIONS['negative']
    elif action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
      return _ENTITY_NAMES[0]
    elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      return action_spec.options[0]
    else:
      raise ValueError(f'Unsupported output type: {action_spec.output_type}')


class SelectiveGMEntity(entity_agent_with_logging.EntityAgentWithLogging):
  """A GM mock that only returns specific entities from next_acting."""

  def __init__(self, name: str, eligible_names: list[str]) -> None:
    self._name = name
    self._eligible_names = list(eligible_names)
    self._lock = threading.Lock()
    self._last_observation_target: str | None = None

  @functools.cached_property
  @override
  def name(self) -> str:
    return self._name

  def set_eligible(self, names: list[str]) -> None:
    with self._lock:
      self._eligible_names = list(names)

  @override
  def observe(self, observation: str) -> None:
    pass

  def get_last_log(self) -> dict[str, str | dict[str, str]]:
    with self._lock:
      return {
          '__make_observation__': {
              'Summary': f'Observation for {self._last_observation_target}'
          }
      }

  @override
  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      return _DEFAULT_ACTION_SPEC_JSON
    elif action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return entity_lib.BINARY_OPTIONS['negative']
    elif action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      with self._lock:
        return ', '.join(self._eligible_names)
    elif action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      with self._lock:
        self._last_observation_target = action_spec.call_to_action.split()[-1]
      return f'You see {self._last_observation_target}'
    elif action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
      return _ENTITY_NAMES[0]
    elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      return action_spec.options[0]
    else:
      raise ValueError(f'Unsupported output type: {action_spec.output_type}')


class AsynchronousTest(absltest.TestCase):

  def test_run_loop(self):
    env = asynchronous.Asynchronous()
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

  def test_next_acting_only_queries_for_passed_entities(self):
    env = asynchronous.Asynchronous()
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=list(_ENTITY_NAMES)
    )
    entity_0 = MockEntity(name=_ENTITY_NAMES[0])

    acting, specs = env.next_acting(gm, [entity_0])

    self.assertLen(acting, 1)
    self.assertEqual(acting[0].name, _ENTITY_NAMES[0])
    self.assertLen(specs, 1)

  def test_next_acting_returns_empty_when_entity_not_eligible(self):
    env = asynchronous.Asynchronous()
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=[_ENTITY_NAMES[1]]
    )
    entity_0 = MockEntity(name=_ENTITY_NAMES[0])

    acting, specs = env.next_acting(gm, [entity_0])

    self.assertEmpty(acting)
    self.assertEmpty(specs)

  def test_next_acting_returns_all_when_all_passed(self):
    env = asynchronous.Asynchronous()
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=list(_ENTITY_NAMES)
    )
    entities = [
        MockEntity(name=_ENTITY_NAMES[0]),
        MockEntity(name=_ENTITY_NAMES[1]),
    ]

    acting, specs = env.next_acting(gm, entities)

    self.assertLen(acting, 2)
    self.assertLen(specs, 2)

  def test_entity_loop_skips_when_not_eligible(self):
    env = asynchronous.Asynchronous(sleep_time=0.0)
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=[_ENTITY_NAMES[1]]
    )
    entity_0 = MockEntity(name=_ENTITY_NAMES[0])

    terminate_event = threading.Event()
    env._entity_loop(
        entity=entity_0,
        game_master=gm,
        max_steps=3,
        verbose=False,
        terminate_event=terminate_event,
    )

    self.assertEqual(entity_0._act_count, 0)

  def test_entity_loop_acts_when_eligible(self):
    env = asynchronous.Asynchronous(sleep_time=0.0)
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=list(_ENTITY_NAMES)
    )
    entity_0 = MockEntity(name=_ENTITY_NAMES[0])

    terminate_event = threading.Event()
    env._entity_loop(
        entity=entity_0,
        game_master=gm,
        max_steps=3,
        verbose=False,
        terminate_event=terminate_event,
    )

    self.assertEqual(entity_0._act_count, 3)

  def test_run_loop_banned_entity_does_not_act(self):
    env = asynchronous.Asynchronous(sleep_time=0.0)
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=[_ENTITY_NAMES[0]]
    )
    entity_0 = MockEntity(name=_ENTITY_NAMES[0])
    entity_1 = MockEntity(name=_ENTITY_NAMES[1])

    env.run_loop(
        game_masters=[gm],
        entities=[entity_0, entity_1],
        max_steps=2,
    )

    self.assertEqual(entity_0._act_count, 2)
    self.assertEqual(entity_1._act_count, 0)

  def test_run_loop_unban_entity_mid_simulation(self):
    env = asynchronous.Asynchronous(sleep_time=0.01)
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=[_ENTITY_NAMES[0]]
    )
    entity_0 = MockEntity(name=_ENTITY_NAMES[0])
    entity_1 = MockEntity(name=_ENTITY_NAMES[1])

    original_entity_loop = env._entity_loop

    def patched_entity_loop(**kwargs):
      entity = kwargs['entity']
      if entity.name == _ENTITY_NAMES[0]:
        original_entity_loop(**kwargs)
        gm.set_eligible(list(_ENTITY_NAMES))
      else:
        original_entity_loop(**kwargs)

    with mock.patch.object(
        env, '_entity_loop', side_effect=patched_entity_loop
    ):
      env.run_loop(
          game_masters=[gm],
          entities=[entity_0, entity_1],
          max_steps=3,
      )

    self.assertGreater(entity_0._act_count, 0)
    self.assertGreater(entity_1._act_count, 0)

  def test_gm_log_lock_prevents_race_condition(self):
    """Verify each thread's log entry is correctly associated with its thread."""
    env = asynchronous.Asynchronous(sleep_time=0.0)
    gm = SelectiveGMEntity(
        name='game_master', eligible_names=list(_ENTITY_NAMES)
    )
    entities = [MockEntity(name=n) for n in _ENTITY_NAMES]

    log = []
    env.run_loop(
        game_masters=[gm],
        entities=entities,
        max_steps=2,
        log=log,
    )

    self.assertNotEmpty(log)
    threads_logged = set()
    for entry in log:
      thread = entry.get('thread')
      if thread:
        threads_logged.add(thread)
        entity_key = f'Entity [{thread}]'
        self.assertIn(
            entity_key,
            entry,
            f'Log entry for thread {thread} should have key {entity_key}',
        )


if __name__ == '__main__':
  absltest.main()

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

"""Tests for parallel questionnaire engine."""

import functools
from unittest import mock

from absl.testing import absltest
from concordia.environment.engines import parallel_questionnaire
from concordia.environment.engines import questionnaire_utils
from concordia.typing import entity as entity_lib


class _ScriptedEntity(entity_lib.Entity):
  """Entity test double that returns scripted responses by output type."""

  def __init__(
      self,
      name: str,
      responses: dict[entity_lib.OutputType, str] | None = None,
      error_output_type: entity_lib.OutputType | None = None,
  ):
    self._name = name
    self._responses = responses or {}
    self._error_output_type = error_output_type
    self.observations: list[str] = []

  @functools.cached_property
  def name(self) -> str:
    return self._name

  def observe(self, observation: str) -> None:
    self.observations.append(observation)

  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    if action_spec.output_type == self._error_output_type:
      raise RuntimeError('forced error')
    if action_spec.output_type in self._responses:
      return self._responses[action_spec.output_type]
    if action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      return action_spec.options[0]
    return ''


class ParallelQuestionnaireEngineTest(absltest.TestCase):

  def test_next_acting_parses_names_and_filters_unknown(self):
    engine = parallel_questionnaire.ParallelQuestionnaireEngine()
    game_master = _ScriptedEntity(
        name='game_master',
        responses={
            entity_lib.OutputType.NEXT_ACTING: 'alice, bob , ,carol, unknown',
        },
    )
    entities = [
      _ScriptedEntity(name='alice'),
      _ScriptedEntity(name='bob'),
      _ScriptedEntity(name='carol'),
    ]

    with mock.patch.object(questionnaire_utils.logging, 'warning') as warn:
      next_entities = engine.next_acting(game_master, entities)

    self.assertEqual(
        [entity.name for entity in next_entities], ['alice', 'bob', 'carol']
    )
    warn.assert_called_once_with(
        '[%s] Ignoring unknown entity names from game master: %s',
        'ParallelQuestionnaireEngine',
        ['unknown'],
    )

  def test_run_loop_shuts_down_executor_on_early_return(self):
    engine = parallel_questionnaire.ParallelQuestionnaireEngine(max_workers=1)
    game_master = _ScriptedEntity(
        name='game_master',
        responses={
            entity_lib.OutputType.TERMINATE: entity_lib.BINARY_OPTIONS[
                'negative'
            ],
            entity_lib.OutputType.MAKE_OBSERVATION: '',
            entity_lib.OutputType.NEXT_ACTING: 'ghost',
        },
    )
    entities = [_ScriptedEntity(name='alice')]

    engine.run_loop(game_masters=[game_master], entities=entities, max_steps=1)

    self.assertIsNone(engine.get_executor())

  def test_run_loop_shuts_down_executor_on_exception(self):
    engine = parallel_questionnaire.ParallelQuestionnaireEngine(max_workers=1)
    game_master = _ScriptedEntity(
        name='game_master',
        error_output_type=entity_lib.OutputType.TERMINATE,
    )

    with self.assertRaisesRegex(RuntimeError, 'forced error'):
      engine.run_loop(game_masters=[game_master], entities=(), max_steps=1)

    self.assertIsNone(engine.get_executor())


if __name__ == '__main__':
  absltest.main()

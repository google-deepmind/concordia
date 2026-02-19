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

"""Tests for questionnaire engine helpers."""

import functools
from unittest import mock

from absl.testing import absltest
from concordia.environment.engines import questionnaire_utils
from concordia.typing import entity as entity_lib


class _NamedEntity(entity_lib.Entity):
  """Simple entity test double used for parser tests."""

  def __init__(self, name: str):
    self._name = name

  @functools.cached_property
  def name(self) -> str:
    return self._name

  def observe(self, observation: str) -> None:
    del observation

  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    del action_spec
    return ''


class QuestionnaireUtilsTest(absltest.TestCase):

  def test_parse_next_acting_entities_normalizes_tokens(self):
    entities_by_name = {
        'alice': _NamedEntity('alice'),
        'bob': _NamedEntity('bob'),
        'carol': _NamedEntity('carol'),
    }

    entities = questionnaire_utils.parse_next_acting_entities(
        'alice, bob , ,carol',
        entities_by_name,
        engine_name='SequentialQuestionnaireEngine',
    )

    self.assertEqual([entity.name for entity in entities], ['alice', 'bob', 'carol'])

  def test_parse_next_acting_entities_filters_unknown_and_logs(self):
    entities_by_name = {'alice': _NamedEntity('alice')}

    with mock.patch.object(questionnaire_utils.logging, 'warning') as warn:
      entities = questionnaire_utils.parse_next_acting_entities(
          'alice,ghost',
          entities_by_name,
          engine_name='ParallelQuestionnaireEngine',
      )

    self.assertEqual([entity.name for entity in entities], ['alice'])
    warn.assert_called_once_with(
        '[%s] Ignoring unknown entity names from game master: %s',
        'ParallelQuestionnaireEngine',
        ['ghost'],
    )

  def test_parse_next_acting_entities_preserves_order_and_duplicates(self):
    entities_by_name = {
        'alice': _NamedEntity('alice'),
        'bob': _NamedEntity('bob'),
    }

    entities = questionnaire_utils.parse_next_acting_entities(
        'bob,alice,bob',
        entities_by_name,
        engine_name='SequentialQuestionnaireEngine',
    )

    self.assertEqual([entity.name for entity in entities], ['bob', 'alice', 'bob'])


if __name__ == '__main__':
  absltest.main()

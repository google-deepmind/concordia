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

"""Tests for the interviewer game master prefab."""

from absl.testing import absltest
from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal
from concordia.prefabs.game_master import interviewer
from concordia.prefabs.simulation import questionnaire_simulation
from concordia.typing import prefab as prefab_lib
import numpy as np

_PLAYERS = ('Alice', 'Bob')


class _ChoiceQuestionnaire(base_questionnaire.QuestionnaireBase):
  """Minimal multiple-choice questionnaire for tests."""

  def aggregate_results(self, player_answers):
    return {}

  def plot_results(self, results_df, label_column=None, kwargs=None):
    pass

  def get_dimension_ranges(self):
    return {}


def _make_questionnaire():
  return _ChoiceQuestionnaire(
      name='mood',
      description='Mood survey',
      questionnaire_type='multiple_choice',
      observation_preprompt='Answer the question.',
      questions=[
          base_questionnaire.Question(
              statement='How does {player_name} feel?',
              dimension='mood',
              choices=['good', 'bad'],
          ),
          base_questionnaire.Question(
              statement='How did {player_name} sleep?',
              dimension='sleep',
              choices=['well', 'badly'],
          ),
      ],
  )


class InterviewerTest(absltest.TestCase):

  def test_questionnaire_simulation_collects_every_answer(self):
    config = prefab_lib.Config(
        default_premise='',
        prefabs={
            'minimal__Entity': minimal.Entity(),
            'interviewer__GameMaster': interviewer.GameMaster(),
        },
        instances=[
            prefab_lib.InstanceConfig(
                prefab='minimal__Entity',
                role=prefab_lib.Role.ENTITY,
                params={'name': name},
            )
            for name in _PLAYERS
        ]
        + [
            prefab_lib.InstanceConfig(
                prefab='interviewer__GameMaster',
                role=prefab_lib.Role.GAME_MASTER,
                params={
                    'name': 'interviewer',
                    'player_names': list(_PLAYERS),
                    'questionnaires': [_make_questionnaire()],
                },
            )
        ],
    )
    simulation = questionnaire_simulation.QuestionnaireSimulation(
        config=config,
        model=no_language_model.NoLanguageModel(),
        embedder=lambda _: np.ones(3),
    )

    simulation.play(max_steps=5)

    answers = (
        simulation.game_masters[0].get_component('questionnaire').get_answers()
    )
    for name in _PLAYERS:
      self.assertCountEqual(
          [answer['question'] for answer in answers[name]],
          ['How does {player_name} feel?', 'How did {player_name} sleep?'],
      )
      for answer in answers[name]:
        # Each answer matches exactly one of the offered choices.
        self.assertEqual(
            [choice['similarity'] for choice in answer['choices']].count(1),
            1,
        )


if __name__ == '__main__':
  absltest.main()

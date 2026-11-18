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
    return self._default_aggregate_results(player_answers)

  def plot_results(self, results_df, label_column=None, kwargs=None):
    pass

  def get_dimension_ranges(self):
    return {'mood': (0, 1), 'sleep': (0, 1)}


def _make_questionnaire():
  return _ChoiceQuestionnaire(
      name='mood',
      description='Mood survey',
      questionnaire_type='multiple_choice',
      observation_preprompt='Answer the question.',
      dimensions=['mood', 'sleep'],
      questions=[
          base_questionnaire.Question(
              statement='How does {player_name} feel?',
              dimension='mood',
              choices=['good', 'bad'],
              ascending_scale=False,
          ),
          base_questionnaire.Question(
              statement='How did {player_name} sleep?',
              dimension='sleep',
              choices=['badly', 'well'],
              ascending_scale=True,
          ),
      ],
  )


def _build_simulation(players=_PLAYERS):
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
          for name in players
      ]
      + [
          prefab_lib.InstanceConfig(
              prefab='interviewer__GameMaster',
              role=prefab_lib.Role.GAME_MASTER,
              params={
                  'name': 'interviewer',
                  'player_names': list(players),
                  'questionnaires': [_make_questionnaire()],
              },
          )
      ],
  )
  return questionnaire_simulation.QuestionnaireSimulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
      embedder=lambda _: np.ones(3),
  )


class InterviewerTest(absltest.TestCase):

  def test_questionnaire_simulation_collects_and_scores_every_answer(self):
    simulation = _build_simulation(_PLAYERS)

    simulation.play()

    questionnaire_comp = simulation.game_masters[0].get_component(
        'questionnaire'
    )
    answers = questionnaire_comp.get_answers()
    for name in _PLAYERS:
      self.assertIn('mood', answers[name])
      player_q_answers = answers[name]['mood']
      self.assertCountEqual(player_q_answers.keys(), ['mood_0', 'mood_1'])
      self.assertEqual(
          player_q_answers['mood_0']['statement'],
          'How does {player_name} feel?',
      )
      self.assertIn(player_q_answers['mood_0']['text'], ('good', 'bad'))
      expected_mood_val = (
          1 if player_q_answers['mood_0']['text'] == 'good' else 0
      )
      self.assertEqual(player_q_answers['mood_0']['value'], expected_mood_val)

      self.assertEqual(
          player_q_answers['mood_1']['statement'],
          'How did {player_name} sleep?',
      )
      self.assertIn(player_q_answers['mood_1']['text'], ('badly', 'well'))
      expected_sleep_val = (
          1 if player_q_answers['mood_1']['text'] == 'well' else 0
      )
      self.assertEqual(player_q_answers['mood_1']['value'], expected_sleep_val)

    results_df = questionnaire_comp.get_questionnaires_results()
    self.assertIsNotNone(results_df)
    self.assertEqual(list(results_df.index), list(_PLAYERS))
    self.assertCountEqual(list(results_df.columns), ['mood', 'sleep'])

  def test_overlapping_player_names_do_not_collide(self):
    players = ('Ann', 'Ann Marie')
    simulation = _build_simulation(players)

    simulation.play()

    answers = (
        simulation.game_masters[0].get_component('questionnaire').get_answers()
    )
    for name in players:
      self.assertCountEqual(answers[name]['mood'].keys(), ['mood_0', 'mood_1'])


if __name__ == '__main__':
  absltest.main()

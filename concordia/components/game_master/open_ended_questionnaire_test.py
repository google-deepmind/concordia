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

"""Tests for the OpenEndedQuestionnaire component."""

from absl.testing import absltest
from concordia.components.game_master import open_ended_questionnaire
from concordia.contrib.data.questionnaires import base_questionnaire
import numpy as np


class _FreeQuestionnaire(base_questionnaire.QuestionnaireBase):
  """Minimal free-text questionnaire for tests."""

  def aggregate_results(self, player_answers):
    return {}

  def plot_results(self, results_df, label_column=None, kwargs=None):
    pass

  def get_dimension_ranges(self):
    return {}


class OpenEndedQuestionnaireTest(absltest.TestCase):
  """Tests for the OpenEndedQuestionnaire component."""

  def _make_component(self, questionnaire_type, embedder):
    questionnaire = _FreeQuestionnaire(
        name='mood',
        description='Mood survey',
        questionnaire_type=questionnaire_type,
        observation_preprompt='Answer the question.',
        questions=[
            base_questionnaire.Question(
                statement='How do you feel?',
                dimension='mood',
                choices=['good', 'bad'],
            )
        ],
    )
    return open_ended_questionnaire.OpenEndedQuestionnaire(
        questionnaires=[questionnaire],
        player_names=['Alice'],
        sequence_of_events=['event'],
        embedder=embedder,
    )

  def test_free_answer_without_embedder_does_not_crash(self):
    # Answering a free-text question without an embedder used to raise
    # `TypeError: 'NoneType' object is not callable`.
    component = self._make_component('free', embedder=None)
    component.pre_observe('[putative_event] Alice: mood_0: happy')
    answer = component.get_answers()[0]['Alice']['mood']['mood_0']
    self.assertEqual(answer['text'], 'happy')
    self.assertEqual(answer['value'], 'happy')
    self.assertIsNone(answer['embedding'])

  def test_open_ended_answer_without_embedder_does_not_crash(self):
    component = self._make_component('open-ended', embedder=None)
    component.pre_observe('[putative_event] Alice: mood_0: happy')
    answer = component.get_answers()[0]['Alice']['mood']['mood_0']
    self.assertEqual(answer['value'], 'happy')

  def test_free_answer_with_embedder_computes_similarities(self):
    def embedder(text):
      # Deterministic one-hot embedding keyed on the first letter.
      index = ord(text[0].lower()) - ord('a')
      vector = np.zeros(26)
      vector[index] = 1.0
      return vector

    component = self._make_component('open-ended', embedder=embedder)
    component.pre_observe('[putative_event] Alice: mood_0: great')
    answer = component.get_answers()[0]['Alice']['mood']['mood_0']
    self.assertEqual(answer['text'], 'great')
    # 'great' and 'good' both start with 'g' so their similarity is 1.0 while
    # 'bad' starts with 'b' so its similarity is 0.0.
    self.assertEqual(answer['value'][0], {'choice': 'good', 'similarity': 1.0})
    self.assertEqual(answer['value'][1], {'choice': 'bad', 'similarity': 0.0})
    self.assertIsNotNone(answer['embedding'])


if __name__ == '__main__':
  absltest.main()

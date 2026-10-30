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

import json

from absl.testing import absltest
from concordia.components.game_master import open_ended_questionnaire
from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
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

  def _make_component(self, questionnaire_type, embedder, choices=None):
    questionnaire = _FreeQuestionnaire(
        name='mood',
        description='Mood survey',
        questionnaire_type=questionnaire_type,
        observation_preprompt='Answer the question.',
        questions=[
            base_questionnaire.Question(
                statement='How do you feel?',
                dimension='mood',
                choices=choices or ['good', 'bad'],
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

  def test_multiple_choice_answer_with_player_name_is_scored(self):
    component = self._make_component(
        'multiple_choice',
        embedder=None,
        choices=['{player_name} feels good', '{player_name} feels bad'],
    )
    # The player is offered 'Alice feels good' / 'Alice feels bad', so the
    # chosen option comes back with the name filled in.
    component.pre_observe('[putative_event] Alice: mood_0: Alice feels bad')
    answer = component.get_answers()[0]['Alice']['mood']['mood_0']
    self.assertEqual(
        answer['value'],
        [
            {'choice': '{player_name} feels good', 'similarity': 0},
            {'choice': '{player_name} feels bad', 'similarity': 1},
        ],
    )

  def test_hyphenated_multiple_choice_is_asked_as_choice(self):
    component = self._make_component('multiple-choice', embedder=None)
    specs = json.loads(
        component.pre_act(
            entity_lib.ActionSpec(
                call_to_action='Alice',
                output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
            )
        )
    )
    action_spec = engine_lib.action_spec_parser(specs[0]['action_spec_str'])
    self.assertEqual(action_spec.output_type, entity_lib.OutputType.CHOICE)
    self.assertEqual(action_spec.options, ('good', 'bad'))


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the AsynchronousQuestionnaire component."""

from absl.testing import absltest
from concordia.components.game_master import asynchronous_questionnaire
from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib


class _ChoiceQuestionnaire(base_questionnaire.QuestionnaireBase):
  """Minimal multiple-choice questionnaire for tests."""

  def aggregate_results(self, player_answers):
    return {}

  def plot_results(self, results_df, label_column=None, kwargs=None):
    pass

  def get_dimension_ranges(self):
    return {}


class AsynchronousQuestionnaireTest(absltest.TestCase):
  """Tests for the AsynchronousQuestionnaire component."""

  def _make_component(self, choices):
    questionnaire = _ChoiceQuestionnaire(
        name='mood',
        description='Mood survey',
        questionnaire_type='multiple_choice',
        observation_preprompt='Answer the question.',
        questions=[
            base_questionnaire.Question(
                statement='How does {player_name} feel?',
                dimension='mood',
                choices=choices,
            )
        ],
    )
    return asynchronous_questionnaire.AsynchronousQuestionnaire(
        questionnaires=[questionnaire],
        player_names=['Alice'],
    )

  def test_multiple_choice_answer_is_scored(self):
    component = self._make_component(['good', 'bad'])
    component.pre_observe('[putative_event] Alice: bad')
    answer = component.get_answers()['Alice'][0]
    self.assertEqual(
        answer['choices'],
        [
            {'choice': 'good', 'similarity': 0},
            {'choice': 'bad', 'similarity': 1},
        ],
    )

  def test_multiple_choice_answer_with_player_name_is_scored(self):
    component = self._make_component(
        ['{player_name} feels good', '{player_name} feels bad']
    )
    action_spec = engine_lib.action_spec_parser(
        component.pre_act(
            entity_lib.ActionSpec(
                call_to_action='What would Alice do next?',
                output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
            )
        )
    )
    # The player is offered the options with their name filled in, so the
    # chosen option comes back with the name filled in as well.
    self.assertEqual(
        action_spec.options, ('Alice feels good', 'Alice feels bad')
    )

    component.pre_observe(f'[putative_event] Alice: {action_spec.options[1]}')
    answer = component.get_answers()['Alice'][0]
    self.assertEqual(
        answer['choices'],
        [
            {'choice': '{player_name} feels good', 'similarity': 0},
            {'choice': '{player_name} feels bad', 'similarity': 1},
        ],
    )


if __name__ == '__main__':
  absltest.main()

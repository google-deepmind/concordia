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

"""Component helping a game master ask a questionnaire."""

import collections
from collections.abc import Sequence
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from concordia.components.game_master import event_resolution
from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import pandas as pd


DefaultDict = collections.defaultdict


PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_TERMINATE_SIGNAL = 'Yes'


class Questionnaire(entity_component.ContextComponent):
  """A component that asks a questionnaire to one or more actors."""

  def __init__(
      self,
      questionnaires: Sequence[base_questionnaire.QuestionnaireBase],
      player_names: Sequence[str],
      pre_act_label: str = 'Current Question',
  ):
    """Initializes the component.

    Args:
      questionnaires: A sequence of questionnaire dictionaries.
      player_names: The names of the players to which the questions are
        addressed.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._player_names = player_names
    self._questionnaires: Dict[str, base_questionnaire.QuestionnaireBase] = {
        questionnaire.name: questionnaire for questionnaire in questionnaires
    }
    self._lock = threading.Lock()

    # q_id -> (questionnaire_type, question_data)
    self._questions_by_id: Dict[
        str, Tuple[str, base_questionnaire.Question]
    ] = {}
    self._all_question_ids: List[str] = []
    self._build_question_list()

    self._answers: Dict[str, Dict[str, Any]] = {
        name: {} for name in player_names
    }
    self._answered_mask: Dict[str, Dict[str, bool]] = {
        name: {qid: False for qid in self._all_question_ids}
        for name in player_names
    }

  def _build_question_list(self):
    self._all_question_ids.clear()
    self._questions_by_id.clear()
    for qn_name, questionnaire in self._questionnaires.items():
      for qn_idx, question in enumerate(questionnaire.questions):
        q_id = f'{qn_name}_{qn_idx}'
        self._all_question_ids.append(q_id)
        self._questions_by_id[q_id] = (qn_name, question)

  def reset(self) -> None:
    """Resets the component to its initial state."""
    self._answers = {name: {} for name in self._player_names}
    self._answered_mask = {
        name: {qid: False for qid in self._all_question_ids}
        for name in self._player_names
    }

  def is_done(self, player_name: str | None = None) -> bool:
    if player_name:
      if player_name not in self._player_names:
        return True
      return all(self._answered_mask[player_name].values())
    else:
      return all(self.is_done(name) for name in self._player_names)

  def _get_action_spec_str(self, player_name: str, question_id: str) -> str:
    """Returns the ActionSpec string for a given question ID."""
    questionnaire_name, current_question = self._questions_by_id[question_id]
    questionnaire = self._questionnaires[questionnaire_name]

    if questionnaire.questionnaire_type == 'multiple_choice':
      output_type = entity_lib.OutputType.CHOICE
      options = current_question.choices
    else:
      output_type = entity_lib.OutputType.FREE
      options = ()

    prompt = (
        f'{questionnaire.observation_preprompt}\n\n'
        f'{current_question.preprompt} {current_question.statement}'
    )
    prompt = prompt.replace('{player_name}', player_name)
    prompt = prompt.replace('"', '\\"').replace('\n', ' ').strip()
    type_str = output_type.name.lower()
    options_str = ','.join(options).replace('{player_name}', player_name)
    action_str = f'prompt: "{prompt}";;type: {type_str}'
    if options_str:
      action_str += f';;options: {options_str}'
    return action_str

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return _TERMINATE_SIGNAL if self.is_done() else ''

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      return ','.join(self._player_names)

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      action_specs_list = []
      player_names = action_spec.call_to_action.split(',')
      for player in player_names:
        if player in self._player_names:
          for q_id in self._all_question_ids:
            if not self._answered_mask[player][q_id]:
              spec_str = self._get_action_spec_str(player, q_id)
              action_specs_list.append({
                  'player_name': player,
                  'question_id': q_id,
                  'action_spec_str': spec_str,
              })
      return json.dumps(action_specs_list)

    return ''

  def pre_observe(self, observation: str) -> str:
    try:
      # Escape the brackets in the tag
      tag_pattern = re.escape(PUTATIVE_EVENT_TAG)
      # Improved regex to handle colons in the answer text
      pattern = re.compile(
          rf'{tag_pattern}\s*([^:]+):\s*([^:]+):\s*(.*)', re.DOTALL
      )
      match = pattern.match(observation)
      if match:
        player_name, q_id, answer_text = match.groups()
        player_name = player_name.strip()
        q_id = q_id.strip()
        answer_text = answer_text.strip()
        if (
            player_name in self._player_names
            and q_id in self._questions_by_id
            and not self._answered_mask[player_name][q_id]
        ):
          self._process_answer(player_name, q_id, answer_text)
          self._answered_mask[player_name][q_id] = True
        else:
          print(
              f'Warning: No match or already answered for {player_name}, {q_id}'
          )
      else:
        print(f'Warning: No regex match for observation: {observation}')
    except re.error as e:
      print(f'Error processing observation: {e} on {observation}')
    return ''

  def _process_answer(
      self, player_name: str, question_id: str, answer_text: str
  ):
    questionnaire_name, current_question = self._questions_by_id[question_id]
    questionnaire = self._questionnaires[questionnaire_name]

    dimension, answer_value = questionnaire.process_answer(
        player_name, answer_text, current_question
    )

    if player_name not in self._answers:
      self._answers[player_name] = {}
    if questionnaire_name not in self._answers[player_name]:
      self._answers[player_name][questionnaire_name] = {}

    self._answers[player_name][questionnaire_name][question_id] = {
        'statement': current_question.statement,
        'text': answer_text,
        'dimension': dimension,
        'value': answer_value,
    }

  def get_answers(self) -> dict[str, dict[str, Any]]:
    """Returns the answers."""
    return self._answers

  def get_questionnaires_results(self) -> pd.DataFrame | None:
    """Aggregates questionnaires results by player and dimension.

    Returns:
      A DataFrame with players as rows and aggregated dimension scores as
      columns,
      or None if no results are found.
    """
    player_names = list(self._answers)
    all_player_results = []

    for player in player_names:
      player_results = {}
      for q_name, questionnaire in self._questionnaires.items():
        if q_name in self._answers[player]:
          player_q_answers = self._answers[player][q_name]
          aggregated = questionnaire.aggregate_results(player_q_answers)
          player_results.update(aggregated)
      all_player_results.append(player_results)

    if not all_player_results:
      print('No questionnaire results found to aggregate.')
      return None

    df = pd.DataFrame(all_player_results, index=player_names)
    return df

  def plot_all_results(
      self,
      results_df: pd.DataFrame,
      label_column: str | None = None,
      kwargs: Optional[dict[str, Any]] = None,
  ):
    """Calls the plot_results function for each questionnaire."""
    if kwargs is None:
      kwargs = {}
    for questionnaire in self._questionnaires.values():
      print(f'Plotting results for {questionnaire.name}')
      questionnaire.plot_results(
          results_df, label_column=label_column, kwargs=kwargs
      )

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'answers': self._answers,
        'answered_mask': self._answered_mask,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._answers = state['answers']  # pytype: disable=annotation-type-mismatch
    self._answered_mask = state['answered_mask']  # pytype: disable=annotation-type-mismatch

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
from collections.abc import Callable, Sequence
import copy
import json
import re
import threading
from typing import Any, Dict, List, Tuple

from concordia.components.game_master import event_resolution
from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np
import pandas as pd

DefaultDict = collections.defaultdict


PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_TERMINATE_SIGNAL = 'Yes'


class OpenEndedQuestionnaire(entity_component.ContextComponent):
  """A component that asks openended questionnaire to one or more actors.

  The component is designed to be used with the ParallelQuestionnaireEngine. The
  component translate the sequence of events into the observation stream; after
  each observation the agents are asked to respond to the questionnaires. Each
  question is asked to each player at each step. The answers are aggregated
  and plotted at the end of the game. If sequence_of_events is None, the
  component will send empty observations to the players.
  """

  def __init__(
      self,
      questionnaires: Sequence[base_questionnaire.QuestionnaireBase],
      player_names: Sequence[str],
      sequence_of_events: Sequence[str] | None = None,
      embedder: Callable[[str], np.ndarray] | None = None,
      pre_act_label: str = 'Current Question',
  ):
    """Initializes the component.

    Args:
      questionnaires: A sequence of questionnaire dictionaries.
      player_names: The names of the players to which the questions are
        addressed.
      sequence_of_events: A sequence of events to be observed by the players.
      embedder: A function that takes a string and returns an embedding.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._player_names = player_names
    self._questionnaires: Dict[str, base_questionnaire.QuestionnaireBase] = {
        questionnaire.name: questionnaire for questionnaire in questionnaires
    }
    self._embedder = embedder
    self._sequence_of_events = sequence_of_events or ['']
    self._lock = threading.Lock()
    self._event_counter = -1

    # q_id -> (questionnaire_type, question_data)
    self._questions_by_id: Dict[
        str, Tuple[str, base_questionnaire.Question]
    ] = {}
    self._all_question_ids: List[str] = []
    self._build_question_list()
    self.reset()

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
    self._answers: list[Dict[str, Dict[str, Any]]] = [
        {name: {} for name in self._player_names}
        for _ in range(len(self._sequence_of_events))
    ]
    self._answered_mask: list[Dict[str, Dict[str, bool]]] = [
        {
            name: {qid: False for qid in self._all_question_ids}
            for name in self._player_names
        }
        for _ in range(len(self._sequence_of_events))
    ]

  def is_done(self, player_name: str | None = None) -> bool:
    if self._event_counter >= len(self._sequence_of_events):
      return True
    if player_name:
      if player_name not in self._player_names:
        return True
      return all(self._answered_mask[self._event_counter][player_name].values())
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

      self._event_counter += 1
      return (
          _TERMINATE_SIGNAL
          if self._event_counter >= len(self._sequence_of_events)
          else ''
      )

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      return ','.join(self._player_names)

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      action_specs_list = []
      player_names = action_spec.call_to_action.split(',')
      for player in player_names:
        if player in self._player_names:
          for q_id in self._all_question_ids:
            if not self._answered_mask[self._event_counter][player][q_id]:
              spec_str = self._get_action_spec_str(player, q_id)
              action_specs_list.append({
                  'player_name': player,
                  'question_id': q_id,
                  'action_spec_str': spec_str,
              })
      return json.dumps(action_specs_list)
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      if self._sequence_of_events:
        return self._sequence_of_events[self._event_counter]
      else:
        return ''
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
            and not self._answered_mask[self._event_counter][player_name][q_id]
        ):
          self._process_answer(player_name, q_id, answer_text)
          self._answered_mask[self._event_counter][player_name][q_id] = True
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

    dimension = current_question.dimension

    if player_name not in self._answers[self._event_counter]:
      self._answers[self._event_counter][player_name] = {}
    if (
        questionnaire_name
        not in self._answers[self._event_counter][player_name]
    ):
      self._answers[self._event_counter][player_name][questionnaire_name] = {}

    if (
        questionnaire.questionnaire_type == 'free'
        or questionnaire.questionnaire_type == 'open-ended'
    ):
      # Embedd answer and choices, computer their cosine similarity
      answer_embedding = self._embedder(answer_text)
      choice_similarities = []
      for choice in current_question.choices:
        choice_embedding = self._embedder(choice)
        similarity = np.dot(answer_embedding, choice_embedding)
        choice_similarities.append({'choice': choice, 'similarity': similarity})

      self._answers[self._event_counter][player_name][questionnaire_name][
          question_id
      ] = {
          'statement': current_question.statement,
          'text': answer_text,
          'dimension': dimension,
          'value': choice_similarities,
          'embedding': answer_embedding,
      }
    elif (
        questionnaire.questionnaire_type == 'multiple_choice'
        or questionnaire.questionnaire_type == 'multiple-choice'
    ):
      # make value 1 for the selected choice, 0 for the rest
      choice_similarities = [
          {'choice': choice, 'similarity': 1 if choice == answer_text else 0}
          for choice in current_question.choices
      ]

      self._answers[self._event_counter][player_name][questionnaire_name][
          question_id
      ] = {
          'statement': current_question.statement,
          'text': answer_text,
          'dimension': dimension,
          'value': choice_similarities,
      }
    else:
      raise ValueError(
          f'Unsupported questionnaire type: {questionnaire.questionnaire_type}'
      )

  def get_answers(self) -> list[dict[str, dict[str, Any]]]:
    """Returns the answers."""
    return self._answers

  def get_aggregated_results(self) -> list[dict[str, Any]]:
    """Returns the aggregated results by player and dimension.

    Returns:
      A list of dictionaries, each representing a single answer/result row.
    """
    data = []
    for step_index, step_answers in enumerate(self._answers):
      for character_name, character_data in step_answers.items():
        for questionnaire_type, questionnaire_data in character_data.items():
          for _, answer_data in questionnaire_data.items():
            row = {
                'step': step_index,
                'character': character_name,
                'questionnaire': (
                    questionnaire_type
                ),  # Include questionnaire type
                'dimension': answer_data.get('dimension'),
                'question': answer_data.get('statement'),
                'answer_text': answer_data.get('text'),
                'choices': answer_data.get(
                    'value'
                ),  # This is a list of dictionaries
            }
            data.append(row)
    return data

  def get_questionnaires_results(self) -> pd.DataFrame | None:
    """Aggregates questionnaires results by player and dimension.

    Returns:
      A DataFrame with players as rows and aggregated dimension scores as
      columns,
      or None if no results are found.
    """
    data = self.get_aggregated_results()
    df = pd.DataFrame(data)

    # Flatten the choices list into separate columns if the 'choices' exists
    if 'choices' in df.columns and not df['choices'].empty:
      # Ensure all elements in the 'choices' column are lists
      valid_choices = df['choices'].apply(lambda x: isinstance(x, list))
      choices_df = df.loc[valid_choices, 'choices'].apply(
          lambda x: pd.Series({
              choice['choice']: choice['similarity']
              for choice in x
              if isinstance(choice, dict)
              and 'choice' in choice
              and 'similarity' in choice
          })
      )

      # Handle rows where 'choices' was not a list or was empty
      df = df.drop('choices', axis=1)
      df = df.join(choices_df)

    return df

  def plot_all_results(
      self, results_df: pd.DataFrame, label_column: str | None = None
  ):
    """Calls the plot_results function for each questionnaire."""
    for questionnaire in self._questionnaires.values():
      print(f'Plotting results for {questionnaire.name}')
      questionnaire.plot_results(results_df, label_column=label_column)

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'answers': copy.deepcopy(self._answers),
        'answered_mask': copy.deepcopy(self._answered_mask),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._answers = state['answers']  # pytype: disable=annotation-type-mismatch
    self._answered_mask = state['answered_mask']  # pytype: disable=annotation-type-mismatch

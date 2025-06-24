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

from collections.abc import Sequence
import re
from typing import Any, Dict

from concordia.components.game_master import event_resolution
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_TERMINATE_SIGNAL = 'Yes'


class Questionnaire(entity_component.ContextComponent):
  """A component that asks a questionnaire to an actor."""

  def __init__(
      self,
      questionnaires: Sequence[Dict[str, Any]],
      pre_act_label: str = 'Current Question',
      player_name_to_question: str | None = None,
  ):
    """Initializes the component.

    Args:
      questionnaires: A sequence of questionnaire dictionaries. Each dictionary
        defines a questionnaire and should be structured as follows:
        {
            "name": str,  # Name of the questionnaire
            "description": str,  # Description of the questionnaire
            "type": str,  # Type of questions (e.g., "multiple_choice")
            "preprompt": str,  # Text to display before each question, can
                              # include {player_name}
            "questions": List[{
                "statement": str,  # The question text
                "choices": List[str],  # The choices for the question
                "ascending_scale": bool,  # True if higher index means higher
                                        # value
            }],
        }
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      player_name_to_question: The name of the player to which the question is
        addressed.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._player_name_to_question = player_name_to_question
    self._questionnaires = questionnaires

    self._answers = {}
    self._last_observation: str | None = None
    self._questionnaire_idx = 0
    self._question_idx = -1

  def is_done(self) -> bool:
    return self._questionnaire_idx >= len(self._questionnaires)

  def get_current_question(self) -> dict[str, Any] | None:
    if self._questionnaire_idx >= len(self._questionnaires):
      return None
    if self._question_idx < 0 or self._question_idx >= len(
        self._questionnaires[self._questionnaire_idx]['questions']
    ):
      return None

    questionnaire = self._questionnaires[self._questionnaire_idx]
    question = questionnaire['questions'][self._question_idx]
    question['preprompt'] = questionnaire['preprompt']
    question['preprompt'] = question['preprompt'].replace(
        '{player_name}', '{name}'
    )
    return question

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Prepares the action for the actor based on the current questionnaire state.

    This method checks the current state of the questionnaire and generates
    the appropriate action spec or a formatted string for the actor.

    Args:
        action_spec: The action specification indicating the desired output type

    Returns:
        str:
            - If action_spec.output_type is TERMINATE:
              - Returns _TERMINATE_SIGNAL if the questionnaire is done.
              - Returns an empty string if the questionnaire is not done.
            - If action_spec.output_type is NEXT_ACTION_SPEC:
              - Returns a formatted string containing the question prompt,
                question type, and options if a question is available.
              - Returns a skip_this_step formatted string if the questionnaire
                is finished or there is an error.
            - Otherwise, returns an empty string.
    """
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      if self.is_done():
        return _TERMINATE_SIGNAL
      else:
        return ''

    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      return ''

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      return ''

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      self._question_idx += 1
      current_question = self.get_current_question()
      if current_question is None:
        # Finished current questionnaire, go to the next one.
        self._questionnaire_idx += 1
        self._question_idx = 0
        if self.is_done():
          # All questionnaires are finished
          return 'type: __SKIP_THIS_STEP__'
        current_question = self.get_current_question()
        if current_question is None:
          # Error: No question available but not done yet
          return 'type: __SKIP_THIS_STEP__'
      if (
          self._questionnaires[self._questionnaire_idx]['type']
          == 'multiple_choice'
      ):
        output_type = entity_lib.OutputType.CHOICE
        options = current_question['choices']
      else:
        output_type = entity_lib.OutputType.FREE
        options = ()
      prompt = (
          f"{current_question['preprompt']} {current_question['statement']}"
      )
      prompt = prompt.replace('"', '\\"').replace('\n', ' ').strip()
      type_str = output_type.name.lower()
      options_str = ','.join(options)
      if options_str:
        return f'prompt: {prompt};;type: {type_str};;options: {options_str}'
      else:
        return f'prompt: "{prompt}";;type: {type_str}'

    return ''

  def pre_observe(self, observation: str) -> str:
    """Stores the observation for later use in post_observe."""
    self._last_observation = observation
    return ''

  def post_observe(self) -> str:
    """Stores the answer of the entity to the current question."""
    if self._last_observation is None:
      return ''

    current_question_data = self.get_current_question()
    if not current_question_data or not self._player_name_to_question:
      return ''

    current_questionnaire = self._questionnaires[self._questionnaire_idx]
    questionnaire_name = current_questionnaire['name']

    # Use self._last_observation instead of a direct observation parameter
    observation_to_process = self._last_observation
    expected_prefix_pattern = (
        re.escape(PUTATIVE_EVENT_TAG)
        + r'\s*'
        + re.escape(self._player_name_to_question)
        + r':\s*'
    )
    match = re.match(expected_prefix_pattern, observation_to_process)
    if match:
      answer_text = observation_to_process[match.end() :].strip()
      question_statement = current_question_data['statement']

      answer_value = None
      choices = current_question_data.get('choices')
      # ascending_scale should be in the question data from JSON, default True
      ascending = current_question_data.get('ascending_scale', True)

      if choices and isinstance(choices, list):
        try:
          idx = choices.index(answer_text)
          if ascending:
            answer_value = idx
          else:
            answer_value = len(choices) - 1 - idx
        except ValueError:
          # answer_text not found in choices, answer_value remains None
          pass

      if self._player_name_to_question not in self._answers:
        self._answers[self._player_name_to_question] = {}
      if questionnaire_name not in self._answers[self._player_name_to_question]:
        self._answers[self._player_name_to_question][questionnaire_name] = {}

      self._answers[self._player_name_to_question][questionnaire_name][
          question_statement
      ] = {
          'text': answer_text,
          'value': answer_value,
      }
    return ''

  def reset(self) -> None:
    """Resets the component to its initial state."""
    self._questionnaire_idx = 0
    self._question_idx = -1
    self._answers = {}

  def get_answers(self) -> dict[str, dict[str, Any]]:
    """Returns the answers to the questionnaire."""
    return self._answers

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'questionnaire_idx': self._questionnaire_idx,
        'question_idx': self._question_idx,
        'answers': self._answers,
        'last_observation': self._last_observation,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._questionnaire_idx = state['questionnaire_idx']
    self._question_idx = state['question_idx']
    self._answers = dict(state['answers'])
    self._last_observation = None

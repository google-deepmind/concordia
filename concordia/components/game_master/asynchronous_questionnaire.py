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

"""Thread-safe component for administering questionnaires asynchronously."""

from collections.abc import Callable, Sequence
import copy
import re
import threading
from typing import Any, Dict, List, Tuple

from absl import logging
from concordia.components.game_master import event_resolution
from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np
import pandas as pd

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_TERMINATE_SIGNAL = 'Yes'


class AsynchronousQuestionnaire(entity_component.ContextComponent):
  """A thread-safe component that asks questions to actors asynchronously.

  Each player entity interacts with this component independently, typically
  within their own thread in an asynchronous engine. Questions are asked
  sequentially per player.
  """

  def __init__(
      self,
      questionnaires: Sequence[base_questionnaire.QuestionnaireBase],
      player_names: Sequence[str],
      embedder: Callable[[str], np.ndarray] | None = None,
      sequence_of_events: Sequence[str] | None = None,
      pre_act_label: str = 'Current Question',
      observe_own_answers: bool = False,
  ):
    """Initializes the component.

    Args:
      questionnaires: A sequence of questionnaire objects.
      player_names: The names of the players to which the questions are
        addressed.
      embedder: Optional embedder for open-ended questions.
      sequence_of_events: Optional sequence of events (e.g. preambles).
      pre_act_label: Label for pre_act context.
      observe_own_answers: If True, after each answer the player will observe a
        summary of the question and their response, storing it in their
        associative memory for use in subsequent questions.
    """
    super().__init__()
    self._pre_act_label = pre_act_label
    self._player_names = list(player_names)
    self._questionnaires = {q.name: q for q in questionnaires}
    self._embedder = embedder
    self._sequence_of_events = sequence_of_events or []
    self._observe_own_answers = observe_own_answers
    self._lock = threading.RLock()

    # Flatten questions for easier indexing
    self._all_questions: List[Tuple[str, base_questionnaire.Question]] = []
    for qn in questionnaires:
      for q in qn.questions:
        self._all_questions.append((qn.name, q))

    self._player_event_index: Dict[str, int] = {
        name: 0 for name in player_names
    }
    self._player_question_index: Dict[str, int] = {
        name: 0 for name in player_names
    }
    self._answers: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in player_names
    }
    self._pending_qa_observations: Dict[str, List[str]] = {
        name: [] for name in player_names
    }
    self._player_observed_event_index: Dict[str, int] = {
        name: -1 for name in player_names
    }

  def reset(self) -> None:
    """Resets the component to its initial state."""
    with self._lock:
      for name in self._player_names:
        self._player_event_index[name] = 0
        self._player_question_index[name] = 0
        self._answers[name] = []
        self._pending_qa_observations[name] = []
        self._player_observed_event_index[name] = -1

  def is_done(self, player_name: str | None = None) -> bool:
    """Returns True if the questionnaire is finished."""
    with self._lock:
      if player_name:
        if player_name not in self._player_event_index:
          return True
        threshold = max(1, len(self._sequence_of_events))
        return self._player_event_index[player_name] >= threshold
      threshold = max(1, len(self._sequence_of_events))
      done = all(
          idx >= threshold
          for idx in self._player_event_index.values()
      )
      return done

  def _get_action_spec(self, player_name: str) -> entity_lib.ActionSpec:
    """Returns the ActionSpec for the player's current question."""
    with self._lock:
      p_event_idx: Dict[str, int] = self._player_event_index  # type: ignore
      event_idx = p_event_idx.get(player_name, 0)
      threshold = max(1, len(self._sequence_of_events))
      if event_idx >= threshold:
        return entity_lib.ActionSpec(
            call_to_action='No more events.',
            output_type=entity_lib.OutputType.FREE,
        )

      p_question_idx: Dict[str, int] = self._player_question_index  # type: ignore
      q_idx = p_question_idx.get(player_name, 0)
      if q_idx >= len(self._all_questions):
        return entity_lib.ActionSpec(
            call_to_action='No more questions for this event.',
            output_type=entity_lib.OutputType.FREE,
        )

      qn_name, question = self._all_questions[q_idx]
      qn = self._questionnaires[qn_name]

      if qn.questionnaire_type in ('multiple_choice', 'multiple-choice'):
        output_type = entity_lib.OutputType.CHOICE
        options = tuple(
            opt.replace('{player_name}', player_name)
            for opt in question.choices
        )
      else:
        output_type = entity_lib.OutputType.FREE
        options = ()

      prompt = (
          f'{qn.observation_preprompt}\n\n'
          f'{question.preprompt} {question.statement}'
      )
      prompt = prompt.replace('{player_name}', player_name).strip()

      return entity_lib.ActionSpec(
          call_to_action=prompt,
          output_type=output_type,
          options=options,
          tag=f'{qn_name}:{event_idx}:{q_idx}',  # metadata for resolution
      )

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return _TERMINATE_SIGNAL if self.is_done() else ''

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      # Async engine passes candidates in options.
      # If any candidate still has questions, they can act.
      acting_players = [p for p in action_spec.options if not self.is_done(p)]
      return ','.join(acting_players)

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      # Use the name from the prompt if possible.
      player_name = ''
      for p in self._player_names:
        if re.search(rf'\b{re.escape(p)}\b', action_spec.call_to_action):
          player_name = p
          break

      if not player_name:
        return ''

      return engine_lib.action_spec_to_string(
          self._get_action_spec(player_name)
      )

    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      # Use the name from the formatted call to action.
      player_name = ''
      for p in self._player_names:
        if re.search(rf'\b{re.escape(p)}\b', action_spec.call_to_action):
          player_name = p
          break
      if not player_name:
        return ''

      with self._lock:
        p_event_idx: Dict[str, int] = self._player_event_index  # type: ignore
        event_idx = p_event_idx.get(player_name, 0)

        # Only observe the event from the sequence once per event_idx
        p_obs_idx: Dict[str, int] = self._player_observed_event_index  # type: ignore
        observed_idx = p_obs_idx.get(player_name, -1)
        if (
            event_idx < len(self._sequence_of_events)
            and event_idx > observed_idx
        ):
          observation = self._sequence_of_events[event_idx]
          self._player_observed_event_index[player_name] = event_idx
        else:
          observation = ''

        # Drain pending Q&A observations for this player.
        if self._observe_own_answers:
          pending = self._pending_qa_observations.get(player_name, [])
          if pending:
            pending_str = '\n\n\n'.join(pending)
            if observation:
              # We yield the pending answers first, then the new event
              # observation. Triple newlines (\n\n\n) signal ObservationToMemory
              # to split them into distinct, separate memory events rather than
              # squashing them.
              observation = pending_str + '\n\n\n' + observation
            else:
              observation = pending_str
            self._pending_qa_observations[player_name] = []

      return observation

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      # call_to_action contains the player's action
      return action_spec.call_to_action

    return ''

  def pre_observe(self, observation: str) -> str:
    # Resolve answers from the PUTATIVE_EVENT_TAG
    # Format: [suggested action] PlayerName: Answer
    tag_pattern = re.escape(event_resolution.PUTATIVE_EVENT_TAG)
    pattern = re.compile(rf'{tag_pattern}\s*([^:]+):\s*(.*)', re.DOTALL)
    match = pattern.match(observation)
    if match:
      player_name, answer_text = match.groups()
      player_name = player_name.strip()
      answer_text = answer_text.strip()

      with self._lock:
        if player_name in self._player_names and not self.is_done(player_name):
          event_idx = self._player_event_index[player_name]
          if not self._all_questions:
            return observation
          q_idx = self._player_question_index[player_name]
          if q_idx >= len(self._all_questions):
            return observation
          qn_name, question = self._all_questions[q_idx]
          qn = self._questionnaires[qn_name]

          # Process the answer like OpenEndedQuestionnaire does
          if qn.questionnaire_type in ('free', 'open-ended'):
            if self._embedder:
              answer_embedding = self._embedder(answer_text)
              choice_similarities = []
              for choice in question.choices:
                choice_embedding = self._embedder(choice)
                similarity = np.dot(answer_embedding, choice_embedding)
                choice_similarities.append({
                    'choice': choice,
                    'similarity': similarity,
                })
              value = choice_similarities
            else:
              value = answer_text
          elif qn.questionnaire_type in ('multiple_choice', 'multiple-choice'):
            value = [
                {
                    'choice': choice,
                    'similarity': 1 if choice == answer_text else 0,
                }
                for choice in question.choices
            ]
          else:
            value = answer_text

          self._answers[player_name].append({
              'step': event_idx,  # Event index
              'character': player_name,
              'questionnaire': qn_name,
              'dimension': question.dimension,
              'question': question.statement,
              'answer_text': answer_text,
              'choices': value,
          })

          # Queue a Q&A observation for the player's memory.
          if self._observe_own_answers:
            qa_text = (
                f'{player_name} was asked: "{question.statement}"\n'
                f'{player_name} answered: "{answer_text}"'
            )
            self._pending_qa_observations[player_name].append(qa_text)

          self._player_question_index[player_name] += 1
          if self._player_question_index[player_name] >= len(
              self._all_questions
          ):
            self._player_event_index[player_name] += 1
            self._player_question_index[player_name] = 0

    return observation

  def get_answers(self) -> Dict[str, List[Dict[str, Any]]]:
    with self._lock:
      return copy.deepcopy(self._answers)

  def get_aggregated_results(self) -> List[Dict[str, Any]]:
    """Returns results compatible with OpenEndedQuestionnaire."""
    with self._lock:
      data = []
      answers_dict: Dict[str, List[Dict[str, Any]]] = self._answers  # type: ignore
      for player_answers in list(answers_dict.values()):
        data.extend(player_answers)
    return data

  def get_questionnaires_results(self) -> pd.DataFrame | None:
    """Aggregates questionnaires results into a DataFrame."""
    data = self.get_aggregated_results()
    if not data:
      return None

    df = pd.DataFrame(data)

    # Flatten choices if present
    if 'choices' in df.columns and not df['choices'].empty:
      # Ensure all elements in the 'choices' column are lists
      valid_choices = df['choices'].apply(lambda x: isinstance(x, list))
      if valid_choices.any():
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
      logging.info('Plotting results for %s', questionnaire.name)
      questionnaire.plot_results(results_df, label_column=label_column)

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'player_event_index': dict(self._player_event_index).copy(),  # type: ignore
          'player_question_index': dict(self._player_question_index).copy(),  # type: ignore
          'player_observed_event_index': dict(self._player_observed_event_index).copy(),  # type: ignore
          'answers': copy.deepcopy(self._answers),
          'pending_qa_observations': copy.deepcopy(
              self._pending_qa_observations
          ),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      # pytype: disable=annotation-type-mismatch
      self._player_event_index = state.get(
          'player_event_index', {name: 0 for name in self._player_names}
      )
      self._player_question_index = state['player_question_index']
      self._player_observed_event_index = state.get(
          'player_observed_event_index',
          {name: -1 for name in self._player_names}
      )
      self._answers = copy.deepcopy(state['answers'])
      self._pending_qa_observations = copy.deepcopy(
          state.get(
              'pending_qa_observations',
              {name: [] for name in self._player_names},
          )
      )
      # pytype: enable=annotation-type-mismatch

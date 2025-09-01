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

"""Base class for defining questionnaires."""

import abc
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Question:
  """Structure for a single questionnaire question."""

  statement: str
  dimension: str
  preprompt: str = ""
  choices: Optional[List[str]] = None
  ascending_scale: bool = True


class QuestionnaireBase(abc.ABC):
  """Abstract base class for questionnaires."""

  def __init__(
      self,
      name: str,
      description: str,
      questionnaire_type: str,
      observation_preprompt: str,
      questions: List[Question],
      preprompt: str = "",
      dimensions: List[str] | None = None,
      context: str = "",
  ):
    self.name = name
    self.description = description
    self.questionnaire_type = questionnaire_type
    self.observation_preprompt = observation_preprompt
    self.default_preprompt = preprompt
    self.questions = questions
    self.dimensions = dimensions
    self.context = context

  def get_config(self) -> Dict[str, Any]:
    """Returns the questionnaire config dictionary."""
    return {
        "name": self.name,
        "description": self.description,
        "questionnaire_type": self.questionnaire_type,
        "preprompt": self.default_preprompt,
        "questions": [dataclasses.asdict(q) for q in self.questions],
    }

  def process_answer(
      self, player_name: str, answer_text: str, question: Question
  ) -> Tuple[str, Any]:
    """Processes a single answer text for a given question.

    Args:
      player_name: The name of the player.
      answer_text: The raw answer text.
      question: The Question object.

    Returns:
      A tuple containing (dimension, answer_value).
      The answer_value can be of any type, typically int or float for scoring.
    """
    answer_value = None
    if question.choices and isinstance(question.choices, list):
      try:
        idx = question.choices.index(
            answer_text.replace(player_name, "{player_name}")
        )
        if question.ascending_scale:
          answer_value = idx
        else:
          answer_value = len(question.choices) - 1 - idx
      except ValueError:
        pass  # Answer not in choices
    return question.dimension, answer_value

  @abc.abstractmethod
  def aggregate_results(
      self, player_answers: Dict[str, Dict[str, Any]]
  ) -> Dict[str, Any]:
    """Aggregates raw answers for a single player for this questionnaire.

    Args:
      player_answers: A dictionary of answers for a single player, structured as
        {question_id: {'statement': str, 'text': str, 'dimension': str, 'value':
        Any}}.

    Returns:
      A dict containing the aggregated results for this questionnaire for the
      player.
      Keys should be the dimension names or specific result keys (e.g.,
      'SVO_Choices').
    """
    pass

  @abc.abstractmethod
  def plot_results(
      self,
      results_df: pd.DataFrame,
      label_column: str | None = None,
      kwargs: dict[str, Any] | None = None,
  ) -> None:
    """Visualizes the aggregated results for this questionnaire.

    Args:
      results_df: DataFrame where rows are players and columns are aggregated
        dimensions, potentially including columns from other questionnaires.
        This method should only use columns relevant to this questionnaire.
      label_column: Optional column name to use for grouping/coloring plots.
      kwargs: Optional dictionary of keyword arguments to pass to the plotting
        function.
    """
    pass

  def _default_aggregate_results(
      self, player_answers: Dict[str, Dict[str, Any]]
  ) -> Dict[str, Any]:
    """Helper function for default aggregation (mean of numeric values per dimension)."""
    dimension_values: Dict[str, List[Any]] = {}
    for _, question_data in player_answers.items():
      dimension = question_data["dimension"]
      value = question_data["value"]
      if isinstance(value, (int, float)):
        if dimension not in dimension_values:
          dimension_values[dimension] = []
        dimension_values[dimension].append(value)

    aggregated = {}
    for dim, values in dimension_values.items():
      aggregated[dim] = np.mean(values) if values else np.nan
    return aggregated

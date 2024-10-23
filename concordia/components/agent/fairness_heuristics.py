# Copyright 2023 DeepMind Technologies Limited.
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

"""Fairness Heuristics component."""

from collections.abc import Mapping
from typing import Any

from concordia.typing import entity_component
from concordia.typing import logging


class FairnessHeuristics(entity_component.ContextComponent):
  """A component embedding fairness heuristics in the decision-making process."""

  def __init__(
      self,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the FairnessHeuristics component.

    Args:
      logging_channel: The channel to use for debug logging.
    """
    self._logging_channel = logging_channel
    self._equity = 0.0
    self._reciprocity = 0.0
    self._altruism = 0.0

  def update_after_event(self, event: str) -> None:
    """Updates the component state after an event.

    Args:
      event: The event to update the state with.
    """
    if "equity" in event:
      self._equity += 1
    if "reciprocity" in event:
      self._reciprocity += 1
    if "altruism" in event:
      self._altruism += 1

  def get_fairness_score(self) -> float:
    """Calculates the fairness score based on equity, reciprocity, and altruism.

    Returns:
      The fairness score.
    """
    return (self._equity + self._reciprocity + self._altruism) / 3

  def get_last_log(self) -> Mapping[str, Any]:
    """Returns the last log of the component."""
    return {
        "equity": self._equity,
        "reciprocity": self._reciprocity,
        "altruism": self._altruism,
    }

  def state(self) -> str:
    """Returns the state of the component."""
    return (
        f"Equity: {self._equity}, "
        f"Reciprocity: {self._reciprocity}, "
        f"Altruism: {self._altruism}"
    )

# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE--2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tit-for-Tat with Forgiveness strategy component."""

from collections.abc import Mapping
import random
from typing import Any

from concordia.typing import entity_component
from concordia.typing import logging


class TitForTatWithForgiveness(entity_component.ContextComponent):
  """A component implementing Tit-for-Tat with Forgiveness strategy."""

  def __init__(
      self,
      forgiveness_probability: float = 0.1,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the Tit-for-Tat with Forgiveness component.

    Args:
      forgiveness_probability: The probability of forgiving a defection.
      logging_channel: The channel to use for debug logging.
    """
    self._forgiveness_probability = forgiveness_probability
    self._logging_channel = logging_channel
    self._last_action = "cooperate"
    self._opponent_last_action = "cooperate"

  def update_after_event(self, event: str) -> None:
    """Updates the component state after an event.

    Args:
      event: The event to update the state with.
    """
    if "defect" in event:
      self._opponent_last_action = "defect"
    else:
      self._opponent_last_action = "cooperate"

  def get_action(self) -> str:
    """Determines the next action based on the Tit-for-Tat with Forgiveness strategy.

    Returns:
      The next action, either "cooperate" or "defect".
    """
    if self._opponent_last_action == "defect":
      if random.random() < self._forgiveness_probability:
        action = "cooperate"
      else:
        action = "defect"
    else:
      action = "cooperate"

    self._last_action = action
    return action

  def get_last_log(self) -> Mapping[str, Any]:
    """Returns the last log of the component."""
    return {
        "last_action": self._last_action,
        "opponent_last_action": self._opponent_last_action,
    }

  def state(self) -> str:
    """Returns the state of the component."""
    return (
        f"Last action: {self._last_action}, "
        f"Opponent last action: {self._opponent_last_action}"
    )

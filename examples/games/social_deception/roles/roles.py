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

"""Base role templates for Social Deception."""

import abc
from collections.abc import Sequence

from examples.games.social_deception import game_tracker


class Role(abc.ABC):
  """Base class for all Social Deception roles."""

  def __init__(
      self,
      name: str,
      alignment: game_tracker.Alignment,
      night_priority: int = 0,
  ):
    self.role_name = name
    self.alignment = alignment
    self.night_priority = night_priority
    self.player_name: str | None = None

  @abc.abstractmethod
  def overhead_introduction(self) -> str:
    """Returns a short overhead summary of the role's ability."""

  @abc.abstractmethod
  def player_introduction(self) -> str:
    """Returns the player introduction for this role.

    This is all the information the player gets about their role at the start of
    the game.

    Returns:
      The player introduction for this role.
    """

  @abc.abstractmethod
  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    """Processes the agent's night choice and returns the result (info).

    Args:
      action: The action taken by the agent.
      tracker_ref: The GameTracker reference.
      day_num: The day number.
      players: The players in the game.

    Returns:
      The result (info) of the night action.
    """

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    """Returns the information the player gets if they are impaired (drunk or poisoned).

    Args:
      tracker_ref: The GameTracker reference.

    Returns:
      The information the player gets if they are impaired.
    """
    del tracker_ref
    return ""


class Townsfolk(Role):
  """Base class for all Townsfolk roles (Good)."""

  def __init__(
      self,
      name: str,
      night_priority: int = 0,
  ):
    super().__init__(
        name,
        game_tracker.Alignment.GOOD,
        night_priority,
    )


class Outsider(Role):
  """Base class for all Outsider roles (Good with handicap)."""

  def __init__(
      self,
      name: str,
      night_priority: int = 0,
  ):
    super().__init__(
        name,
        game_tracker.Alignment.GOOD,
        night_priority,
    )


class Minion(Role):
  """Base class for all Minion roles (Evil support)."""

  def __init__(
      self,
      name: str,
      night_priority: int = 0,
  ):
    super().__init__(
        name,
        game_tracker.Alignment.EVIL,
        night_priority,
    )


class Demon(Role):
  """Base class for all Demon roles (Evil leader)."""

  def __init__(
      self,
      name: str,
      night_priority: int = 0,
  ):
    super().__init__(
        name,
        game_tracker.Alignment.EVIL,
        night_priority,
    )

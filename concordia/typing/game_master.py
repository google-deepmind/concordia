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


"""The abstract class that defines simulacrum game master interface.

This is an environment side simulacrum. It is responsible for providing the
observations for players and providing the outcomes for actions. It also
manages the simulated world dynamics (if there are any).
Reference: Generative Agents: Interactive Simulacra of Human Behavior
https://arxiv.org/abs/2304.03442
"""

import abc
from collections.abc import Sequence


class GameMaster(metaclass=abc.ABCMeta):
  """A game master class."""

  @abc.abstractmethod
  def name(
      self,
  ) -> str:
    """Returns the name of the game."""
    raise NotImplementedError

  @abc.abstractmethod
  def update_from_player(
      self,
      player_name: str,
      action_attempt: str,
  ) -> str:
    """Returns the outcome of the action attempt.

    Args:
      player_name: the name of the player performing the action
      action_attempt: a description of an action that the player is trying to
        perform. It can succeed or fail.

    Returns:
      the outcome of the action_attempt.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def view_for_player(
      self,
      player_name: str,
  ) -> str:
    """Returns the view of the game state for a specific player.

    Args:
      player_name: the name of the player to generate a view for

    Returns:
      the view of the game state for the player.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def run_episode(self, max_steps: int) -> Sequence[str]:
    """Runs a single episode until the end.

    Args:
      max_steps: the maximum number of steps

    Returns:
      a list of events that happened
    """

    raise NotImplementedError

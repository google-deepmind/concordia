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


"""Base class for generative agent (and game master) components."""

import abc
from typing import Sequence


class Component(metaclass=abc.ABCMeta):
  """A building block of a generative agent / game master.

  A concept constructed from memory or observations stream or (game master)
  event statements. Components mediate memory and observations into the
  context of action. In general, each component is updated by querying for
  relevant memories and then summarising the result.
  """

  @abc.abstractmethod
  def name(
      self,
  ) -> str:
    """Returns the name of the component."""
    raise NotImplementedError

  def state(
      self,
  ) -> str | None:
    """Returns the current state of the component.

    Returns:
      state of the component or None. If none is returned, then the component
      will be omitted while forming the context of action.
    """
    pass

  def partial_state(
      self,
      player_name: str,
  ) -> str | None:
    """Returns the specified player's view of the component's current state.

    Args:
      player_name: the name of the player for which the view is generated.

    Returns:
      specified player's view of the component's current state or None. If none
      is returned, then the component will not be sent to the player.
    """

    del player_name
    return None

  def observe(
      self,
      observation: str,
  ) -> None:
    """Observe data."""
    del observation
    return None

  def update(
      self,
  ) -> None:
    """Updates the component from memory.

    Returns:
      The updated state of the component.
    """
    pass

  def update_before_event(
      self,
      cause_statement: str,
  ) -> None:
    """Updates the component player`s action attempt.

    Args:
      cause_statement: The cause statement to update the component before event.

    Returns:
      New state of the component or None.
    """
    del cause_statement
    return None

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    """Updates the component from the event statement and document.

    Args:
      event_statement: The event statement to update the component from.

    Returns:
      The summary of the update or None.
    """
    del event_statement
    return None

  def terminate_episode(self) -> bool:
    return False

  def get_last_log(
      self,
  ):
    """Returns a dictionary with latest log of activity."""
    return None

  def get_components(
      self,
  ) -> Sequence['Component']:
    """Returns a list of components or an empty list."""
    return []

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

"""Base classes for Entity components."""

import abc
from collections.abc import Mapping
from concordia.typing import entity as entity_lib


ComponentName = str
ComponentContext = str
ComponentContextMapping = Mapping[ComponentName, ComponentContext]


class BaseComponent:
  """A base class for components."""

  def __init__(self):
    self._entity = None

  def set_entity(self, entity: entity_lib.Entity) -> None:
    """Sets the entity that this component belongs to."""
    self._entity = entity

  def get_entity(self) -> entity_lib.Entity:
    """Returns the entity that this component belongs to.

    Raises:
      ValueError: If the entity is not set.
    """
    if self._entity is None:
      raise ValueError("Entity is not set.")
    return self._entity


class EntityComponent(BaseComponent):
  """A building block of an entity.

  Components are stand-alone pieces of functionality insterted into a GameObject
  that have hooks for processing events for acting and observing.
  """

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Returns the relevant information for the entity to act.

    This function will be called by the entity to gather all the relevant
    information from its components before the entity acts.

    Args:
      action_spec: The action spec for the action attempt.

    Returns:
      The relevant information for the entity to act.
    """
    del action_spec
    return ""

  def post_act(
      self,
      action_attempt: str,
  ) -> str:
    """Informs the components of the action attempted.

    This function will be called by the entity to inform the components of the
    action attempted.

    Args:
      action_attempt: The action that the entity attempted.

    Returns:
      Any information that the component needs to bubble up to the entity.
    """
    del action_attempt
    return ""

  def pre_observe(
      self,
      observation: str,
  ) -> str:
    """Returns relevant information from the component to process observations.

    This function will be called by the entity to inform the components of the
    observation. The component should return any relevant information that the
    entity needs to process the observation.

    Args:
      observation: The observation that the entity received.

    Returns:
      The relevant information for the object to process the observation.
    """
    del observation
    return ""

  def post_observe(
      self,
  ) -> str:
    """Returns relevant information after processing an observation.

    This function will be called after all components have been informed of the
    observation. The component should return any relevant information that the
    entity needs to finalize the observation.

    Returns:
      Any information that the component needs to bubble up to the entity.
    """
    return ""

  def update(
      self,
  ) -> None:
    """Updates the component.

    This function will be called by the entity after all components have
    received a `post_act` or `post_observe` call. This is an opportunity for the
    component to update its internal state, and replace any cached information.
    """

  def get_last_log(
      self,
  ):
    """Returns a dictionary with latest log of activity."""
    return None


class ActingComponent(BaseComponent, metaclass=abc.ABCMeta):
  """A privileged component that decides what action to take."""

  @abc.abstractmethod
  def get_action_attempt(
      self,
      context: ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Decides the action of an entity.

    This function will be called by the entity with the context obtained from
    `pre_act` from all its components to decide the action of the entity.

    Args:
      context: The context for the action attempt. This is a mapping of
        component name to the information that the component returned in
        `pre_act`.
      action_spec: The action spec for the action attempt.

    Returns:
      The action that the entity is attempting.
    """
    raise NotImplementedError()

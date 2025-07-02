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
from collections.abc import Collection, Mapping
import enum
import functools
from typing import TypeVar

from concordia.typing import entity as entity_lib
from concordia.typing.deprecated import logging as logging_lib


ComponentName = str
ComponentContext = str
ComponentContextMapping = Mapping[ComponentName, ComponentContext]
ComponentT = TypeVar("ComponentT", bound="BaseComponent")

_LeafT = str | int | float | None
_ValueT = _LeafT | Collection["_ValueT"] | Mapping[str, "_ValueT"]
ComponentState = Mapping[str, _ValueT]

EntityState = Mapping[str, ComponentState | Mapping[str, ComponentState]]


class Phase(enum.Enum):
  """Phases of a component entity lifecycle.

  Attributes:
    READY: The agent is ready to `observe` or `act`. This will be followed by
      `PRE_ACT` or `PRE_OBSERVE`.
    PRE_ACT: The agent has received a request to act. Components are being
      requested for their action context. This will be followed by `POST_ACT`.
    POST_ACT: The agent has just submitted an action attempt. Components are
      being informed of the action attempt. This will be followed by `UPDATE`.
    PRE_OBSERVE: The agent has received an observation. Components are being
      informed of the observation. This will be followed by `POST_OBSERVE`.
    POST_OBSERVE: The agent has just observed. Components are given a chance to
      provide context after processing the observation. This will be followed by
      `UPDATE`.
    UPDATE: The agent is updating its internal state. This will be followed by
      `READY`.
  """

  READY = enum.auto()
  PRE_ACT = enum.auto()
  POST_ACT = enum.auto()
  PRE_OBSERVE = enum.auto()
  POST_OBSERVE = enum.auto()
  UPDATE = enum.auto()

  @functools.cached_property
  def successors(self) -> Collection["Phase"]:
    """Returns the phases which may follow the current phase."""
    match self:
      case Phase.READY:
        return (Phase.PRE_ACT, Phase.PRE_OBSERVE)
      case Phase.PRE_ACT:
        return (Phase.POST_ACT,)
      case Phase.POST_ACT:
        return (Phase.UPDATE,)
      case Phase.PRE_OBSERVE:
        return (Phase.POST_OBSERVE,)
      case Phase.POST_OBSERVE:
        return (Phase.UPDATE,)
      case Phase.UPDATE:
        return (Phase.READY,)
      case _:
        raise NotImplementedError()

  def check_successor(self, successor: "Phase") -> None:
    """Raises ValueError if successor is not a valid next phase."""
    if successor not in self.successors:
      raise ValueError(f"The transition from {self} to {successor} is invalid.")


class BaseComponent(metaclass=abc.ABCMeta):
  """A base class for components."""

  _entity: "EntityWithComponents | None" = None

  def set_entity(self, entity: "EntityWithComponents") -> None:
    """Sets the entity that this component belongs to.

    Args:
      entity: The entity that this component belongs to.

    Raises:
      RuntimeError: If the entity is already set.
    """
    if self._entity is not None and self._entity != entity:
      raise RuntimeError("Entity is already set.")
    self._entity = entity

  def get_entity(self) -> "EntityWithComponents":
    """Returns the entity that this component belongs to.

    Raises:
      RuntimeError: If the entity is not set.
    """
    if self._entity is None:
      raise RuntimeError("Entity is not set.")
    return self._entity

  @abc.abstractmethod
  def get_state(self) -> ComponentState:
    """Returns the state of the component.
    
    See `set_state` for details. The default implementation returns an empty
    dictionary.
    """
    return {}

  @abc.abstractmethod
  def set_state(self, state: ComponentState) -> None:
    """Sets the state of the component.
    
    This is used to restore the state of the component. The state is assumed to
    be the one returned by `get_state`.
    The state does not need to contain any information that is passed in the 
    initialization of the component (e.g. the memory bank, names of other 
    components etc.)
    It is assumed that set_state is called on the component after it was 
    initialized with the same parameters as the one used to restore it.
    The default implementation does nothing, which implies that the component
    does not have any state.

    Example (Creating a copy):
      obj1 = Component(**kwargs)
      state = obj.get_state()
      obj2 = Component(**kwargs)
      obj2.set_state(state)
      # obj1 and obj2 will behave identically.

    Example (Restoring previous behavior):
      obj = Component(**kwargs)
      state = obj.get_state()
      # do more with obj
      obj.set_state(state)
      # obj will now behave the same as it did before.
    
    Note that the state does not need to contain any information that is passed
    in __init__ (e.g. the memory bank, names of other components etc.)

    Args:
      state: The state of the component.
    """
    del state
    return None


class ComponentWithLogging(BaseComponent):
  """A base class for components with logging."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._logging_channel = logging_lib.NoOpLoggingChannel

  def set_logging_channel(self, logging_channel: logging_lib.LoggingChannel):
    """Sets the logging channel for the component."""
    self._logging_channel = logging_channel


class EntityWithComponents(entity_lib.Entity):
  """An entity that contains components."""

  @abc.abstractmethod
  def get_phase(self) -> Phase:
    """Returns the current phase of the component entity."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_component(
      self,
      name: str,
      *,
      type_: type[ComponentT] = BaseComponent,
  ) -> ComponentT:
    """Returns the component with the given name.

    Args:
      name: The name of the component to fetch.
      type_: If passed, the returned component will be cast to this type.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_state(self) -> EntityState:
    """Returns the state of the entity."""
    raise NotImplementedError()

  @abc.abstractmethod
  def set_state(self, state: EntityState) -> None:
    """Sets the state of the entity."""
    raise NotImplementedError()


class ContextComponent(BaseComponent):
  """A building block of a EntityWithComponents.

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


class ContextProcessorComponent(BaseComponent, metaclass=abc.ABCMeta):
  """A component that processes context from EntityWithComponents."""

  def pre_act(self, contexts: ComponentContextMapping) -> None:
    """Processes the pre_act contexts returned by the EntityWithComponents.

    Args:
      contexts: A mapping from ComponentName to ComponentContext.
    """
    del contexts

  def post_act(self, contexts: ComponentContextMapping) -> None:
    """Processes the post_act contexts returned by the EntityWithComponents.

    Args:
      contexts: A mapping from ComponentName to ComponentContext.
    """
    del contexts

  def pre_observe(self, contexts: ComponentContextMapping) -> None:
    """Processes the pre_observe contexts returned by the EntityWithComponents.

    Args:
      contexts: A mapping from ComponentName to ComponentContext.
    """
    del contexts

  def post_observe(self, contexts: ComponentContextMapping) -> None:
    """Processes the post_observe contexts returned by the EntityWithComponents.

    Args:
      contexts: The context from other components.
    """
    del contexts

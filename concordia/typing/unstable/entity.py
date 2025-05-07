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

"""The abstract class that defines an Entity interface."""

import abc
from collections.abc import Sequence
import dataclasses
import enum
import functools
from typing import Any


@enum.unique
class OutputType(enum.Enum):
  """The type of output that a entity can produce."""
  # Common output types
  FREE = enum.auto()
  CHOICE = enum.auto()
  FLOAT = enum.auto()
  # Game master output types
  MAKE_OBSERVATION = enum.auto()
  NEXT_ACTING = enum.auto()
  NEXT_ACTION_SPEC = enum.auto()
  RESOLVE = enum.auto()
  TERMINATE = enum.auto()
  NEXT_GAME_MASTER = enum.auto()
  SKIP_THIS_STEP = enum.auto()

PLAYER_ACTION_TYPES = (
    OutputType.FREE,
    OutputType.CHOICE,
    OutputType.FLOAT,
)
GAME_MASTER_ACTION_TYPES = (
    OutputType.MAKE_OBSERVATION,
    OutputType.NEXT_ACTING,
    OutputType.NEXT_ACTION_SPEC,
    OutputType.RESOLVE,
    OutputType.TERMINATE,
    OutputType.NEXT_GAME_MASTER,
    OutputType.SKIP_THIS_STEP,
)
FREE_ACTION_TYPES = (
    OutputType.FREE,
    OutputType.MAKE_OBSERVATION,
    OutputType.NEXT_ACTION_SPEC,
    OutputType.RESOLVE,
)
CHOICE_ACTION_TYPES = (
    OutputType.CHOICE,
    OutputType.NEXT_ACTING,
    OutputType.TERMINATE,
    OutputType.NEXT_GAME_MASTER,
)

BINARY_OPTIONS = {'affirmative': 'Yes', 'negative': 'No'}


@dataclasses.dataclass(frozen=True, kw_only=True)
class ActionSpec:
  """A specification of the action that entity is queried for.

  Attributes:
    call_to_action: formatted text conditioning entity response.
      {name} and {timedelta} will be inserted by the entity.
    output_type: type of output - FREE, CHOICE or FLOAT
    options: if multiple choice, then provide possible answers here
    tag: a tag to add to the activity memory (e.g. action, speech, etc.)
  """

  call_to_action: str
  output_type: OutputType
  options: Sequence[str] = ()
  tag: str | None = None

  def __post_init__(self):
    if self.output_type in CHOICE_ACTION_TYPES:
      if not self.options:
        raise ValueError('Options must be provided for CHOICE output type.')
      if len(set(self.options)) != len(self.options):
        raise ValueError('Options must not contain duplicate choices.')
    elif self.options:
      raise ValueError('Options not supported for non-CHOICE output type.')
    object.__setattr__(self, 'options', tuple(self.options))

  def validate(self, action: str) -> None:
    """Validates the specified action against the action spec.

    Args:
      action: The action to validate.

    Raises:
      ValueError: If the action is invalid.
    """
    if self.output_type == OutputType.FREE:
      return
    elif self.output_type == OutputType.CHOICE:
      if action not in self.options:
        raise ValueError(f'Action {action!r} is not one of {self.options!r}.')
    elif self.output_type == OutputType.FLOAT:
      try:
        float(action)
      except ValueError:
        raise ValueError(f'Action {action!r} is not a valid float.') from None
    else:
      raise NotImplementedError(f'Unsupported output type: {self.output_type}')


def free_action_spec(**kwargs) -> ActionSpec:
  """Returns an action spec with output type FREE."""
  return ActionSpec(output_type=OutputType.FREE, **kwargs)


def float_action_spec(**kwargs) -> ActionSpec:
  """Returns an action spec with output type FLOAT."""
  return ActionSpec(output_type=OutputType.FLOAT, **kwargs)


def choice_action_spec(**kwargs) -> ActionSpec:
  """Returns an action spec with output type CHOICE."""
  return ActionSpec(output_type=OutputType.CHOICE, **kwargs)


def skip_this_step_action_spec(**kwargs) -> ActionSpec:
  """Returns an action spec with output type SKIP_THIS_STEP."""
  return ActionSpec(
      output_type=OutputType.SKIP_THIS_STEP, call_to_action='', **kwargs
  )


DEFAULT_CALL_TO_ACTION = (
    'What would {name} do next?'
    ' Give a specific activity.'
    ' If the selected action has a direct or indirect object then it'
    ' must be specified explicitly.'
)

DEFAULT_ACTION_SPEC = free_action_spec(
    call_to_action=DEFAULT_CALL_TO_ACTION,
    tag='action',
)


DEFAULT_CALL_TO_SPEECH = (
    'Given the above, what is {name} likely to say next? Respond in'
    ' the format `{name} -- "..."` For example, '
    'Cristina -- "Hello! Mighty fine weather today, right?", '
    'Ichabod -- "I wonder if the alfalfa is ready to harvest", or '
    'Townsfolk -- "Good morning".\n'
)

DEFAULT_SPEECH_ACTION_SPEC = free_action_spec(
    call_to_action=DEFAULT_CALL_TO_SPEECH,
    tag='speech',
)


class Entity(metaclass=abc.ABCMeta):
  """Base class for entities.

  Entities are the basic building blocks of a game. They are the entities
  that the game master explicitly keeps track of. Entities can be anything,
  from the player's character to an inanimate object. At its core, an entity
  is an entity that has a name, can act, and can observe.

  Entities are sent observations by the game master, and they can be asked to
  act by the game master. Multiple observations can be sent to an entity before
  a request for an action attempt is made. The entities are responsible for
  keeping track of their own state, which might change upon receiving
  observations or acting.
  """

  @functools.cached_property
  @abc.abstractmethod
  def name(self) -> str:
    """The name of the entity."""
    raise NotImplementedError()

  @abc.abstractmethod
  def act(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str:
    """Returns the entity's intended action given the action spec.

    Args:
      action_spec: The specification of the action that the entity is queried
        for. This might be a free-form action, a multiple choice action, or
        a float action. The action will always be a string, but it should be
        compliant with the specification.

    Returns:
      The entity's intended action.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def observe(self, observation: str) -> None:
    """Informs the Entity of an observation.

    Args:
      observation: The observation for the entity to process. Always a string.
    """
    raise NotImplementedError()


class EntityWithLogging(Entity):
  """An agent interface for taking actions."""

  @abc.abstractmethod
  def get_last_log(self) -> dict[str, Any]:
    """Returns debugging information in the form of a dictionary."""
    raise NotImplementedError

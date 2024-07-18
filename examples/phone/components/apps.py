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


"""Classes for implementing virtual apps simulation."""

import abc
from collections.abc import Sequence
import dataclasses
import datetime
import inspect
import re
import textwrap
import typing
from typing import Any

import docstring_parser  # pytype: disable=import-error  # Fails on GitHub.
import termcolor

_DATE_FORMAT = '%Y-%m-%d %H:%M'

_ARGUMENT_REGEX = re.compile(r'(?P<param>\w+):\s*(?P<value>[^\n]+)')

_ARGUMENT_PARSERS = {
    'datetime.datetime': lambda date: datetime.datetime.strptime(
        date, _DATE_FORMAT
    ),
    'str': str,
    'int': int,
}

_ACTION_PROPERTY = '__app_action__'


def app_action(method):
  """A decorator that marks PhoneApp methods as callable actions."""
  method.__app_action__ = True
  return method


class ActionArgumentError(Exception):
  """An error that is raised when argument parsing fails."""


@dataclasses.dataclass(frozen=True)
class Parameter:
  """Represents a parameter that can be passed to an action."""

  name: str
  kind: type[Any]
  description: str | None

  def full_description(self):
    return f"{self.name}: {self.description or ''}, type: {self.kind}"

  def value_from_text(self, text: str):
    origin = typing.get_origin(self.kind)
    if not origin:
      return self._parse_single_argument(text)
    else:
      if origin != list:
        raise RuntimeError(f'Unsupported argument type {origin}')
      return self._parse_list_argument(text)

  def _parse_single_argument(self, text):
    parser = _ARGUMENT_PARSERS.get(self.kind, self.kind)
    return parser(text)

  def _parse_list_argument(self, text: str):
    arg = typing.get_args(self.kind)
    parser = _ARGUMENT_PARSERS.get(arg, arg)
    return [parser(e) for e in text.split(',')]

  @classmethod
  def create(
      cls, parameter: inspect.Parameter, docstring: docstring_parser.Docstring
  ):
    """Create a Parameter from a method docstring and inspect.Parameter."""
    description = next(
        (
            p.description
            for p in docstring.params
            if p.arg_name == parameter.name
        ),
        None,
    )
    return cls(parameter.name, parameter.annotation, description)


@dataclasses.dataclass(frozen=True)
class ActionDescriptor:
  """Represents an action that can be invoked on a PhoneApp."""

  name: str
  description: str
  parameters: Sequence[Parameter]
  docstring: dataclasses.InitVar[docstring_parser.Docstring]

  def __post_init__(self, docstring: docstring_parser.Docstring):
    pass

  def instructions(self):
    return (
        f'The {self.name} action expects the following parameters:\n'
        + '\n'.join(p.full_description() for p in self.parameters)
        + textwrap.dedent("""
    All parameters must be provided, each in its own line, for example:
    param1: value1
    param2: value2
    """)
    )

  @classmethod
  def from_method(cls, method):
    doc = docstring_parser.parse(method.__doc__)
    description = f"{doc.short_description}\n{doc.long_description or ''}"
    parameters = inspect.signature(method).parameters.items()
    method_parameters = [
        Parameter.create(p, doc) for name, p in parameters if name != 'self'
    ]
    return cls(
        name=method.__name__,
        description=description,
        parameters=method_parameters,
        docstring=doc,
    )


class PhoneApp(metaclass=abc.ABCMeta):
  """Base class for apps that concordia can interact with using plain English.

  Extend this class and decorated any method that should be callable from the
  simulation with @app_action.
  """

  _log_color = 'blue'

  @abc.abstractmethod
  def name(self) -> str:
    """Returns the name of the app."""
    raise NotImplementedError

  @abc.abstractmethod
  def description(self) -> str:
    """Returns a description of the app."""
    raise NotImplementedError

  def _print(self, entry, color=None):
    print(termcolor.colored(entry, color or self._log_color))

  def actions(self) -> Sequence[ActionDescriptor]:
    """Returns this app's callable actions."""
    methods = inspect.getmembers(self, predicate=inspect.ismethod)
    return [
        ActionDescriptor.from_method(m)
        for _, m in methods
        if hasattr(m, _ACTION_PROPERTY)
    ]

  def full_description(self):
    """Returns a description of the app and all the actions it supports."""
    return textwrap.dedent(f"""\
    {self.name()}: {self.description()}
    The app supports the following actions:
    """) + '\n'.join(f'{a.name}: {a.description}' for a in self.actions())

  def invoke_action(self, action: ActionDescriptor, args_text: str) -> str:
    r"""Invokes the action on this app instance with the given arguments.

    Args:
      action: The action to invoke.
      args_text: The arguments to pass to the action, each in its own line with
        a colon separating the parameter name from the value, for example:
        'param1: value1\nparam2: value2'

    Returns:
      Textual description of the result of invoking the action.

    Raises:
      ActionArgumentError: If any of the arguments expected by the action are
      missing or if unexpected arguments are provided.
    """
    args = _parse_argument_text(args_text)
    expected_params = {p.name: p for p in action.parameters}

    # Check for missing arguments
    missing_args = set(expected_params) - set(args)
    if missing_args:
      raise ActionArgumentError(
          f"Missing argument(s): {', '.join(missing_args)}"
      )

    # Check for unexpected arguments
    unexpected_args = set(args) - set(expected_params)
    if unexpected_args:
      raise ActionArgumentError(
          f"Unexpected argument(s): {', '.join(unexpected_args)}"
      )

    # Process values
    processed_args = {
        name: expected_params[name].value_from_text(args[name])
        for name in expected_params
    }

    return getattr(self, action.name)(**processed_args)


@dataclasses.dataclass(frozen=True)
class Phone:
  """Represent a player's phone."""

  player_name: str
  apps: Sequence[PhoneApp]

  def description(self):
    return textwrap.dedent(f"""\
    {self.player_name} has a smartphone.
    {self.player_name} uses their phone frequently to achieve their daily goals.
    {self.player_name}'s phone has only the following apps available:
    {', '.join(self.app_names())}."
    """)

  def app_names(self):
    return [a.name() for a in self.apps]


# Parse multiline argument text to a text dictionary:
# 'param1: value1\n param2: value2' is parsed to:
# {'param1': 'value1', 'param2': 'value2'}
def _parse_argument_text(args_text: str) -> dict[str, str]:
  matches = _ARGUMENT_REGEX.finditer(args_text)
  return {m.group('param'): m.group('value') for m in matches}


@dataclasses.dataclass(frozen=True, slots=True)
class _Meeting:
  time: str
  participant: str
  title: str


class ToyCalendar(PhoneApp):
  """A toy calendar app."""

  def __init__(self):
    self._meetings = []

  def name(self):
    return 'Calendar'

  def description(self):
    return 'Lets you schedule meetings with other people.'

  @app_action
  def add_meeting(self, time: str, participant: str, title: str):
    """Add a meeting to the calendar.

    This action schedules a meeting with the participant
    and sends them a notification about the meeting.

    Args:
      time: The time of the meeting, e.g., tomorrow, in two weeks.
      participant: The name of the participant.
      title: The title of the meeting, e.g., Alice / John 1:1.

    Returns:
      A description of the added meeting.
    Raises:
      ActionArgumentError: If the format of any of the arguments is invalid.
    """
    meeting = _Meeting(time=time, participant=participant, title=title)
    self._meetings.append(meeting)
    output = (
        f"🗓️ A meeting with '{meeting.participant}' was scheduled at"
        f" '{meeting.time}' with title '{meeting.title}'."
    )
    self._print(output)
    return output

  @app_action
  def check_calendar(self, num_recent_meetings: int):
    """Check the calendar for scheduled meetings.

    This action checks the calendar to view and confirm meetings.

    Args:
        num_recent_meetings (int): The number of most recent meetings to check.
                                  Use a large number (e.g., 1000) to see all
                                  meetings.

    Returns:
        str: A description of the scheduled meetings.
    """
    if not self._meetings:
      output = 'No meetings scheduled.'
    else:
      meetings_to_check = self._meetings[-num_recent_meetings:]
      output = f'Scheduled meetings (showing last {num_recent_meetings}):\n'
      for meeting in meetings_to_check:
        output += (
            f"- Title: '{meeting.title}', Time: '{meeting.time}',"
            f" Participant: '{meeting.participant}'\n"
        )

    self._print(output)
    return output

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


"""Agent components for representing observation stream."""

from collections.abc import Callable, Sequence
import datetime
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class Observation(component.Component):
  """Component that displays and adds observations to memory."""

  def __init__(
      self,
      agent_name: str,
      clock_now: Callable[[], datetime.datetime],
      timeframe: datetime.timedelta,
      memory: associative_memory.AssociativeMemory,
      component_name: str = 'Current observation',
      verbose: bool = False,
      log_colour='green',
  ):
    """Initializes the component.

    Args:
      agent_name: Name of the agent.
      clock_now: Function that returns the current time.
      timeframe: Delta from current moment to display observations from, e.g. 1h
        would display all observations made in the last hour.
      memory: Associative memory to add and retrieve observations.
      component_name: Name of this component.
      verbose: Whether to print the observations.
      log_colour: Colour to print the log.
    """
    self._agent_name = agent_name
    self._log_colour = log_colour
    self._name = component_name
    self._memory = memory
    self._timeframe = timeframe
    self._clock_now = clock_now

    self._verbose = verbose

  def name(self) -> str:
    return self._name

  def state(self):
    mems = self._memory.retrieve_time_interval(
        self._clock_now() - self._timeframe, self._clock_now(), add_time=True
    )
    # removes memories that are not observations
    mems = [mem for mem in mems if '[observation]' in mem]

    if self._verbose:
      self._log('\n'.join(mems) + '\n')
    return '\n'.join(mems) + '\n'

  def get_last_log(self):
    return {
        'Summary': 'observation',
        'state': self.state().splitlines(),
    }

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def observe(self, observation: str):
    self._memory.add(
        f'[observation] {observation}',
        tags=['observation'],
    )


class ObservationSummary(component.Component):
  """Component that summarises observations from a segment of time."""

  def __init__(
      self,
      agent_name: str,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory: associative_memory.AssociativeMemory,
      components: list[component.Component],
      component_name: str = 'Summary of observations',
      prompt: str | None = None,
      display_timeframe: bool = True,
      verbose: bool = False,
      log_colour='green',
  ):
    """Initializes the component.

    Args:
      agent_name: Name of the agent.
      model: Language model to summarise the observations.
      clock_now: Function that returns the current time.
      timeframe_delta_from: delta from the current moment to the begnning of the
        segment to summarise, e.g. 4h would summarise all observations that
        happened from 4h ago until clock_now minus timeframe_delta_until.
      timeframe_delta_until: delta from the current moment to the end of the
        segment to summarise.
      memory: Associative memory retrieve observations from.
      components: List of components to summarise.
      component_name: Name of the component.
      prompt: Language prompt for summarising memories and components.
      display_timeframe: Whether to display the time interval as text.
      verbose: Whether to print the observations.
      log_colour: Colour to print the log.
    """
    self._model = model
    self._agent_name = agent_name
    self._log_colour = log_colour
    self._name = component_name
    self._memory = memory
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._clock_now = clock_now
    self._components = components
    self._state = ''
    self._display_timeframe = display_timeframe
    self._prompt = prompt or (
        'Summarize the observations above into one sentence '
        f'about {self._agent_name}.'
    )

    self._verbose = verbose
    self._history = []

  def name(self) -> str:
    return self._name

  def state(self):
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def get_components(self) -> Sequence[component.Component]:
    return self._components

  def update(self):
    context = '\n'.join([
        f"{self._agent_name}'s " + (comp.name() + ':\n' + comp.state())
        for comp in self._components
    ])

    segment_start = self._clock_now() - self._timeframe_delta_from
    segment_end = self._clock_now() - self._timeframe_delta_until

    mems = self._memory.retrieve_time_interval(
        segment_start,
        segment_end,
        add_time=True,
    )

    # removes memories that are not observations
    mems = [mem for mem in mems if '[observation]' in mem]

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(context + '\n')
    prompt.statement(
        f'Recent observations of {self._agent_name}:\n' + f'{mems}\n'
    )
    self._state = (
        self._agent_name
        + ' '
        + prompt.open_question(
            self._prompt,
            answer_prefix=f'{self._agent_name} ',
            max_tokens=1200,
        )
    )

    if self._display_timeframe:
      if segment_start.date() == segment_end.date():
        interval = segment_start.strftime(
            '%d %b %Y [%H:%M:%S  '
        ) + segment_end.strftime('- %H:%M:%S]: ')
      else:
        interval = segment_start.strftime(
            '[%d %b %Y %H:%M:%S  '
        ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
      self._state = f'{interval} {self._state}'

    if self._verbose:
      self._log(self._state)

    update_log = {
        'date': self._clock_now(),
        'Summary': 'observation summary',
        'State': self._state,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)

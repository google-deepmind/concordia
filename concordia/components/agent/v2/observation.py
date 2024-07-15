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

"""A simple component to receive observations."""

from collections.abc import Callable, Mapping
import datetime
import types

from concordia.associative_memory import associative_memory
from concordia.components.agent.v2 import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
import termcolor


class Observation(action_spec_ignored.ActionSpecIgnored):
  """A simple component to receive observations."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      timeframe: datetime.timedelta,
      memory: associative_memory.AssociativeMemory,
      verbose: bool = False,
      log_color: str = 'green',
  ):
    """Initializes the observation component.

    Args:

      clock_now: Function that returns the current time.
      timeframe: Delta from current moment to display observations from, e.g. 1h
        would display all observations made in the last hour.
      memory: Memory bank to add and retrieve observations.
      verbose: Whether to print the observations.
      log_color: Color to print the log.
    """
    self._timeframe = timeframe
    self._clock_now = clock_now
    self._memory = memory

    self._verbose = verbose
    self._log_color = log_color

  def pre_observe(
      self,
      observation: str,
  ) -> str:
    self._memory.add(
        f'[observation] {observation.strip()}',
        tags=['observation'],
    )
    return ''

  def make_pre_act_context(self) -> str:
    mems = self._memory.retrieve_time_interval(
        self._clock_now() - self._timeframe, self._clock_now(), add_time=True
    )
    if self._verbose:
      self._log('\n'.join(mems) + '\n')
    # removes memories that are not observations
    mems = [mem for mem in mems if '[observation]' in mem]
    return '\n'.join(mems) + '\n'

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def get_last_log(self):
    return {
        'Summary': 'observation',
        'state': self.get_pre_act_context().splitlines(),
    }


class ObservationSummary(action_spec_ignored.ActionSpecIgnored):
  """Component that summarizes observations from a segment of time."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory: associative_memory.AssociativeMemory,
      components: Mapping[str, action_spec_ignored.ActionSpecIgnored] = (
          types.MappingProxyType({})
      ),
      prompt: str | None = None,
      display_timeframe: bool = True,
      verbose: bool = False,
      log_color='green',
  ):
    """Initializes the component.

    Args:
      model: Language model to summarise the observations.
      clock_now: Function that returns the current time.
      timeframe_delta_from: delta from the current moment to the begnning of the
        segment to summarise, e.g. 4h would summarise all observations that
        happened from 4h ago until clock_now minus timeframe_delta_until.
      timeframe_delta_until: delta from the current moment to the end of the
        segment to summarise.
      memory: Associative memory retrieve observations from.
      components: Components to summarise along with the observations.
      prompt: Language prompt for summarising memories and components.
      display_timeframe: Whether to display the time interval as text.
      verbose: Whether to print the observations.
      log_color: Color to print the log.
    """
    self._model = model
    self._clock_now = clock_now
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._memory = memory
    self._components = dict(components)

    self._prompt = prompt or (
        'Summarize the observations above into one or two sentences.'
    )
    self._display_timeframe = display_timeframe

    self._verbose = verbose
    self._log_color = log_color

    self._last_log = None

  def make_pre_act_context(self) -> str:
    agent_name = self.get_entity().name
    context = '\n'.join([
        f"{agent_name}'s {key}:\n{component.get_pre_act_context()}"
        for key, component in self._components.items()
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
        f'Recent observations of {agent_name}:\n' + f'{mems}\n'
    )
    result = (
        agent_name
        + ' '
        + prompt.open_question(
            self._prompt,
            answer_prefix=f'{agent_name} ',
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
      result = f'{interval} {result}'

    if self._verbose:
      self._log(result)

    self._last_log = {
        'Summary': 'observation summary',
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }

    return result

  def get_last_log(self):
    if self._last_log:
      return self._last_log.copy()

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

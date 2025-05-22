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

from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import logging


DEFAULT_OBSERVATION_PRE_ACT_KEY = 'Observation'
DEFAULT_OBSERVATION_SUMMARY_PRE_ACT_KEY = 'Summary of recent observations'


class Observation(action_spec_ignored.ActionSpecIgnored):
  """A simple component to receive observations."""

  def __init__(
      self,
      clock_now: Callable[[], datetime.datetime],
      timeframe: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
      pre_act_key: str = DEFAULT_OBSERVATION_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the observation component.

    Args:

      clock_now: Function that returns the current time.
      timeframe: Delta from current moment to display observations from, e.g. 1h
        would display all observations made in the last hour.
      memory_component_name: Name of the memory component to add observations to
        in `pre_observe` and to retrieve observations from in `pre_act`.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._clock_now = clock_now
    self._timeframe = timeframe
    self._memory_component_name = memory_component_name

    self._logging_channel = logging_channel

  def pre_observe(
      self,
      observation: str,
  ) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[observation] {observation}',
        metadata={'tags': ['observation']},
    )
    return ''

  def _make_pre_act_value(self) -> str:
    """Returns the latest observations to preact."""
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
        time_from=self._clock_now() - self._timeframe,
        time_until=self._clock_now(),
        add_time=True,
    )
    mems = memory.retrieve(scoring_fn=interval_scorer)
    # Remove memories that are not observations.
    mems = [mem.text for mem in mems if '[observation]' in mem.text]
    result = '\n'.join(mems) + '\n'
    self._logging_channel(
        {'Key': self.get_pre_act_key(), 'Value': result.splitlines()})

    return result


class ObservationSummary(action_spec_ignored.ActionSpecIgnored):
  """Component that summarizes observations from a segment of time."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe_delta_from: datetime.timedelta,
      timeframe_delta_until: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[str, str] = types.MappingProxyType({}),
      prompt: str | None = None,
      display_timeframe: bool = True,
      pre_act_key: str = DEFAULT_OBSERVATION_SUMMARY_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
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
      memory_component_name: Name of the memory component from which to retrieve
        observations to summarize.
      components: The components to condition the summary on. This is a mapping
        of component name to its label in the context for the
        summarization prompt.
      prompt: Language prompt for summarising memories and components.
      display_timeframe: Whether to display the time interval as text.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe_delta_from = timeframe_delta_from
    self._timeframe_delta_until = timeframe_delta_until
    self._memory_component_name = memory_component_name
    self._components = dict(components)

    self._prompt = prompt or (
        'Summarize the observations above into one or two sentences.'
    )
    self._display_timeframe = display_timeframe

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    context = '\n'.join([
        f"{agent_name}'s"
        f' {label}:\n{self.get_named_component_pre_act_value(key)}'
        for key, label in self._components.items()
    ])

    segment_start = self._clock_now() - self._timeframe_delta_from
    segment_end = self._clock_now() - self._timeframe_delta_until

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
        time_from=segment_start,
        time_until=segment_end,
        add_time=True,
    )
    mems = memory.retrieve(scoring_fn=interval_scorer)

    # removes memories that are not observations
    mems = [mem.text for mem in mems if '[observation]' in mem.text]

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

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    return result

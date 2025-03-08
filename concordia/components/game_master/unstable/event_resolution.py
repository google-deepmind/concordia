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

"""Component that helps a game master decide whose turn is next."""

from collections.abc import Callable, Mapping, Sequence
import types

from concordia.components.agent.unstable import action_spec_ignored
from concordia.components.agent.unstable import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import logging
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import entity_component


DEFAULT_RESOLUTION_PRE_ACT_KEY = '\nEvent'

GET_ACTIVE_ENTITY_QUERY = ('Which entity just took an action?'
                           ' Respond using only the entity\'s name and no'
                           ' other words.')
GET_PUTATIVE_ACTION_QUERY = 'What is {name} attempting to do?'

EVENT_TAG = '[event]'


class EventResolution(entity_component.ContextComponent):
  """A component that resolves events.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      event_resolution_steps: (
          Sequence[
              Callable[
                  [interactive_document.InteractiveDocument, str, str], str
              ]
          ]
          | None
      ) = None,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      pre_act_key: str = DEFAULT_RESOLUTION_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      event_resolution_steps: thinking steps for the event resolution component
        to use whenever it converts putative events like action suggestions into
        real events in the simulation.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._event_resolution_steps = event_resolution_steps
    self._components = dict(components)
    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

    self._active_entity_name = None
    self._putative_action = None

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    self._active_entity_name = None
    self._putative_action = None
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      entity_name = self.get_entity().name
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f"{entity_name}'s"
          f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'Statements:\n{component_states}\n')
      self._active_entity_name = prompt.open_question(
          GET_ACTIVE_ENTITY_QUERY)
      self._putative_action = prompt.open_question(
          GET_PUTATIVE_ACTION_QUERY.format(name=self._active_entity_name),
          max_tokens=1200)
      prompt, event_statement = thought_chains.run_chain_of_thought(
          thoughts=self._event_resolution_steps,
          premise=self._putative_action,
          document=prompt,
          active_player_name=self._active_entity_name,
      )
      result = f'{self._pre_act_key}: {event_statement}'
      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_key,
         'Value': result,
         'Prompt': prompt_to_log})
    return result

  def get_active_entity_name(self) -> str | None:
    """Returns the name of the entity that just took an action."""
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    """Returns the putative action from the entity that just took an action."""
    return self._putative_action


class EventMemorizer(entity_component.ContextComponent):
  """A component that records events by writing them to memory.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      exclude: Sequence[str] = tuple(['Yes', 'No', 'type: free']),
      pre_act_key: str = '',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      memory_component_name: The name of the memory component in which to write
        records of events.
      exclude: Specific events to exclude from the memory.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._memory_component_name = memory_component_name
    self._exclude = exclude
    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel
    self._latest_action_spec = None

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec = action_spec
    return ''

  def post_act(
      self,
      event: str,
  ) -> str:
    """Record the `event` in memory."""
    if self._latest_action_spec == entity_lib.OutputType.RESOLVE:
      memory = self.get_entity().get_component(
          self._memory_component_name, type_=memory_component.Memory
      )
      if event not in self._exclude:
        memory.add(f'{EVENT_TAG} {event}')
    return ''


class DisplayEvents(action_spec_ignored.ActionSpecIgnored):
  """A component that displays recent events in `pre_act` loaded from memory.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      num_events_to_retrieve: int = 100,
      pre_act_key: str = 'Recent events',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      memory_component_name: The name of the memory component in which to write
        records of events.
      num_events_to_retrieve: The number of events to retrieve.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._num_events_to_retrieve = num_events_to_retrieve
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.Memory
    )
    events = memory.scan(selector_fn=lambda x: EVENT_TAG in x)

    limit = self._num_events_to_retrieve
    if limit > len(events):
      limit = len(events)
    events = events[-limit:]

    return '\n'.join(events)

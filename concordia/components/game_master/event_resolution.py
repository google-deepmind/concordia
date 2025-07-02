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

from collections.abc import Callable, Sequence

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


DEFAULT_RESOLUTION_COMPONENT_KEY = '__resolution__'
DEFAULT_RESOLUTION_PRE_ACT_LABEL = 'Event'
DEFAULT_SEND_PRE_ACT_VALUES_TO_PLAYERS_PRE_ACT_LABEL = ''

PUTATIVE_EVENT_TAG = '[putative_event]'
EVENT_TAG = '[event]'


class EventResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
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
      components: Sequence[str] = (),
      notify_observers: bool = False,
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      next_acting_component_key: str = (
          next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      ),
      pre_act_label: str = DEFAULT_RESOLUTION_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      event_resolution_steps: thinking steps for the event resolution component
        to use whenever it converts putative events like action suggestions into
        real events in the simulation.
      components: Keys of components to condition the event resolution on.
      notify_observers: Whether to explicitly notify observers of the event.
      make_observation_component_key: The name of the MakeObservation component
        to use to notify observers of the event.
      memory_component_key: The name of the Memory component to use to retrieve
        memories.
      next_acting_component_key: The name of the NextActing component to use
        to get the name of the player whose turn it is.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._event_resolution_steps = event_resolution_steps
    self._components = components
    self._notify_observers = notify_observers
    self._pre_act_label = pre_act_label

    self._make_observation_component_key = make_observation_component_key
    self._memory_component_key = memory_component_key
    self._next_acting_component_key = next_acting_component_key

    self._active_entity_name = None
    self._putative_action = None

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    observers_prompt_to_log = ''
    self._active_entity_name = None
    self._putative_action = None
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      self._active_entity_name = self.get_entity().get_component(
          self._next_acting_component_key,
          type_=next_acting_components.NextActing
      ).get_currently_active_player()
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component.Memory
      )
      suggestions = memory.scan(selector_fn=lambda x: PUTATIVE_EVENT_TAG in x)
      if not suggestions:
        raise RuntimeError('No suggested events to resolve.')
      if self._active_entity_name is None:
        raise RuntimeError('No active entity suggesting an event to resolve.')

      putative_action = suggestions[-1][
          suggestions[-1].find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG) :
      ]

      # Check if the action starts with the active entity name and a colon,
      # remove it if present, and strip leading whitespace.
      prefix_to_remove = f' {self._active_entity_name}:'
      if putative_action.startswith(prefix_to_remove):
        self._putative_action = putative_action[
            len(prefix_to_remove) :
        ].strip()
      else:
        self._putative_action = putative_action

      putative_action = f'Putative event to resolve: {putative_action}'
      prompt.statement(putative_action)
      prompt, event_statement = thought_chains.run_chain_of_thought(
          thoughts=self._event_resolution_steps,
          premise=putative_action,
          document=prompt,
          active_player_name=self._active_entity_name,
      )

      observers_prompt_to_log = ''
      if self._notify_observers:
        make_observation = self.get_entity().get_component(
            self._make_observation_component_key,
            type_=make_observation_component.MakeObservation,
        )
        observers_prompt = prompt.copy()
        observers_prompt.statement(f'Event that occurred: {event_statement}')
        observer_names_str = observers_prompt.open_question(
            question=('Which entities are aware of the event? Answer with a '
                      'comma-separated list of entity names.')
        )
        observer_names = observer_names_str.split(',')
        for name in observer_names:
          make_observation.add_to_queue(name.strip(' .,'), event_statement)

        observers_prompt_to_log = observers_prompt.view().text()

      result = f'{self._pre_act_label}: {event_statement}\n'
      prompt_to_log = prompt.view().text()

    self._log(
        key=self._pre_act_label,
        value=result,
        prompt=prompt_to_log,
        observers_prompt=observers_prompt_to_log,
    )
    return result

  def get_active_entity_name(self) -> str | None:
    """Returns the name of the entity that just took an action."""
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    """Returns the putative action from the entity that just took an action."""
    return self._putative_action

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        '_active_entity_name': self._active_entity_name,
        '_putative_action': self._putative_action,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._active_entity_name = state['_active_entity_name']
    self._putative_action = state['_putative_action']

  def _log(self, key: str, value: str, prompt: str, observers_prompt: str):
    self._logging_channel({
        'Key': key,
        'Summary': value,
        'Value': value,
        'Prompt': prompt,
        'Details': {'Observers prompt': observers_prompt,},
    })


class DisplayEvents(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A component that displays recent events in `pre_act` loaded from memory.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      num_events_to_retrieve: int = 100,
      pre_act_label: str = 'Recent events',
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      memory_component_key: The name of the memory component in which to write
        records of events.
      num_events_to_retrieve: The number of events to retrieve.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._num_events_to_retrieve = num_events_to_retrieve

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    events = memory.scan(selector_fn=lambda x: EVENT_TAG in x)

    limit = self._num_events_to_retrieve
    if limit > len(events):
      limit = len(events)
    events = events[-limit:]
    events = [f'{i}). {event.split(EVENT_TAG)[-1]}'
              for i, event in enumerate(events)]

    events_str = '\n'.join(events)

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Summary': events_str,
        'Value': events_str,
    })

    return events_str

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass


class SendEventToRelevantPlayers(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that passes events to relevant players."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      make_observation_component_key: str,
      pre_act_label: str = DEFAULT_SEND_PRE_ACT_VALUES_TO_PLAYERS_PRE_ACT_LABEL,
  ):
    """Initializes a component that sends component pre-act values to players.

    This component sends the events during the post-act phase to the players
    that are aware of the event, by queueing them in the MakeObservation
    component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players.
      make_observation_component_key: he key for a
        MakeObservation component to add the pre-act values to the queue of
        events to observe.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._player_names = player_names
    self._pre_act_label = pre_act_label
    self._queue = {}
    self._last_action_spec = None
    self._optional_make_observation_component_key = (
        make_observation_component_key)

    self._map_names_to_previous_observations = {
        player_name: '' for player_name in player_names
    }

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._last_action_spec = action_spec
    return ''

  def post_act(
      self,
      action_attempt: str,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if (
        self._last_action_spec
        and self._last_action_spec.output_type == entity_lib.OutputType.RESOLVE
    ):
      prompt = interactive_document.InteractiveDocument(self._model)
      for active_entity_name in self._player_names:
        prompt.statement(f'{action_attempt}\n')
        proceed = prompt.yes_no_question(
            question=f'Is {active_entity_name} aware of the latest event above?'
        )

        if proceed:
          # Remove their previous observation since they have already seen it.
          result = action_attempt.replace(
              self._map_names_to_previous_observations[active_entity_name], ''
          )
        if self._optional_make_observation_component_key:
          make_observation = self.get_entity().get_component(
              self._optional_make_observation_component_key,
              type_=make_observation_component.MakeObservation,
          )
          make_observation.add_to_queue(active_entity_name, result)

        self._map_names_to_previous_observations[active_entity_name] += result

      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_label,
         'Summary': result,
         'Value': result,
         'Prompt': prompt_to_log}
    )
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        '_queue': self._queue,
        '_last_action_spec': self._last_action_spec,
        '_map_names_to_previous_observations': (
            self._map_names_to_previous_observations
        ),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._queue = state['_queue']
    self._last_action_spec = state['_last_action_spec']
    self._map_names_to_previous_observations = state[
        '_map_names_to_previous_observations'
    ]

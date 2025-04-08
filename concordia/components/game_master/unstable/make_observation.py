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

from collections.abc import Mapping, Sequence
import types

from concordia.components.agent.unstable import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import logging
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import entity_component


DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY = '__make_observation__'
DEFAULT_MAKE_OBSERVATION_PRE_ACT_LABEL = '\nPrompt'
DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}? What do they now observe?'
    ' Only include information of which they are aware.')

GET_ACTIVE_ENTITY_QUERY = (
    'Who is being asked about? Respond using only their name and no other '
    'words. Use their full name if known.'
)


class MakeObservation(entity_component.ContextComponent):
  """A component that generates observations to send to players.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      reformat_observations_in_specified_style: str = '',
      pre_act_label: str = DEFAULT_MAKE_OBSERVATION_PRE_ACT_LABEL,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      reformat_observations_in_specified_style: If non-empty, the component
        will ask the model to reformat the observation to fit the style
        specified in this string. By default, the component does not reformat
        the observation. When turned on, a reasonable starting style to try in
        the case that you have information from a time component that you want
        to include in the observation is:
        "The format to use when describing the current situation to a player is:
        "//date or time//situation description"."
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._components = dict(components)
    self._reformat_observations_in_specified_style = (
        reformat_observations_in_specified_style
    )
    self._pre_act_label = pre_act_label
    self._logging_channel = logging_channel

    self._queue = {}

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
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      prompt.statement(
          f'Working out the answer to: "{action_spec.call_to_action}"')
      active_entity_name = prompt.open_question(
          GET_ACTIVE_ENTITY_QUERY).strip()

      if active_entity_name in self._queue and self._queue[active_entity_name]:
        result = ''
        for event in self._queue[active_entity_name]:
          result += event + '\n'

        self._queue[active_entity_name] = []
      else:
        result = prompt.open_question(
            question=(f'What does {active_entity_name} observe now? Never '
                      'repeat information that was already provided to '
                      f'{active_entity_name} unless absolutely necessary. Keep '
                      'the story moving forward.'),
            max_tokens=1200)

      if self._reformat_observations_in_specified_style:
        prompt.statement(
            'Required observation format: '
            f'{self._reformat_observations_in_specified_style}')
        result_without_newlines = result.replace('\n', '').strip()
        correct_format = prompt.yes_no_question(
            question=(
                f'Draft: {active_entity_name} will observe:'
                f' "{result_without_newlines}"\nIs the draft formatted'
                ' correctly in the specified format?'
            )
        )
        if not correct_format:
          result = prompt.open_question(
              question=(f'Reformat {active_entity_name}\'s draft observation '
                        'to fit the required format.'),
              max_tokens=1200,
              terminators=())

      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_label,
         'Value': result,
         'Prompt': prompt_to_log})
    return result

  def add_to_queue(self, entity_name: str, event: str):
    """Adds an event to the queue of events to observe."""
    if entity_name not in self._queue:
      self._queue[entity_name] = []
    self._queue[entity_name].append(event)


class MakeObservationFromQueueOnly(entity_component.ContextComponent):
  """A component that decides which game master to use next."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      reformat_observations_in_specified_style: str = '',
      pre_act_label: str = DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      reformat_observations_in_specified_style: If non-empty, the component will
        ask the model to reformat the observation to fit the style specified in
        this string. By default, the component does not reformat the
        observation. When turned on, a reasonable starting style to try in the
        case that you have information from a time component that you want to
        include in the observation is: "The format to use when describing the
        current situation to a player is: "//date or time//situation
        description"."
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._pre_act_label = pre_act_label
    self._logging_channel = logging_channel
    self._reformat_observations_in_specified_style = (
        reformat_observations_in_specified_style
    )
    self._queue = {}

    self._currently_active_game_master = None

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(
          f'Working out the answer to: "{action_spec.call_to_action}"'
      )
      active_entity_name = prompt.open_question(GET_ACTIVE_ENTITY_QUERY).strip()

      if active_entity_name in self._queue and self._queue[active_entity_name]:
        result = ''
        for event in self._queue[active_entity_name]:
          result += event + '\n'

        self._queue[active_entity_name] = []
      else:
        result = ''

      if self._reformat_observations_in_specified_style:
        prompt.statement(
            'Required observation format: '
            f'{self._reformat_observations_in_specified_style}'
        )
        result_without_newlines = result.replace('\n', '').strip()
        correct_format = prompt.yes_no_question(
            question=(
                f'Draft: {active_entity_name} will observe:'
                f' "{result_without_newlines}"\nIs the draft formatted'
                ' correctly in the specified format?'
            )
        )
        if not correct_format:
          result = prompt.open_question(
              question=(
                  f"Reformat {active_entity_name}'s draft observation "
                  'to fit the required format.'
              ),
              max_tokens=1200,
              terminators=(),
          )

      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_label, 'Value': result, 'Prompt': prompt_to_log}
    )
    return result

  def get_currently_active_game_master(self) -> str | None:
    return self._currently_active_game_master


class SendComponentPreActValuesToPlayers(entity_component.ContextComponent):
  """A component that passes component pre-act values to players.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      pre_act_label: str = DEFAULT_MAKE_OBSERVATION_PRE_ACT_LABEL,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._player_names = player_names
    self._components = dict(components)
    self._pre_act_label = pre_act_label
    self._logging_channel = logging_channel

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

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(
          f'Working out the answer to: "{action_spec.call_to_action}"')
      active_entity_idx = prompt.multiple_choice_question(
          question=GET_ACTIVE_ENTITY_QUERY,
          answers=self._player_names)
      active_entity_name = self._player_names[active_entity_idx]
      component_states_string = '\n'.join([
          f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states_string}\n')
      proceed = prompt.yes_no_question(
          question=(
              f'Is {active_entity_name} aware of the latest event above?'))
      if proceed:
        # Remove their previous observation since they have already seen it.
        result = component_states_string.replace(
            self._map_names_to_previous_observations[active_entity_name], '')

      self._map_names_to_previous_observations[active_entity_name] += result
      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_label,
         'Value': result,
         'Prompt': prompt_to_log})
    return result

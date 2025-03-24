# Copyright 2025 DeepMind Technologies Limited.
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

"""A component to represent any world state variables the GM deems important."""

from collections.abc import Mapping, Sequence
import types

from concordia.components.agent.unstable import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import logging
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import entity_component


class WorldState(entity_component.ContextComponent):
  """A component that represents the world state."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      pre_act_key: str = '\nState',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the world state component.

    Args:
      model: The language model to use.
      components: The components to condition the world state changes on. It
        is a mapping of the component name to a label to use in the prompt.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    self._pre_act_key = pre_act_key
    self._model = model
    self._components = dict(components)
    self._logging_channel = logging_channel

    self._state = {}
    self._latest_action_spec = None

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def get_pre_act_key(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_key

  def get_pre_act_value(self) -> str:
    """Returns the current world state."""
    result = '\n'.join([
        f'{name}: {value}' for name, value in self._state.items()
    ])
    return result + '\n'

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec = action_spec
    result = self.get_pre_act_value()
    self._logging_channel({
        'Key': self._pre_act_key,
        'Value': result,
    })
    return result

  def post_act(
      self,
      event: str,
  ) -> str:
    if (self._latest_action_spec is not None and
        self._latest_action_spec.output_type == entity_lib.OutputType.RESOLVE):
      prompt = interactive_document.InteractiveDocument(self._model)

      component_states = '\n'.join([
          f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'\n{component_states}\n')
      prompt.statement(f'State prior to the latest event:\n{self._state}')
      prompt.statement(f'The latest event: {event}')
      important_variables_str = prompt.open_question(
          question=(
              'Given the context above, what state variables are important '
              'to write down now so that they can be used later? '
              'Respond with a comma-separated list of variable names. '
              'Variables must be written in the format: "name|value". '
              'For example: "name1|value1,name2|value2,name3|value3". '
              'Useful variables reflect properties of the protagonists, '
              'or the world they inhabit, that are expected to change in '
              'value over time and expected to exert some influence on '
              'how the story plays out. Update the value for all existing '
              'state variables and add new state variables as needed to '
              'account for consequences of the latest event. If a '
              'particular state variable is no longer needed then there '
              'is no need to include it in the list.'
          ),
          max_tokens=512,
      )
      important_variables = important_variables_str.split(',')
      for variable in important_variables:
        split_variable = variable.split('|')
        if len(split_variable) == 2:
          name, value = variable.split('|')
          self._state[name.strip()] = value.strip()

    return ''


class Locations(entity_component.ContextComponent):
  """A component that represents locations of entities in the world."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      entity_names: Sequence[str],
      prompt: str,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      pre_act_key: str = '\nEntity locations',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use.
      entity_names: Names of entities to track locations for.
      prompt: description of all locations to be specifically represented in the
        world. This is used to prompt the model to generate concrete variables
        representing the locations and their properties (e.g. their topology). 
      components: The components to condition location changes on. It
        is a mapping of the component name to a label to use in the prompt.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.
    """
    self._pre_act_key = pre_act_key
    self._model = model
    self._entity_names = entity_names
    self._prompt = prompt
    self._components = dict(components)
    self._logging_channel = logging_channel

    self._locations = {}
    self._entity_locations = {name: '' for name in entity_names}
    self._latest_action_spec = None

    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(self._prompt)
    locations_str = chain_of_thought.open_question(
        question=(
            'Given the context above, what locations are important to write '
            'down now so that they can be used later? Respond with a '
            'comma-separated list of locations using the format: '
            '"location 1|properties of location 1,'
            'location 2|properties of location 2,'
            'location 3|properties of location 3,...".'
        ),
        max_tokens=1000,
        terminators=(),
    )
    locations_and_properties = locations_str.split(',')
    for location_and_property_str in locations_and_properties:
      location_and_property = location_and_property_str.strip().split('|')
      if len(location_and_property) == 2:
        location, properties = location_and_property
        self._locations[location.strip()] = properties.strip()

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def get_pre_act_key(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_key

  def get_pre_act_value(self) -> str:
    """Returns the current world state."""
    result = '\n'.join([
        f'{name}: {value}' for name, value in self._entity_locations.items()
        if value
    ])
    return result + '\n'

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec = action_spec
    result = self.get_pre_act_value()
    self._logging_channel({
        'Key': self._pre_act_key,
        'Value': result,
    })
    return result

  def post_act(
      self,
      event: str,
  ) -> str:
    if (self._latest_action_spec is not None and
        self._latest_action_spec.output_type == entity_lib.OutputType.RESOLVE):
      prompt = interactive_document.InteractiveDocument(self._model)

      component_states = '\n'.join([
          f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'\n{component_states}\n')
      locations_and_properties = '\n'.join([
          f'{name}: {value}' for name, value in self._locations.items()
      ])
      prompt.statement('All locations and their properties:\n'
                       f'{locations_and_properties}')
      prompt.statement('Known location of each entity prior to the latest '
                       f'event:\n{self._entity_locations}')
      prompt.statement(f'The latest event: {event}')
      entity_locations_str = prompt.open_question(
          question=(
              'Given the context above, where are the named people currently '
              'located? Respond with an empty string for anyone whose '
              'current location is unknown. Format the response as a '
              'comma-separated list e.g. '
              '"person1|location1,person2|location2,person3|location3,...".'),
          max_tokens=512,
      )
      names_to_locations = entity_locations_str.split(',')
      for name_to_location_str in names_to_locations:
        name_and_location = name_to_location_str.strip().split('|')
        if len(name_and_location) == 2:
          name, location = name_and_location
          if name.strip() in self._entity_names:
            self._entity_locations[name.strip()] = location.strip()

    return ''

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

from collections.abc import Sequence

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class WorldState(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that represents the world state."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Sequence[str] = (),
      pre_act_label: str = '\nState',
  ):
    """Initializes the world state component.

    Args:
      model: The language model to use.
      components: Keys of components to condition the world state on.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    self._pre_act_label = pre_act_label
    self._model = model
    self._components = components

    self._state = {}
    self._latest_action_spec = None

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

  def get_pre_act_label(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_label

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
        'Key': self._pre_act_label,
        'Summary': result,
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

      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
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

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'state': self._state,
        'latest_action_spec': self._latest_action_spec,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._state = state['state']
    self._latest_action_spec = state['latest_action_spec']


class Locations(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that represents locations of entities in the world."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      entity_names: Sequence[str],
      prompt: str,
      components: Sequence[str] = (),
      pre_act_label: str = '\nEntity locations',
  ):
    """Initializes the component.

    Args:
      model: The language model to use.
      entity_names: Names of entities to track locations for.
      prompt: description of all locations to be specifically represented in the
        world. This is used to prompt the model to generate concrete variables
        representing the locations and their properties (e.g. their topology). 
      components: Keys of components to condition entity locations on.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    self._pre_act_label = pre_act_label
    self._model = model
    self._entity_names = entity_names
    self._prompt = prompt
    self._components = components

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

  def get_pre_act_label(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_label

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
        'Key': self._pre_act_label,
        'Summary': result,
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

      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
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

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'locations': self._locations,
        'entity_locations': self._entity_locations,
        'latest_action_spec': self._latest_action_spec,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._locations = state['locations']
    self._entity_locations = state['entity_locations']
    self._latest_action_spec = state['latest_action_spec']


class GenerativeClock(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component to represent a generative clock updated via language model."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      prompt: str,
      start_time: str,
      components: Sequence[str] = (),
      format_description_key: str = 'Clock format description',
      pre_act_label: str = '\nClock',
  ):
    """Initializes the component.

    This component is used to represent a clock that is updated by the language
    model based on the specified components and the latest event. The clock is
    updated on each step of the simulation, and is used to represent the
    current time or date.

    Args:
      model: The language model to use.
      prompt: description of what the clock represents, how it gets updated,
        and what it is used for.
      start_time: The initial time of the clock.
      components: Keys of components to condition clock updates on.
      format_description_key: The key to prepend to the descripton of the
        desired format to use in the prompt for the sample that produces the
        clock's update on each step.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    self._pre_act_label = pre_act_label
    self._model = model
    self._format_description_key = format_description_key
    self._prompt = prompt
    self._components = components

    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(self._prompt)
    chain_of_thought.statement(f'Start time: {start_time}')
    self._clock_description = chain_of_thought.open_question(
        question=(
            'Given the context above, when is the clock updated? How are times '
            'represented internally? (usually as a number of steps). And, how '
            'do internal time repesentations map to the time representations '
            'communicated to players?'
        ),
        max_tokens=1000,
        terminators=(),
    )

    self._num_steps = 0
    self._time = start_time
    self._prompt_to_log = ''
    self._latest_action_spec = None

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

  def get_pre_act_label(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_label

  def get_pre_act_value(self) -> str:
    """Returns the current world state."""
    return self._time + '\n'

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._latest_action_spec = action_spec
    result = self.get_pre_act_value()
    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'Prompt': self._prompt_to_log,
    })
    return result

  def post_act(
      self,
      event: str,
  ) -> str:
    if (self._latest_action_spec is not None and
        self._latest_action_spec.output_type == entity_lib.OutputType.RESOLVE):
      prompt = interactive_document.InteractiveDocument(self._model)

      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'\n{component_states}\n')

      prompt.statement(
          f'{self._format_description_key}:\n{self._clock_description}\n***\n')

      prompt.statement(f'Number of simulation steps so far: {self._num_steps}')
      prompt.statement(f'Time prior to the latest event: {self._time}')

      self._num_steps += 1
      prompt.statement(f'The next event: {event}')
      prompt.statement('After the next event, the number of simulation steps '
                       f'will be: {self._num_steps}.')
      self._time = prompt.open_question(
          question=(
              'Given the context above, and after the event, what is the new '
              'time? Never respond with a sentence like "the time is unchanged"'
              ' or anything to that effect. Also never respond with "unknown". '
              'Correct responses always follow the '
              f'{self._format_description_key} above.'
          ),
          max_tokens=128,
      )

      self._prompt_to_log = prompt.view().text()

    return ''

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'num_steps': self._num_steps,
        'time': self._time,
        'prompt_to_log': self._prompt_to_log,
        'latest_action_spec': self._latest_action_spec,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._num_steps = state['num_steps']
    self._time = state['time']
    self._prompt_to_log = state['prompt_to_log']
    self._latest_action_spec = state['latest_action_spec']

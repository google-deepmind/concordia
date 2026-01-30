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

"""A component that represents the state of the spaceships system."""

from collections.abc import Sequence
import random

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component_module
from concordia.components.game_master import make_observation as make_observation_component_module
from concordia.components.game_master import terminate as terminate_component_module
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class SpaceshipSystem(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that represents the state of a spaceship system."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      system_name: str,
      system_max_health: int,
      system_failure_probability: float,
      warning_message: str,
      pre_act_label: str,
      memory_component_key: str = (
          memory_component_module.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      terminator_component_key: str = (
          terminate_component_module.DEFAULT_TERMINATE_COMPONENT_KEY
      ),
      observation_component_key: str = (
          make_observation_component_module.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      components: Sequence[str] = (),
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      system_name: The name of the system to represent.
      system_max_health: The maximum health of the system.
      system_failure_probability: The probability that the system will fail.
      warning_message: The warning message to display when the system fails.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      memory_component_key: The name of the memory component.
      terminator_component_key: The name of the terminator component.
      observation_component_key: The name of the observation component.
      components: The names of the components to condition the system on.
      verbose: Whether to print verbose debug information.
    """
    super().__init__()
    self._model = model
    self._system_name = system_name
    self._system_max_health = system_max_health
    self._system_failure_probability = system_failure_probability
    self._warning_message = warning_message
    self._pre_act_label = pre_act_label

    self._current_health = system_max_health
    self._is_failing = False
    self._terminator_component_key = terminator_component_key
    self._observation_component_key = observation_component_key
    self._memory_component_key = memory_component_key
    self._components = components
    self._verbose = verbose
    self._step_counter = 0

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:

      if (
          not self._is_failing
          and random.random() < self._system_failure_probability
      ):
        self._is_failing = True

      result = (
          f'System name: {self._system_name}\n'
          f'System max health: {self._system_max_health}\n'
          f'System current health: {self._current_health}\n'
      )
      if self._verbose:
        logging.info('%s', result)

      if self._is_failing:
        make_observation = self.get_entity().get_component(
            self._observation_component_key,
            type_=make_observation_component_module.MakeObservation,
        )

        if self._verbose:
          logging.info('%s', self._warning_message)
        make_observation.add_to_queue(
            'All',
            self._warning_message,
        )
        memory = self.get_entity().get_component(
            self._memory_component_key, type_=memory_component_module.Memory
        )
        memory.add(self._warning_message)
        self._current_health -= 1

      self._logging_channel({
          'Key': self._pre_act_label,
          'Value': result,
          'Measurements': {
              'System name': self._system_name,
              'System current health': self._current_health,
              'Step counter': self._step_counter,
          },
      })
      self._step_counter += 1
    return result

  def post_act(
      self,
      event: str,
  ) -> str:

    if self._is_failing:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')

      prompt.statement(f'System name: {self._system_name}')
      prompt.statement(f'System system message: {self._warning_message}')
      prompt.statement(f'Event: {event}')

      was_system_fixed = prompt.yes_no_question(
          question=(
              'Given the context above, was the system fixed during the event?'
          )
      )

      if was_system_fixed:
        self._current_health = self._system_max_health
        self._is_failing = False
        make_observation = self.get_entity().get_component(
            self._observation_component_key,
            type_=make_observation_component_module.MakeObservation,
        )
        make_observation.add_to_queue(
            'All',
            f'The {self._system_name} was fixed.',
        )
        memory = self.get_entity().get_component(
            self._memory_component_key, type_=memory_component_module.Memory
        )
        memory.add(f'The {self._system_name} was fixed.')
        if self._verbose:
          logging.info('The %s was fixed.', self._system_name)
      else:
        if self._current_health <= 0:
          terminator = self.get_entity().get_component(
              self._terminator_component_key,
              type_=terminate_component_module.Terminate,
          )
          terminator.terminate()

    return ''

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'system_name': self._system_name,
        'system_max_health': self._system_max_health,
        'system_failure_probability': self._system_failure_probability,
        'current_health': self._current_health,
        'is_failing': self._is_failing,
        'verbose': self._verbose,
        'step_counter': self._step_counter,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._system_name = state['system_name']
    self._system_max_health = state['system_max_health']
    self._system_failure_probability = state['system_failure_probability']
    self._current_health = state['current_health']
    self._is_failing = state['is_failing']
    self._verbose = state['verbose']
    self._step_counter = state['step_counter']

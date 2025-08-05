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

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component_module
from concordia.components.game_master import make_observation as make_observation_component_module
from concordia.components.game_master import next_acting as next_acting_component_module
from concordia.components.game_master import terminate as terminate_component_module
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_DEATH_COMPONENT_KEY = 'death'


class Death(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that implements the death mechanics of the game master."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str,
      actor_names: Sequence[str],
      memory_component_key: str = (
          memory_component_module.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      terminator_component_key: str = (
          terminate_component_module.DEFAULT_TERMINATE_COMPONENT_KEY
      ),
      observation_component_key: str = (
          make_observation_component_module.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      fixed_order_next_acting_component_key: str = (
          next_acting_component_module.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      ),
      death_message: str = '{actor_name} has died.',
      components: Sequence[str] = (),
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      actor_names: The names of the actors.
      memory_component_key: The name of the memory component.
      terminator_component_key: The name of the terminator component.
      observation_component_key: The name of the observation component.
      fixed_order_next_acting_component_key: The name of the next acting
        component, so that the game master can remove the actor from the
        sequence of players. Currently, only fixed order component is supported.
      death_message: The message to display when an actor dies.
      components: The names of the components to condition the system on.
      verbose: Whether to print verbose debug information.
    """
    super().__init__()
    self._model = model
    self._pre_act_label = pre_act_label

    self._actors_names = list(actor_names)
    self._terminator_component_key = terminator_component_key
    self._observation_component_key = observation_component_key
    self._memory_component_key = memory_component_key
    self._next_acting_component_key = fixed_order_next_acting_component_key
    self._death_message = death_message
    self._components = components
    self._verbose = verbose
    self._step_counter = 0
    self._last_action_spec = None

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
    self._last_action_spec = action_spec
    return ''

  def post_act(
      self,
      event: str,
  ) -> str:

    if (
        self._last_action_spec
        and self._last_action_spec.output_type != entity_lib.OutputType.RESOLVE
    ):
      return ''

    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    prompt.statement(f'{component_states}\n')
    prompt.statement(f'Event: {event}')
    prompt.statement(f'Actor names: {self._actors_names}')

    death_happened = prompt.yes_no_question(
        question=(
            'Have any of the actors died or have been killed as the result of'
            ' this event? Interpret any ambiguity as the actor(s) dying. For'
            ' example, if the actor was shot, assume they died.'
        )
    )

    if death_happened:

      who_died_str = prompt.open_question(
          question=(
              'Who died? List their names using a comma-separated list. For'
              ' example, use "John Doe, Jane Doe". If no one died, output'
              ' "NO_DEATH".'
          ),
      )
      self._logging_channel({
          'Summary': f'Who died: {who_died_str}',
          'Value': who_died_str,
          'Prompt': prompt.view().text().splitlines(),
          'Measurements': {
              'actors_alive': len(self._actors_names),
          },
      })
      if who_died_str.upper().replace('.', '').strip() == 'NO_DEATH':
        return ''
      who_died_list = who_died_str.split(',')

      for actor in who_died_list:
        actor = actor.strip().replace('.', '')
        if actor not in self._actors_names:
          continue
        self.get_entity().get_component(
            self._next_acting_component_key,
            type_=next_acting_component_module.NextActingInFixedOrder,
        ).remove_actor_from_sequence(actor)

        make_observation = self.get_entity().get_component(
            self._observation_component_key,
            type_=make_observation_component_module.MakeObservation,
        )
        make_observation.add_to_queue(
            actor,
            self._death_message.format(actor_name=actor),
        )
        memory = self.get_entity().get_component(
            self._memory_component_key, type_=memory_component_module.Memory
        )
        memory.add(self._death_message.format(actor_name=actor))
        self._actors_names.remove(actor)
      if self._verbose:
        print(self._death_message.format(actor_name=who_died_str))
    else:
      if not self._actors_names:
        terminator = self.get_entity().get_component(
            self._terminator_component_key,
            type_=terminate_component_module.Terminate,
        )
        terminator.terminate()

    return ''

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    action_spec_dict = (self._last_action_spec.to_dict()
                        if self._last_action_spec else None)
    return {
        'actors_names': self._actors_names,
        'step_counter': self._step_counter,
        'last_action_spec': action_spec_dict,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._actors_names = state['actors_names']
    self._step_counter = state['step_counter']
    action_spec_dict = state['last_action_spec']
    if action_spec_dict and isinstance(action_spec_dict, dict):
      self._last_action_spec = entity_lib.action_spec_from_dict(
          action_spec_dict
      )
    else:
      self._last_action_spec = None
    self._last_action_spec = state['last_action_spec']

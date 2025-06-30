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

from collections.abc import Sequence
import copy
import threading

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY = '__make_observation__'
DEFAULT_MAKE_OBSERVATION_PRE_ACT_LABEL = '\nPrompt'
DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}? What do they now observe?'
    ' Only include information of which they are aware.'
)

GET_ACTIVE_ENTITY_QUERY = (
    'Who is being asked about? Respond using only their name and no other '
    'words. Use their full name if known.'
)


class MakeObservation(entity_component.ContextComponent,
                      entity_component.ComponentWithLogging):
  """A component that generates observations to send to players."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Sequence[str] = (),
      reformat_observations_in_specified_style: str = '',
      pre_act_label: str = DEFAULT_MAKE_OBSERVATION_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players.
      components: Keys of components to condition the observation on.
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

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._player_names = player_names
    self._components = components
    self._reformat_observations_in_specified_style = (
        reformat_observations_in_specified_style
    )
    self._pre_act_label = pre_act_label
    self._lock = threading.Lock()

    self._queue = {}

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
    log_entry = {}
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      prompt.statement(
          f'Working out the answer to: "{action_spec.call_to_action}"'
      )
      idx = prompt.multiple_choice_question(
          question=GET_ACTIVE_ENTITY_QUERY,
          answers=self._player_names,
      )
      active_entity_name = self._player_names[idx]
      log_entry['Active Entity'] = active_entity_name
      with self._lock:
        log_entry['queue'] = copy.deepcopy(self._queue)

        if (
            active_entity_name in self._queue
            and self._queue[active_entity_name]
        ):
          log_entry['queue_active_entity'] = copy.deepcopy(
              self._queue[active_entity_name]
          )
          result = ''
          for event in self._queue[active_entity_name]:
            result += event + '\n\n\n'

          self._queue[active_entity_name] = []
        else:
          result = prompt.open_question(
              question=(
                  f'What does {active_entity_name} observe now? Never '
                  'repeat information that was already provided to '
                  f'{active_entity_name} unless absolutely necessary. Keep '
                  'the story moving forward.'
              ),
              max_tokens=1200,
          )

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
                  f'Reformat {active_entity_name}\'s draft observation '
                  'to fit the required format.'
              ),
              max_tokens=1200,
              terminators=(),
          )

      prompt_to_log = prompt.view().text()

    log_entry['Key'] = self._pre_act_label
    log_entry['Summary'] = result
    log_entry['Value'] = result
    log_entry['Prompt'] = prompt_to_log
    self._logging_channel(log_entry)
    return result

  def add_to_queue(self, entity_name: str, event: str):
    """Adds an event to the queue of events to observe."""
    with self._lock:
      if entity_name not in self._queue:
        self._queue[entity_name] = []
      self._queue[entity_name].append(event)

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    with self._lock:
      return {'queue': copy.deepcopy(self._queue)}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    with self._lock:
      self._queue = copy.deepcopy(state['queue'])







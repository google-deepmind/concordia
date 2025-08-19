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

"""Component helping a game master pick which game master to use next."""

from collections.abc import Mapping, Sequence


from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY = '__next_game_master__'

DEFAULT_NEXT_GAME_MASTER_PRE_ACT_LABEL = '\nGame Master'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    'Which rule set should we use for the next step?')


class NextGameMaster(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides which game master to use next.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      map_game_master_names_to_choices: Mapping[str, str],
      call_to_action: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
      components: Sequence[str] = (),
      pre_act_label: str = DEFAULT_NEXT_GAME_MASTER_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      map_game_master_names_to_choices: Names of game masters (rule sets) to
        choose from, mapped to the multiple choice question option corresponding
        to that game master.
      call_to_action: The question to ask the model to select a game master.
      components: Keys of components to condition the game master selection on.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._game_master_names = list(map_game_master_names_to_choices.keys())
    self._game_master_choices = list(map_game_master_names_to_choices.values())
    self._call_to_action = call_to_action
    self._components = components
    self._pre_act_label = pre_act_label

    self._currently_active_game_master = None

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def _get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self._get_component_pre_act_label(key)}:\n'
        f'{self._get_named_component_pre_act_value(key)}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      idx = prompt.multiple_choice_question(
          question=self._call_to_action,
          answers=self._game_master_choices)
      self._currently_active_game_master = self._game_master_names[idx]
      result = self._currently_active_game_master
      prompt_to_log = prompt.view().text()

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'Prompt': prompt_to_log,
    })
    return result

  def get_currently_active_game_master(self) -> str | None:
    return self._currently_active_game_master

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {'currently_active_game_master': self._currently_active_game_master}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._currently_active_game_master = state['currently_active_game_master']

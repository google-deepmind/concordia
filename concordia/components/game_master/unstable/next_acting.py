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

from concordia.components.agent import action_spec_ignored
from concordia.components.agent.unstable import memory as memory_component
from concordia.document import interactive_document
from concordia.environment.scenes.unstable import runner as scene_runner
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging


DEFAULT_NEXT_ACTING_COMPONENT_NAME = '__next_acting__'
# Initiative is the Dungeons & Dragons term for the rule system that controls
# turn taking.
DEFAULT_NEXT_ACTING_PRE_ACT_KEY = '\nInitiative'


class NextActing(entity_component.ContextComponent):
  """A component that decides whose turn is next.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      pre_act_key: str = DEFAULT_NEXT_ACTING_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players to choose from.
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
    self._player_names = player_names
    self._components = dict(components)
    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
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
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      entity_name = self.get_entity().name
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f"{entity_name}'s"
          f' {prefix}:\n{self._get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      idx = prompt.multiple_choice_question(
          question='Whose turn is next?',
          answers=self._player_names)
      result = self._player_names[idx]
      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_key,
         'Value': result,
         'Prompt': prompt_to_log})
    return result


class NextActingFromSceneSpec(entity_component.ContextComponent):
  """A component that decides whose turn is next using the current scene spec.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      pre_act_key: str = DEFAULT_NEXT_ACTING_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      memory_component_name: The name of the memory component.
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
    self._components = dict(components)
    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def _get_current_scene_participants(self) -> Sequence[str]:
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.Memory
    )
    return scene_runner.get_current_scene_participants(memory=memory)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      entity_name = self.get_entity().name
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f"{entity_name}'s"
          f' {prefix}:\n{self._get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      scene_participants = self._get_current_scene_participants()

      idx = prompt.multiple_choice_question(
          question='Whose turn is next?',
          answers=scene_participants)
      result = scene_participants[idx]
      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_key,
         'Value': result,
         'Prompt': prompt_to_log})
    return result

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
from concordia.components.agent.unstable import memory as memory_component
from concordia.document import interactive_document
from concordia.environment.scenes.unstable import runner as scene_runner
from concordia.environment.unstable import engine as engine_lib
from concordia.language_model import language_model
from concordia.typing import logging
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import entity_component
from concordia.typing.unstable import scene as scene_lib


DEFAULT_NEXT_ACTING_COMPONENT_NAME = '__next_acting__'
# Initiative is the Dungeons & Dragons term for the rule system that controls
# turn taking.
DEFAULT_NEXT_ACTING_PRE_ACT_KEY = '\nInitiative'
DEFAULT_CALL_TO_NEXT_ACTING = 'Who is next to act?'

DEFAULT_NEXT_ACTION_SPEC_COMPONENT_NAME = '__next_action_spec__'
DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_KEY = '\nType of action'
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    'In what action spec format should {name} respond?')


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

    self._currently_active_player = None

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
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f'{prefix}:\n{self._get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      idx = prompt.multiple_choice_question(
          question='Whose turn is next?',
          answers=self._player_names)
      result = self._player_names[idx]
      self._currently_active_player = result

    return result

  def get_currently_active_player(self) -> str | None:
    return self._currently_active_player


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

    self._currently_active_player = None

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
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f'{prefix}:\n{self._get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      scene_participants = self._get_current_scene_participants()

      idx = prompt.multiple_choice_question(
          question='Whose turn is next?',
          answers=scene_participants)
      result = scene_participants[idx]

      self._currently_active_player = result

    return result

  def get_currently_active_player(self) -> str | None:
    return self._currently_active_player


class NextActionSpec(entity_component.ContextComponent):
  """A component that decides whose turn is next.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      next_acting_component_name: str = DEFAULT_NEXT_ACTING_COMPONENT_NAME,
      pre_act_key: str = DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players to choose from.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      call_to_next_action_spec: prompt to use for the game master to decide on
        what action spec to use for the next turn. Will be formatted to
        substitute {name} for the name of the player whose turn is next.
      next_acting_component_name: The name of the NextActing component to use
        to get the name of the player whose turn is next.
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
    self._call_to_next_action_spec = call_to_next_action_spec
    self._next_acting_component_name = next_acting_component_name
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
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f'{prefix}:\n{self._get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      active_player = self.get_entity().get_component(
          self._next_acting_component_name, type_=NextActing
      ).get_currently_active_player()
      prompt.statement(
          'Example formatted action specs:\n1). "prompt: p;;type: free"\n'
          '2). "prompt: p;;type: choice;;options: x, y, z".\nNote that p is a '
          'string of any length, typically a question, and x, y, z, etc are '
          'multiple choice answer responses. For instance, a valid format '
          'could be indicated as '
          'prompt: Where will Edgar go?;;type: choice;;'
          'options: home, London, Narnia, the third moon of Jupiter')
      result = prompt.open_question(
          question=self._call_to_next_action_spec.format(name=active_player),
          max_tokens=512)

    return result


class NextActionSpecFromSceneSpec(entity_component.ContextComponent):
  """A component that decides the next action spec using the current scene spec.
  """

  def __init__(
      self,
      scenes: Sequence[scene_lib.ExperimentalSceneSpec],
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      pre_act_key: str = DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      scenes: All scenes to be used in the episode.
      memory_component_name: The name of the memory component.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._memory_component_name = memory_component_name
    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

    # Extract all scene type specs from the provided scenes.
    self._scene_type_specs = {}
    for scene in scenes:
      self._scene_type_specs[scene.scene_type.name] = scene.scene_type

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def _get_current_scene_type(self) -> scene_lib.ExperimentalSceneTypeSpec:
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.Memory
    )
    scene_type_str = scene_runner.get_current_scene_type(memory=memory)
    scene_type_spec = self._scene_type_specs[scene_type_str]
    return scene_type_spec

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    action_spec_string = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      scene_type_spec = self._get_current_scene_type()
      action_spec = scene_type_spec.action_spec
      action_spec_string = engine_lib.action_spec_to_string(action_spec)

    return action_spec_string

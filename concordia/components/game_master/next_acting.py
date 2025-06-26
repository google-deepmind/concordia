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

from collections.abc import Sequence, Mapping
import random

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import scene_tracker as scene_tracker_component
from concordia.document import interactive_document
from concordia.environment import engine as engine_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import scene as scene_lib


DEFAULT_NEXT_ACTING_COMPONENT_KEY = '__next_acting__'
# Initiative is the Dungeons & Dragons term for the rule system that controls
# turn taking.
DEFAULT_NEXT_ACTING_PRE_ACT_LABEL = '\nInitiative'
DEFAULT_CALL_TO_NEXT_ACTING = 'Who is next to act?'

DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY = '__next_action_spec__'
DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_LABEL = '\nType of action'
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    'In what action spec format should {name} respond? Respond in  '
    'one of the provided formats and use no additional words.')


class NextActing(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides whose turn is next.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Sequence[str] = (),
      pre_act_label: str = DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players to choose from.
      components: Keys of components to condition the answer on.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._player_names = player_names
    self._components = components
    self._pre_act_label = pre_act_label

    self._currently_active_player = None

    if not self._player_names:
      raise ValueError('No player names provided.')

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
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      idx = prompt.multiple_choice_question(
          question='Whose turn is next?',
          answers=self._player_names)
      result = self._player_names[idx]
      self._currently_active_player = result

    return result

  def get_currently_active_player(self) -> str | None:
    return self._currently_active_player

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'currently_active_player': self._currently_active_player,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._currently_active_player = state['currently_active_player']


class NextActingAllEntities(entity_component.ContextComponent):
  """A component that always selects all entities to act next for async environments."""

  def __init__(
      self,
      player_names: Sequence[str],
      pre_act_label: str = DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      player_names: Names of players.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    super().__init__()
    self._player_names = player_names
    self._pre_act_label = pre_act_label

    if not self._player_names:
      raise ValueError('No player names provided.')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      result = ','.join(self._player_names)
    return result

  def get_currently_active_player(self) -> str | None:
    """Not applicable for this component as all players are always active."""
    return None

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass


class NextActingInFixedOrder(entity_component.ContextComponent):
  """A component that decides whose turn is next in a fixed sequence.
  """

  def __init__(
      self,
      sequence: Sequence[str],
      pre_act_label: str = DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      sequence: Sequence of player names. The game master will select players
        to take turns in this order. The sequence will be cycled through.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._sequence = sequence

    self._pre_act_label = pre_act_label

    self._currently_active_player_idx = None

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      idx = self._currently_active_player_idx
      if idx is None:
        idx = 0
      idx = (idx + 1) % len(self._sequence)
      result = self._sequence[idx]
      self._currently_active_player_idx = idx

    return result

  def get_currently_active_player(self) -> str | None:
    if self._currently_active_player_idx is None:
      return None
    return self._sequence[self._currently_active_player_idx]

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'currently_active_player_idx': self._currently_active_player_idx,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._currently_active_player_idx = state['currently_active_player_idx']


class NextActingInRandomOrder(entity_component.ContextComponent):
  """A component that decides whose turn is next in a random sequence.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      replace: bool = False,
      pre_act_label: str = DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      player_names: Sequence of player names. The game master will select
        players out of this sequence randomly, either with or without
        replacement.
      replace: Whether to sample players with or without replacement.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._player_names = player_names
    self._replace = replace

    self._pre_act_label = pre_act_label

    self._currently_available_indices = list(range(len(self._player_names)))

    self._currently_active_player_idx = None

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      # If sampling without replacement then we need to reset the list of
      # indices after we have sampled all of them so we can start
      # sampling from the beginning again.
      if not self._replace and not self._currently_available_indices:
        self._currently_available_indices = list(range(len(self._player_names)))

      self._currently_active_player_idx = random.choice(
          self._currently_available_indices)
      result = self._player_names[self._currently_active_player_idx]
      if not self._replace:
        self._currently_available_indices.remove(
            self._currently_active_player_idx)

    return result

  def get_currently_active_player(self) -> str | None:
    if self._currently_active_player_idx is None:
      return None
    return self._player_names[self._currently_active_player_idx]

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'currently_active_player_idx': self._currently_active_player_idx,
        'currently_available_indices': self._currently_available_indices,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._currently_active_player_idx = state['currently_active_player_idx']
    self._currently_available_indices = state['currently_available_indices']


class NextActingFromSceneSpec(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides whose turn is next using the current scene spec.
  """

  def __init__(
      self,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      scene_tracker_component_key: str = (
          scene_tracker_component.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
      ),
      pre_act_label: str = DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      memory_component_key: The name of the memory component.
      scene_tracker_component_key: The name of the scene tracker component.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
    """
    super().__init__()
    self._memory_component_key = memory_component_key
    self._scene_tracker_component_key = scene_tracker_component_key
    self._pre_act_label = pre_act_label

    self._currently_active_player = None
    self._counter = 0

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def _get_current_scene_participants(self) -> Sequence[str]:

    scene_tracker = self.get_entity().get_component(
        self._scene_tracker_component_key,
        type_=scene_tracker_component.SceneTracker,
    )
    return scene_tracker.get_participants()

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      scene_participants = self._get_current_scene_participants()
      idx = self._counter % len(scene_participants)
      result = scene_participants[idx]
      self._counter += 1

      self._currently_active_player = result

    return result

  def get_currently_active_player(self) -> str | None:
    return self._currently_active_player

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'currently_active_player': self._currently_active_player,
        'counter': self._counter,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._currently_active_player = state['currently_active_player']
    self._counter = state['counter']


class NextActionSpec(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides whose turn is next.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Sequence[str] = (),
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      next_acting_component_key: str = DEFAULT_NEXT_ACTING_COMPONENT_KEY,
      pre_act_label: str = DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players to choose from.
      components: Keys of components to condition the action spec on.
      call_to_next_action_spec: prompt to use for the game master to decide on
        what action spec to use for the next turn. Will be formatted to
        substitute {name} for the name of the player whose turn is next.
      next_acting_component_key: The name of the NextActing component to use
        to get the name of the player whose turn is next.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._player_names = player_names
    self._components = components
    self._call_to_next_action_spec = call_to_next_action_spec
    self._next_acting_component_key = next_acting_component_key
    self._pre_act_label = pre_act_label

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
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      active_player = self.get_entity().get_component(
          self._next_acting_component_key, type_=NextActing
      ).get_currently_active_player()
      prompt.statement(
          'Example formatted action specs:\n'
          f'1). "prompt: what does {active_player} do?;;type: free"\n'
          f'2). "prompt: what does {active_player} say?;;type: free"\n'
          f'3). "prompt: Where will {active_player} go?;;type: choice;;'
          'options: home, London, Narnia, the third moon of Jupiter"\n'
          f'4). "prompt: Where will {active_player} go?;;type: choice;;'
          'options: stay here, go elsewhere"\n'
          'Note that prompts can be of any length, they are typically '
          'questions, and multiple choice answer responses must be '
          'provided in the form of a comma-separated list of options.')
      result = prompt.open_question(
          question=self._call_to_next_action_spec.format(name=active_player),
          max_tokens=1024,
          terminators=())

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass


class NextActionSpecFromSceneSpec(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides the next action spec using the current scene spec.
  """

  def __init__(
      self,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      scene_tracker_component_key: str = (
          scene_tracker_component.DEFAULT_SCENE_TRACKER_COMPONENT_KEY
      ),
      next_acting_component_key: str = (
          DEFAULT_NEXT_ACTING_COMPONENT_KEY
      ),
      pre_act_label: str = DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      memory_component_key: The name of the memory component.
      scene_tracker_component_key: The name of the scene tracker component.
      next_acting_component_key: The name of the NextActingFromSceneSpec
        component to use to get the name of the player whose turn is next.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._memory_component_key = memory_component_key
    self._scene_tracker_component_key = scene_tracker_component_key
    self._pre_act_label = pre_act_label
    self._next_acting_component_key = next_acting_component_key

  def _get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def _get_current_scene_type(self) -> scene_lib.SceneTypeSpec:

    scene_tracker = self.get_entity().get_component(
        self._scene_tracker_component_key,
        type_=scene_tracker_component.SceneTracker,
    )

    return scene_tracker.get_current_scene_type()

  def get_current_active_player(self) -> str | None:
    next_acting = self.get_entity().get_component(
        self._next_acting_component_key, type_=NextActingFromSceneSpec
    )
    return next_acting.get_currently_active_player()

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    action_spec_string = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      scene_type_spec = self._get_current_scene_type()
      action_spec = scene_type_spec.action_spec
      if isinstance(action_spec, Mapping):
        action_spec = action_spec.get(self.get_current_active_player())
      action_spec_string = engine_lib.action_spec_to_string(action_spec)
      self._logging_channel({'Action spec': action_spec_string,
                             'Scene type spec': scene_type_spec})
    return action_spec_string

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass


class FixedActionSpec(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that always returns the same action spec.
  """

  def __init__(
      self,
      action_spec: entity_lib.ActionSpec,
      pre_act_label: str = DEFAULT_NEXT_ACTION_SPEC_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      action_spec: The action spec to return whenever pre_act is called with
        output type NEXT_ACTION_SPEC.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._fixed_entity_action_spec = action_spec
    self._pre_act_label = pre_act_label

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    entity_action_spec_string = ''
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      entity_action_spec_string = engine_lib.action_spec_to_string(
          self._fixed_entity_action_spec)

    return entity_action_spec_string

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass

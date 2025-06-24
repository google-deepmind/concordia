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

from collections.abc import Callable, Sequence
from concordia.components.agent import memory as memory_component_module
from concordia.components.game_master import make_observation as make_observation_component_module
from concordia.components.game_master import next_game_master as next_game_master_component_module
from concordia.components.game_master import terminate as terminate_component_module
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import scene as scene_lib

_SCENE_COUNTER_TAG = '[scene counter]'

_SCENE_TYPE_TAG = '[scene type]'
_SCENE_PARTICIPANTS_TAG = '[scene participants]'
_PARTICIPANTS_DELIMITER = ', '

DEFAULT_SCENE_TRACKER_PRE_ACT_LABEL = '\nCurrent Scene'

_TERMINATE_SIGNAL = 'Yes'

DEFAULT_SCENE_TRACKER_COMPONENT_KEY = (
    next_game_master_component_module.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY)

DEFAULT_NEXT_GAME_MASTER_NAME = 'default_rules'


class SceneTracker(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that decides which game master to use next."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      scenes: Sequence[scene_lib.SceneSpec],
      observation_component_key: str = (
          make_observation_component_module.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      memory_component_key: str = (
          memory_component_module.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      terminator_component_key: str = (
          terminate_component_module.DEFAULT_TERMINATE_COMPONENT_KEY
      ),
      default_next_game_master_name: str = DEFAULT_NEXT_GAME_MASTER_NAME,
      pre_act_label: str = DEFAULT_SCENE_TRACKER_PRE_ACT_LABEL,
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      scenes: All scenes to be used in the episode.
      observation_component_key: The name of the observation component.
      memory_component_key: The name of the memory component.
      terminator_component_key: The name of the terminator component.
      default_next_game_master_name: The name of the next game master to use in
        cases where the scene does not specify a game master.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      verbose: Whether to print verbose debug information.
    """
    super().__init__()
    self._model = model
    self._pre_act_label = pre_act_label
    self._memory_component_key = memory_component_key
    self._observation_component_key = observation_component_key
    self._terminator_component_key = terminator_component_key
    self._default_next_game_master_name = default_next_game_master_name
    self._scenes = scenes
    self._verbose = verbose

    self._round_idx_to_scene = {}
    round_idx = 0
    for scene in self._scenes:
      for idx in range(scene.num_rounds):
        self._round_idx_to_scene[round_idx] = {'scene': scene,
                                               'step_within_scene': idx}
        round_idx += 1

    self._max_rounds = round_idx

  def _get_scene_step_and_scene(
      self,
  ) -> tuple[int, scene_lib.SceneSpec, int]:
    memory_component = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component_module.Memory
    )
    counter_states = memory_component.scan(
        lambda x: x.startswith(_SCENE_COUNTER_TAG))
    counter_state = len(counter_states)
    if counter_state == self._max_rounds:
      return -1, self._scenes[0], counter_state
    elif counter_state > self._max_rounds:
      raise RuntimeError(
          f'Counter state {counter_state} is greater than max number of rounds'
          f' {self._max_rounds}.'
      )
    step_within_scene = (
        self._round_idx_to_scene[counter_state]['step_within_scene'])
    scene = self._round_idx_to_scene[counter_state]['scene']
    return step_within_scene, scene, counter_state

  def is_done(self) -> bool:
    _, _, global_step = self._get_scene_step_and_scene()
    if global_step >= self._max_rounds:
      return True
    return False

  def get_current_scene_type(self) -> scene_lib.SceneTypeSpec:
    _, scene, _ = self._get_scene_step_and_scene()
    return scene.scene_type

  def get_participants(self) -> Sequence[str]:
    _, scene, _ = self._get_scene_step_and_scene()
    participants = scene.participants
    if scene.scene_type.possible_participants:
      participants = list(
          set(participants).intersection(scene.scene_type.possible_participants)
      )
    return participants

  def _get_premise(
      self, scene: scene_lib.SceneSpec, participant: str
  ) -> Sequence[str | Callable[[str], str]]:
    if scene.premise is None:
      premises = scene.scene_type.default_premise[participant]
    else:
      premises = scene.premise[participant]

    result = []
    for premise in premises:
      if isinstance(premise, str):
        assert isinstance(premise, str), type(premise)  # For pytype.
        result.append(premise)
      else:
        assert isinstance(premise, Callable), type(premise)  # For pytype.
        evaluated_premise = premise(participant)
        result.append(evaluated_premise)

    return result

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component_module.Memory
      )
      step_within_scene, current_scene, _ = self._get_scene_step_and_scene()

      if self._verbose:
        print(f'Scene game master: {current_scene.scene_type.game_master_name}')
        print(f'Step counter: {step_within_scene}')

      if self.is_done():
        terminator = self.get_entity().get_component(
            self._terminator_component_key,
            type_=terminate_component_module.Terminate,
        )
        terminator.terminate()
        self._logging_channel({
            'Summary': 'Terminating the simulation.',
        })
        return _TERMINATE_SIGNAL

      if step_within_scene == 0:
        make_observation = self.get_entity().get_component(
            self._observation_component_key,
            type_=make_observation_component_module.MakeObservation,
        )
        memory.add(f'{_SCENE_TYPE_TAG} {current_scene.scene_type.name}')
        memory.add(
            f'{_SCENE_PARTICIPANTS_TAG} {", ".join(self.get_participants())}'
        )
        for participant in self.get_participants():
          for observation in self._get_premise(
              scene=current_scene,
              participant=participant):
            make_observation.add_to_queue(participant, observation)
            memory.add(f'{participant} observed the following: {observation}')

    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      _, next_scene, _ = self._get_scene_step_and_scene()
      if next_scene.scene_type.game_master_name is None:
        return self._default_next_game_master_name
      return next_scene.scene_type.game_master_name

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      step_within_scene, current_scene, global_step = (
          self._get_scene_step_and_scene()
      )

      self._logging_channel({
          'Current scene': current_scene,
          'Scene type': current_scene.scene_type,
          'Summary': f'Scene: {current_scene.scene_type.name}',
          'Step within scene': step_within_scene,
          'Global step': global_step,
          'Scene participants': ', '.join(self.get_participants()),
      })

      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component_module.Memory
      )
      global_step += 1
      memory.add(f'{_SCENE_COUNTER_TAG}({global_step})')

    return ''

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'round_idx_to_scene': self._round_idx_to_scene,
        'max_rounds': self._max_rounds,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._round_idx_to_scene = state['round_idx_to_scene']
    self._max_rounds = state['max_rounds']

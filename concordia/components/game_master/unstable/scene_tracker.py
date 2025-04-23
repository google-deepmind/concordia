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

from collections.abc import Sequence
import threading
from concordia.components.agent.unstable import memory as memory_component
from concordia.components.game_master.unstable import make_observation as make_observation_component
from concordia.language_model import language_model
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import entity_component
from concordia.typing.unstable import scene as scene_lib

_SCENE_TYPE_TAG = '[scene type]'
_SCENE_PARTICIPANTS_TAG = '[scene participants]'
_PARTICIPANTS_DELIMITER = ', '

DEFAULT_SCENE_TRACKER_COMPONENT_KEY = '__scene_tracker__'

DEFAULT_SCENE_TRACKER_PRE_ACT_LABEL = '\nCurrent Scene'


class ThreadSafeCounter:
  """A thread-safe counter that can only be incremented."""

  def __init__(self, initial_value=0):
    """Initializes the counter with an optional initial value.

    Args:
        initial_value (int): The initial value of the counter. Defaults to 0.
    """
    self._value = initial_value
    self._lock = threading.Lock()

  def increment(self, amount=1):
    """Increments the counter by the specified amount.

    Args:
        amount (int): The amount to increment the counter by. Defaults to 1.
    """
    with self._lock:
      self._value += amount

  def value(self):
    """Returns the current value of the counter.

    Returns:
        int: The current value of the counter.
    """
    with self._lock:
      return self._value


class SceneTracker(entity_component.ContextComponent,
                   entity_component.ComponentWithLogging):
  """A component that decides which game master to use next."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      scenes: Sequence[scene_lib.ExperimentalSceneSpec],
      step_counter: ThreadSafeCounter,
      observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = DEFAULT_SCENE_TRACKER_PRE_ACT_LABEL,
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      scenes: All scenes to be used in the episode.
      step_counter: The counter to use for the step within the scene.
      observation_component_key: The name of the observation component.
      memory_component_key: The name of the memory component.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      verbose: Whether to print verbose debug information.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._pre_act_label = pre_act_label
    self._memory_component_key = memory_component_key

    self._observation_component_key = observation_component_key
    self._step_counter = step_counter
    self._scenes = scenes
    self._verbose = verbose

  def _get_scene_step_and_scene(
      self,
  ) -> tuple[int, scene_lib.ExperimentalSceneSpec]:
    counter = self._step_counter.value()
    for scene in self._scenes:
      if counter < scene.num_rounds:
        return (counter, scene)
      else:
        counter -= scene.num_rounds
    raise ValueError(
        f'No scene found for global step {self._step_counter.value()}. '
        'Probably the simulation is not terminated after all scenes are done.'
    )

  def get_current_scene_type(self) -> scene_lib.ExperimentalSceneTypeSpec:
    _, scene = self._get_scene_step_and_scene()
    return scene.scene_type

  def get_participants(self) -> Sequence[str]:
    _, scene = self._get_scene_step_and_scene()
    participants = scene.participants
    if scene.scene_type.possible_participants:
      participants = list(
          set(participants).intersection(scene.scene_type.possible_participants)
      )
    return participants

  def get_observation_for_participant(self, participant: str) -> str:
    scene_step, scene = self._get_scene_step_and_scene()
    if scene_step == 0:
      return '\n'.join(scene.scene_type.premise[participant])
    return ''

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    step_within_scene, current_scene = self._get_scene_step_and_scene()

    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      if self._verbose:
        print('Next game master pre-act')
        print(f'Scene game master: {current_scene.scene_type.game_master_name}')
        print(f'Step counter: {step_within_scene}')
      if step_within_scene == 0:

        make_observation = self.get_entity().get_component(
            self._observation_component_key,
            type_=make_observation_component.MakeObservation,
        )

        print(f'Starting the scene: {current_scene.scene_type.name}')
        memory.add(f'{_SCENE_TYPE_TAG} {current_scene.scene_type.name}')
        memory.add(
            f'{_SCENE_PARTICIPANTS_TAG} {", ".join(self.get_participants())}'
        )
        print(
            'Adding to memory:'
            f' {_SCENE_TYPE_TAG} {current_scene.scene_type.name}'
        )
        print(
            'Adding to memory:'
            f' {_SCENE_PARTICIPANTS_TAG} {", ".join(self.get_participants())}'
        )
        for participant in self.get_participants():
          if 'game_master' in participant:
            for observation in current_scene.scene_type.premise[participant]:
              memory.add(observation)
          else:
            for observation in current_scene.scene_type.premise[participant]:
              make_observation.add_to_queue(participant, observation)

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      print('Resolve')
      print(f'current scene: {current_scene.scene_type.name}')
      print(f'step counter: {step_within_scene}')
      self._logging_channel({
          'Summary': f'Scene: {current_scene.scene_type.name}',
          'Step within scene': step_within_scene,
          'Global step': self._step_counter.value(),
          'Scene participants': ', '.join(self.get_participants()),
      })
      self._step_counter.increment(amount=1)

    return result

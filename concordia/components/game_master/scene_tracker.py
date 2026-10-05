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

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import json
from typing import Any, NoReturn

from absl import logging
from concordia.components.agent import memory as memory_component_module
from concordia.components.game_master import make_observation as make_observation_component_module
from concordia.components.game_master import next_game_master as next_game_master_component_module
from concordia.components.game_master import terminate as terminate_component_module
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import scene as scene_lib

_SCENE_COUNTER_TAG = '[scene counter]'
_SCENE_PREMISE_QUEUED_TAG = '[scene premise queued]'

_SCENE_TYPE_TAG = '[scene type]'
_SCENE_PARTICIPANTS_TAG = '[scene participants]'
_PARTICIPANTS_DELIMITER = ', '

DEFAULT_SCENE_TRACKER_PRE_ACT_LABEL = '\nCurrent Scene'

_TERMINATE_SIGNAL = 'Yes'

DEFAULT_SCENE_TRACKER_COMPONENT_KEY = (
    next_game_master_component_module.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
)

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
        self._round_idx_to_scene[round_idx] = {
            'scene': scene,
            'step_within_scene': idx,
        }
        round_idx += 1

    self._max_rounds = round_idx

  def _get_scene_counter(self) -> int:
    memory_component = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component_module.Memory
    )
    counter_states = memory_component.scan(
        lambda x: x.startswith(_SCENE_COUNTER_TAG)
    )
    return len(counter_states)

  def _get_scene_step_and_scene(
      self,
  ) -> tuple[int, scene_lib.SceneSpec, int]:
    counter_state = self._get_scene_counter()
    if counter_state == self._max_rounds:
      if self._scenes:
        scene = self._scenes[0]
      else:
        scene = scene_lib.SceneSpec(
            scene_type=scene_lib.SceneTypeSpec(name=''),
            participants=(),
            num_rounds=0,
        )
      return -1, scene, counter_state
    elif counter_state > self._max_rounds:
      raise RuntimeError(
          f'Counter state {counter_state} is greater than max number of rounds'
          f' {self._max_rounds}.'
      )
    step_within_scene = self._round_idx_to_scene[counter_state][
        'step_within_scene'
    ]
    scene = self._round_idx_to_scene[counter_state]['scene']
    return step_within_scene, scene, counter_state

  def is_done(self) -> bool:
    global_step = self._get_scene_counter()
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
      premises = scene.scene_type.default_premise[participant]  # pyrefly: ignore[unsupported-operation]
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

  def _maybe_queue_scene_start_premises(self) -> None:
    """Queue scene start premises if at step 0 and not already queued.

    This method uses a memory marker to track which scenes have had their
    premises queued. This ensures premises are queued exactly once per scene,
    on the first action we see at step 0, regardless of action type.

    This handles the case where an initializer GM runs first without a
    SceneTracker, causing RESOLVE to run before TERMINATE. By checking on
    any action type, we queue premises as soon as the main GM sees step 0.
    """
    if self.is_done():
      return
    step_within_scene, current_scene, global_step = (
        self._get_scene_step_and_scene()
    )

    # Only queue premises at the start of a scene (step 0)
    if step_within_scene != 0:
      return

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component_module.Memory
    )

    # Check if we've already queued premises for this scene
    marker = f'{_SCENE_PREMISE_QUEUED_TAG}({global_step})'
    existing_markers = memory.scan(lambda x: x == marker)
    if existing_markers:
      return

    # Mark this scene as having its premises queued
    memory.add(marker)

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
          scene=current_scene, participant=participant
      ):
        make_observation.add_to_queue(participant, observation)  # pyrefly: ignore[bad-argument-type]
        memory.add(f'{participant} observed the following: {observation}')

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    # Queue scene start premises on ANY action at step 0.
    # This handles the case where an initializer GM runs first, causing
    # RESOLVE to execute before TERMINATE on the main GM.
    self._maybe_queue_scene_start_premises()

    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      step_within_scene, current_scene, _ = self._get_scene_step_and_scene()

      if self._verbose:
        logging.info(
            'Scene game master: %s', current_scene.scene_type.game_master_name
        )
        logging.info('Step counter: %s', step_within_scene)

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

    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      if self.is_done():
        return self._default_next_game_master_name
      _, next_scene, _ = self._get_scene_step_and_scene()
      if next_scene.scene_type.game_master_name is None:
        return self._default_next_game_master_name
      return next_scene.scene_type.game_master_name

    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      if self.is_done():
        return ''
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
    """Return editable configuration; progress remains in the memory component.

    Callable premises cannot be reconstructed from JSON. Such configurations
    retain their existing checkpoint contract and expose no editable fields.
    """
    return self.get_dynamic_state()

  def get_dynamic_state(self) -> entity_component.ComponentState:
    """Expose literal scene configuration through the ordinary component API."""

    def encode(value):
      if isinstance(value, entity_lib.ActionSpec):
        return {'action_spec': value.to_dict()}
      if isinstance(value, datetime.datetime):
        return {'datetime': value.isoformat()}
      if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: encode(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
      if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
          raise ValueError('State keys must be strings')
        return {key: encode(item) for key, item in value.items()}
      if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
      if value is None or type(value) in (str, bool, int, float):
        return value
      raise ValueError('Configuration contains non-literal state')

    try:
      result = {'scenes': encode(self._scenes)}
      json.dumps(result, allow_nan=False)
      self._parse_scenes(result['scenes'])
      return result
    except (TypeError, ValueError):
      return {}

  def _parse_scenes(
      self,
      records: Any,
      previous: Sequence[Mapping[str, Any]] = (),
  ) -> tuple[list[scene_lib.SceneSpec], dict[int, dict[str, Any]]]:
    """Validate and decode serialized scene records."""

    def fail(path, message) -> NoReturn:
      raise ValueError(path + ': ' + message)

    def keys(value, allowed, path):
      if not isinstance(value, dict) or set(value) - set(allowed):
        fail(path, 'expected object with supported fields')

    def names(value, path, nullable=False):
      if value is None and nullable:
        return None
      if (
          not isinstance(value, list)
          or any(not isinstance(x, str) or not x.strip() for x in value)
          or len(set(value)) != len(value)
      ):
        fail(path, 'expected distinct nonempty names')
      return list(value)

    def premise(value, path):
      if value is None:
        return None
      if not isinstance(value, dict) or any(
          not isinstance(k, str) for k in value
      ):
        fail(path, 'expected names mapped to lists of literal observations')
      for name, observations in value.items():
        if not isinstance(observations, list) or any(
            not isinstance(x, str) for x in observations
        ):
          fail(path + '.' + name, 'expected list of text observations')
      return {k: list(v) for k, v in value.items()}

    def decode_spec(value, path):
      if (
          not isinstance(value, dict)
          or set(value) != {'action_spec'}
          or not isinstance(value['action_spec'], dict)
      ):
        fail(path, 'expected one action specification')
      payload = value['action_spec']
      if (
          not isinstance(payload.get('call_to_action'), str)
          or not isinstance(payload.get('output_type'), str)
          or (
              payload.get('tag') is not None
              and not isinstance(payload['tag'], str)
          )
          or not isinstance(payload.get('options', []), list)
          or any(
              not isinstance(option, str)
              for option in payload.get('options', [])
          )
      ):
        fail(path, 'invalid action specification fields')
      try:
        return entity_lib.action_spec_from_dict(payload)
      except (ValueError, TypeError, KeyError):
        fail(path, 'invalid action specification')

    def action(value, path):
      if value is None:
        return None
      if not isinstance(value, dict):
        fail(path, 'expected action specification or name mapping')
      if (
          set(value) == {'action_spec'}
          and isinstance(value['action_spec'], dict)
          and 'output_type' in value['action_spec']
      ):
        return decode_spec(value, path)
      result = {}
      for name, spec in value.items():
        if not isinstance(name, str) or not name.strip():
          fail(path, 'expected nonempty entity names')
        result[name] = decode_spec(spec, path + '.' + name)
      return result

    if not isinstance(records, list) or len(records) > 1000:
      fail('scenes', 'expected 0–1000 records')
    scenes = []
    schedule = {}
    for index, record in enumerate(records):
      path = f'scenes[{index}]'
      keys(
          record,
          ('scene_type', 'participants', 'num_rounds', 'start_time', 'premise'),
          path,
      )
      # A partial record preserves fields omitted by an editor. The list is
      # still an explicit ordered replacement, so records can be removed.
      if index < len(previous):
        baseline = previous[index]
        record = {**baseline, **record}
        if isinstance(record.get('scene_type'), dict):
          record['scene_type'] = {
              **baseline['scene_type'],
              **record['scene_type'],
          }
      kind = record.get('scene_type')
      keys(
          kind,
          (
              'name',
              'game_master_name',
              'default_premise',
              'action_spec',
              'possible_participants',
          ),
          path + '.scene_type',
      )
      if not isinstance(kind.get('name'), str) or not kind['name'].strip():
        fail(path + '.scene_type.name', 'expected nonempty text')
      gm = kind.get('game_master_name')
      if gm is not None and (not isinstance(gm, str) or not gm.strip()):
        fail(path + '.scene_type.game_master_name', 'expected name or null')
      rounds = record.get('num_rounds')
      if (
          not isinstance(rounds, int)
          or isinstance(rounds, bool)
          or not 1 <= rounds <= 100000
      ):
        fail(path + '.num_rounds', 'expected integer from 1 to 100000')
      participants = names(record.get('participants'), path + '.participants')
      if not participants:
        fail(path + '.participants', 'expected at least one participant')
      default = premise(
          kind.get('default_premise'), path + '.scene_type.default_premise'
      )
      override = premise(record.get('premise'), path + '.premise')
      active_premise = override if override is not None else default
      allowed = names(
          kind.get('possible_participants'),
          path + '.scene_type.possible_participants',
          True,
      )
      effective = [
          name for name in participants if not allowed or name in allowed
      ]
      if active_premise is None or any(
          name not in active_premise for name in effective
      ):
        fail(
            path + '.premise',
            'every active participant requires observations (an empty list is'
            ' valid)',
        )
      start = record.get('start_time')
      if start is not None:
        try:
          if not isinstance(start, dict) or set(start) != {'datetime'}:
            raise ValueError()
          start = datetime.datetime.fromisoformat(start['datetime'])
        except (TypeError, ValueError):
          fail(path + '.start_time', 'expected ISO datetime object or null')
      scene = scene_lib.SceneSpec(
          scene_type=scene_lib.SceneTypeSpec(
              name=kind['name'],
              game_master_name=gm,
              default_premise=default,
              action_spec=action(
                  kind.get('action_spec'), path + '.scene_type.action_spec'
              ),
              possible_participants=allowed,
          ),
          participants=participants,
          num_rounds=rounds,
          start_time=start,
          premise=override,
      )
      scenes.append(scene)
      if len(schedule) + rounds > 100000:
        fail('scenes', 'combined round count exceeds 100000')
      for step in range(rounds):
        schedule[len(schedule)] = {'scene': scene, 'step_within_scene': step}
    return scenes, schedule

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Validate and replace configuration atomically without resetting progress.

    Omitted fields are preserved. An empty checkpoint leaves configuration
    unchanged. Errors name the offending field without including arbitrary
    supplied state values.

    Args:
      state: Component state dictionary containing scene definitions.
    """
    if not state:
      return
    if set(state) != {'scenes'}:
      raise ValueError('state: expected only scenes')
    current = self.get_dynamic_state()
    if not current:
      raise ValueError(
          'scenes: callable or unsupported configuration is not editable'
      )
    previous: Any = current['scenes']
    scenes, schedule = self._parse_scenes(state['scenes'], previous)
    # The scheduling cursor is stored in memory, not in this configuration.
    if self._entity is not None:
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component_module.Memory
      )
      counter = len(memory.scan(lambda x: x.startswith(_SCENE_COUNTER_TAG)))
      if counter > len(schedule):
        raise ValueError('scenes: new schedule ends before current progress')
    self._scenes = scenes
    self._round_idx_to_scene = schedule
    self._max_rounds = len(schedule)

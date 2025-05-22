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

"""Scene runner."""

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.agents.deprecated import entity_agent
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.typing.deprecated import clock as game_clock
from concordia.typing.deprecated import logging as logging_lib
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import json as json_lib

_SCENE_TYPE_TAG = '[scene type]'
_SCENE_PARTICIPANTS_TAG = '[scene participants]'
_PARTICIPANTS_DELIMITER = ', '


def _get_interscene_messages(
    key: str,
    agent_name: str,
    scene_type_spec: scene_lib.SceneTypeSpec,
) -> list[str]:
  """Get messages to for both players and game master before and after a scene.

  Args:
    key: either get the scene's `premise` or its `conclusion` messages. Each
      message should be a string.
    agent_name: get interscene messages for which agent
    scene_type_spec: configuration of the scene

  Returns:
    messages: a list of strings to report.
  """
  if key == 'premise':
    section = scene_type_spec.premise if scene_type_spec.premise else {}
  elif key == 'conclusion':
    section = scene_type_spec.conclusion if scene_type_spec.conclusion else {}
  else:
    raise ValueError(f'Unknown key: {key}')
  raw_messages = section.get(agent_name, [])

  messages = []
  for raw_message in raw_messages:
    if isinstance(raw_message, str):
      result = raw_message
      messages.append(result)

  return messages


def run_scenes(
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    players: Sequence[entity_agent.EntityAgent],
    clock: game_clock.GameClock,
    verbose: bool = False,
    compute_metrics: Mapping[str, logging_lib.Metric] | None = None,
    log: list[Mapping[str, Any]] | None = None,
) -> None:
  """Run a sequence of scenes.

  Args:
    scenes: sequence of scene configurations
    players: full list of players (a subset may participate in each scene)
    clock: the game clock which may be advanced between scenes
    verbose: if true then print intermediate outputs
    compute_metrics: Optionally, a function to compute metrics.
    log: Optionally, a log to append debug information to.
  """
  players_by_name = {player.name: player for player in players}
  if len(players_by_name) != len(players):
    raise ValueError('Duplicate player names')

  for scene_idx, scene in enumerate(scenes):
    scene_simulation = scene.scene_type.engine
    game_master = scene.scene_type.game_master

    scene_type_spec = scene.scene_type
    possible_participants = scene_type_spec.possible_participants
    participants_from_scene = scene.participants
    if possible_participants is None:
      participant_names = participants_from_scene
    else:
      if participants_from_scene:
        participant_names = list(set(possible_participants).intersection(
            participants_from_scene
        ))
      else:
        participant_names = possible_participants

    participants_str = _PARTICIPANTS_DELIMITER.join(participant_names)
    participants = [players_by_name[name] for name in participant_names]

    if verbose:
      print(f'\n\n    Scene {scene_idx}    Participants: {participants_str}\n')

    # Prepare to run the scene
    clock.set(scene.start_time)

    all_premises = ''
    for participant in participants:
      premise_messages = _get_interscene_messages(
          key='premise',
          agent_name=participant.name,
          scene_type_spec=scene.scene_type,
      )
      for message in premise_messages:
        all_premises += f'{participant.name} -- premise: {message}      \n'
        if verbose:
          print(f'{participant.name} -- premise: {message}')
        participant.observe(message)
        game_master.observe(message)

    # Run the scene
    for _ in range(scene.num_rounds):
      game_master.observe(f'{_SCENE_TYPE_TAG} {scene.scene_type.name}')
      game_master.observe(f'{_SCENE_PARTICIPANTS_TAG} {participants_str}')
      # run_loop modifies log in place by appending to it
      scene_simulation.run_loop(
          game_masters=[game_master],
          entities=participants,
          max_steps=scene.num_rounds * len(participants),
          verbose=verbose,
          log=log,
      )

    # Conclude the scene
    for participant in participants:
      conclusion_messages = _get_interscene_messages(
          key='conclusion',
          agent_name=participant.name,
          scene_type_spec=scene.scene_type,
      )
      for message in conclusion_messages:
        if verbose:
          print(f'{participant.name} -- conclusion: {message}')
        participant.observe(message)
        game_master.observe(message)

    # Branch off a metric scene if applicable
    if scene.scene_type.save_after_each_scene:
      serialized_agents = {}
      for participant in participants:
        serialized_agents = {}
        json_representation = json_lib.save_to_json(participant)
        serialized_agents[participant.name] = json_representation

      if compute_metrics is not None:
        compute_metrics(serialized_agents)


def _get_latest_memory_item(
    memory: memory_component.Memory,
    tag: str,
) -> str:
  """Return the latest item prefixed by a tag in the memory."""
  retrieved = memory.scan(
      selector_fn=lambda x: x.startswith(tag),
  )
  if retrieved:
    result = retrieved[-1]
    return result[result.find(tag) + len(tag) + 1:]
  else:
    return ''


def get_current_scene_type(memory: memory_component.Memory) -> str:
  """Return the latest item prefixed by '[scene type]' in the memory."""
  tag = f'{observation_component.OBSERVATION_TAG} {_SCENE_TYPE_TAG}'
  return _get_latest_memory_item(memory, tag=tag)


def get_current_scene_participants(
    memory: memory_component.Memory) -> Sequence[str]:
  """Return the latest item prefixed by '[scene participants]' in the memory."""
  tag = f'{observation_component.OBSERVATION_TAG} {_SCENE_PARTICIPANTS_TAG}'
  participants_str = _get_latest_memory_item(memory, tag=tag)
  return participants_str.split(_PARTICIPANTS_DELIMITER)

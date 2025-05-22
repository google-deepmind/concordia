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

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.environment.deprecated import game_master
from concordia.typing.deprecated import clock as game_clock
from concordia.typing.deprecated import logging as logging_lib
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import json as json_lib


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
    environment: game_master.GameMaster,
    scenes: Sequence[scene_lib.SceneSpec],
    players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
    clock: game_clock.GameClock,
    verbose: bool = False,
    compute_metrics: Mapping[str, logging_lib.Metric] | None = None,
) -> None:
  """Run a sequence of scenes.

  Args:
    environment: the game master
    scenes: sequence of scene configurations
    players: full list of players (a subset may participate in each scene)
    clock: the game clock which may be advanced between scenes
    verbose: if true then print intermediate outputs
    compute_metrics: Optionally, a function to compute metrics.
  """
  players_by_name = {player.name: player for player in players}
  if len(players_by_name) != len(players):
    raise ValueError('Duplicate player names')

  for scene_idx, scene in enumerate(scenes):
    if scene.scene_type.override_game_master:
      this_scene_environment = scene.scene_type.override_game_master
      this_scene_game_master_memory = this_scene_environment.get_memory()
    else:
      this_scene_environment = environment
      this_scene_game_master_memory = environment.get_memory()

    participant_names = [config.name for config in scene.participant_configs]
    if verbose:
      print(f'\n\n    Scene {scene_idx}    Participants: {participant_names}\n')
    participants = [players_by_name[name] for name in participant_names]

    # Prepare to run the scene
    clock.set(scene.start_time)

    all_premises = ''
    for participant in participants:
      premise_messages = _get_interscene_messages(
          key='premise',
          agent_name=participant.name,  # pytype: disable=attribute-error
          scene_type_spec=scene.scene_type,
      )
      for message in premise_messages:
        all_premises += f'{participant.name} -- premise: {message}      \n'  # pytype: disable=attribute-error
        if verbose:
          print(f'{participant.name} -- premise: {message}')  # pytype: disable=attribute-error
        participant.observe(message)  # pytype: disable=attribute-error
        this_scene_game_master_memory.add(message)

    # Add the scene and its premise to the history
    scene_update_log_entry = game_master.LogEntry(
        date=clock.now(),
        event_statement=all_premises,
        summary=f'Scene {scene_idx} --- Participants: {participant_names}',
    )
    environment.insert_history(log_entry=scene_update_log_entry)

    # Run the scene
    for _ in range(scene.num_rounds):
      this_scene_game_master_memory.add(f'[scene type] {scene.scene_type.name}')
      this_scene_environment.step(
          active_players=participants,
          action_spec_override=scene.scene_type.action_spec,
      )
      if this_scene_environment.terminate_episode():
        break

    # Conclude the scene
    for participant in participants:
      conclusion_messages = _get_interscene_messages(
          key='conclusion',
          agent_name=participant.name,  # pytype: disable=attribute-error
          scene_type_spec=scene.scene_type,
      )
      for message in conclusion_messages:
        if verbose:
          print(f'{participant.name} -- conclusion: {message}')  # pytype: disable=attribute-error
        participant.observe(message)  # pytype: disable=attribute-error
        this_scene_game_master_memory.add(message)

    # Branch off a metric scene if applicable
    if scene.scene_type.save_after_each_scene:
      serialized_agents = {}
      for participant in participants:
        serialized_agents = {}
        json_representation = json_lib.save_to_json(participant)
        serialized_agents[participant.name] = json_representation  # pytype: disable=attribute-error

      if compute_metrics is not None:
        compute_metrics(serialized_agents)

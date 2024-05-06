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

"""Dataclasses used to structure simulations using scenes."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime

from concordia.associative_memory import formative_memories
from concordia.environment import game_master
from concordia.typing import agent as agent_lib


@dataclasses.dataclass(frozen=True)
class SceneTypeSpec:
  """A specification for a type of scene.

  Attributes:
    name: name of this type of scene.
    premise: map player names to messages they receive before the scene.
      Messages may be either literal strings or functions that return strings.
    conclusion: map player names to messages they receive after the scene.
      Messages may be either literal strings or functions mapping player names
      to strings.
    action_spec: optionally specify an action spec other than the default for
      the game master to ask the agents to produce during steps of this scene.
    override_game_master: optionally specify a game master to use instead of the
      default one.
  """
  name: str
  premise: Mapping[str, Sequence[str | Callable[[str], str]]] | None = None
  conclusion: Mapping[str,
                      Sequence[str | Callable[[str], str]]] | None = None
  action_spec: agent_lib.ActionSpec | None = None
  override_game_master: game_master.GameMaster | None = None


@dataclasses.dataclass(frozen=True)
class SceneSpec:
  """Specify a specific scene.

  Attributes:
    scene_type: Select a specific type of scene.
    start_time: Automatically advance the clock to this time when the scene
      starts.
    participant_configs: Which players participate in the scene.
    num_rounds: How many rounds the scene lasts.
  """
  scene_type: SceneTypeSpec
  start_time: datetime.datetime
  participant_configs: Sequence[formative_memories.AgentConfig]
  num_rounds: int

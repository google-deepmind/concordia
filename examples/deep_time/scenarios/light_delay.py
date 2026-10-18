# Copyright 2026 DeepMind Technologies Limited.
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

"""Light Delay: terraform a colony light-years from home.

Colonists on a cold world must keep themselves fed and breathing while they
warm the sky until it is habitable. Home is years away at the speed of
light: what it learns of the colony is years old, and what it sends arrives
years later.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import colony as colony_prefab
from examples.deep_time.prefabs import mission
from concordia.typing import prefab as prefab_lib

NAME = 'light_delay'


def build_instances(
    context: game_lib.GameContext,
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  return [
      mission.mission_config(context),
      colony_prefab.colony_config(context),
      chronicler.chronicler_config(context),
  ]

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

"""Snowball, a geologic-era scenario.

A frozen ocean world: ice reflects so much sunlight that the cold sustains
itself (the ice-albedo trap). With no ice-free rock, silicate weathering
stalls while volcanoes keep outgassing, so CO2 builds until the equator cracks
open and the planet snaps to its warm state. The Steward can hurry the thaw
by rousing volcanoes or darkening the ice; the Chronicler records the
deglaciation. Geologic-era scenarios have no species actors.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'snowball'


def build_instances(
    context: game_lib.GameContext,
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  return [
      steward.steward_config(context, name='Gaia'),
      chronicler.chronicler_config(context),
  ]

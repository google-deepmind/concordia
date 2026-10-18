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

"""Great Migration: herds on the move as the world warms.

The sun strengthens through the scenario, so the cool grassland the bison
need (`world/herds.py`: they graze their tile and move toward food and a
climate that suits them) creeps poleward and thins at the equator; the
wolves follow the herds. The Steward slows the warming with bright clouds,
sows grass where the herds must go and thins the packs that press them too
hard (both aimable); the goal counts the animals themselves.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'great_migration'
BIOMES = ('grass', 'forests')


def build_instances(
    context: game_lib.GameContext,
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  return (
      [steward.steward_config(context, name='Gaia')]
      + [species_prefab.species_config(context, key) for key in BIOMES]
      + [chronicler.chronicler_config(context)]
  )

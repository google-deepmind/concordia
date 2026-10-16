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

"""Savanna Fire: keep the woods through the lightning season.

A hot, dry world where lightning strikes ten times as often as elsewhere
(`world/fire.py`): fires run through dry country thick with grass and wood,
burn the woods back to grass and spread their scars. The Steward summons
rain over the country she chooses (soaked ground does not burn that season,
and burns less as it dries over the next few) and plants woods again; bison
graze, wolves hunt, geese pass. The
goal is the woods themselves, against how much land they held at the start.
"""

from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'savanna_fire'
BIOMES = ('grass', 'forests')


def build_instances(
    context: game_lib.GameContext, models: model_setup.Models
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  return (
      [steward.steward_config(context, models.slow, name='Gaia')]
      + [
          species_prefab.species_config(context, key, models.slow)
          for key in BIOMES
      ]
      + [chronicler.chronicler_config(context, models.slow)]
  )

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

"""Archipelago: green the islands, one sea crossing at a time.

Grass and forest start on the largest of a sea of islands. They spread over
an island tile by tile (`world/tilelife.py`) but cannot cross the sea alone:
migrating geese sow the seeds of what they ate where they land
(`world/herds.py`), and the Steward sows where she chooses, aimed at an
island from the phone or spread over all bare land by a model. The goal is
counted island by island (`Planet.islands`).
"""

from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'archipelago'
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

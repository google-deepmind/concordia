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

"""Herds: Daisyworld with a grazing food web.

The tutorial's daisies, Steward, Advisor and Chronicler, plus a grazer clade.
Grazers eat daisies with a Holling type II response; their evolved diet
decides whether they crop white or black petals, and so whether they warm or
cool the planet. Self-interested herds follow the abundant white petals and
darken the world, so the Steward may need to thin them.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import grazers
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'herds'


def build_instances(
    context: game_lib.GameContext,
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  return [
      steward.steward_config(context, name='Gaia'),
      species_prefab.species_config(context, 'white'),
      species_prefab.species_config(context, 'black'),
      grazers.grazers_config(context),
      chronicler.chronicler_config(context),
  ]

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

"""Great Acceleration: coal, steam and steel against the air.

Three peoples who already know the iron age discover coal. Steam engines
power their machines (energy multiplies labor), steel mills make better
tools, and railways carry more, but every ton of coal puts carbon and soot
into the air. They must make enough energy without doubling the air's CO2
or leaving the habitable band; the Steward tends forests and the climate.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import polity
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'great_acceleration'
BIOMES = ('grass', 'forests')


def build_instances(
    context: game_lib.GameContext,
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  civ = context.planet.civ
  assert civ is not None
  return (
      [steward.steward_config(context, name='Gaia')]
      + [species_prefab.species_config(context, key) for key in BIOMES]
      + [polity.polity_config(context, key) for key in sorted(civ.polities)]
      + [chronicler.chronicler_config(context)]
  )

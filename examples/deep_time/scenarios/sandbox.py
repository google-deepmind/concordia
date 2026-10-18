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

"""Sandbox: every system at once, won by a seed ship.

Three peoples of the electric age share a living world of grass and forest
with the Steward. Coal, treaties, the Gazette, Gaia's backlash, trade,
automation, plans and rocketry are all in play. They win by launching a seed
ship (rockets, solar panels and seeds gathered from a world that is still
diverse) before the climate or the biosphere breaks.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import polity
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'sandbox'
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

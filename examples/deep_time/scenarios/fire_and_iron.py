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

"""Fire and Iron: the first civilizations and their forests.

Three peoples farm, cut wood, burn charcoal, mine ore and smelt iron on a
temperate world (`world/civ.py`, recipes on the contrib stock-flow ledger).
Felling forests pales the land and kilns warm the air; farms need a kind
climate. Grasslands and forests are the plant entities, each people is an
entity on the contrib async-deliberation prefab that shifts its labor every
generation, and the Steward keeps the climate kind while the peoples forge
their iron age.
"""

from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import polity
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'fire_and_iron'
BIOMES = ('grass', 'forests')


def build_instances(
    context: game_lib.GameContext, models: model_setup.Models
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  civ = context.planet.civ
  assert civ is not None
  return (
      [steward.steward_config(context, models.slow, name='Gaia')]
      + [
          species_prefab.species_config(context, key, models.slow)
          for key in BIOMES
      ]
      + [
          polity.polity_config(context, key, models.slow)
          for key in sorted(civ.polities)
      ]
      + [chronicler.chronicler_config(context, models.slow)]
  )

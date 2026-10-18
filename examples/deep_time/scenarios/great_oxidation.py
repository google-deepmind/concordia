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

"""Great Oxidation: carry life through the oxygen crisis.

Three kinds of microbial mat share a faint-sun world: methanogens warm it
with methane, cyanobacteria make oxygen, and aerobes need it. Reduced iron
soaks up the oxygen until it runs out; then oxygen floods the air, poisons
the anaerobes and burns the methane away, and the planet cools. Every mat is
an entity with the same three cognition tiers as the daisies, and the
Steward can seed aerobes, delay the crisis with iron or fight the cold.
"""

from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from concordia.typing import prefab as prefab_lib

NAME = 'great_oxidation'
MATS = ('methanogens', 'cyanobacteria', 'aerobes')


def build_instances(
    context: game_lib.GameContext,
) -> list[prefab_lib.InstanceConfig]:
  """Returns the entity instances of the scenario."""
  return (
      [steward.steward_config(context, name='Gaia')]
      + [species_prefab.species_config(context, key) for key in MATS]
      + [chronicler.chronicler_config(context)]
  )

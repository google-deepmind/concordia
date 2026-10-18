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

"""Deep Time scenarios. Each module exposes `NAME` and `build_instances`."""

from collections.abc import Callable
import types

from examples.deep_time import game as game_lib
from examples.deep_time.scenarios import archipelago
from examples.deep_time.scenarios import daisyworld
from examples.deep_time.scenarios import fire_and_iron
from examples.deep_time.scenarios import great_acceleration
from examples.deep_time.scenarios import great_migration
from examples.deep_time.scenarios import great_oxidation
from examples.deep_time.scenarios import herds
from examples.deep_time.scenarios import light_delay
from examples.deep_time.scenarios import sandbox
from examples.deep_time.scenarios import savanna_fire
from examples.deep_time.scenarios import snowball
from concordia.typing import prefab as prefab_lib

# A scenario's entities, as instance configs with plain params (each role's
# prefab holds the game and builds the entity).
BuildInstances = Callable[
    [game_lib.GameContext],
    list[prefab_lib.InstanceConfig],
]

SCENARIOS: dict[str, types.ModuleType] = {
    archipelago.NAME: archipelago,
    daisyworld.NAME: daisyworld,
    fire_and_iron.NAME: fire_and_iron,
    great_acceleration.NAME: great_acceleration,
    great_migration.NAME: great_migration,
    great_oxidation.NAME: great_oxidation,
    herds.NAME: herds,
    light_delay.NAME: light_delay,
    sandbox.NAME: sandbox,
    savanna_fire.NAME: savanna_fire,
    snowball.NAME: snowball,
}


def builder(name: str) -> BuildInstances:
  if name not in SCENARIOS:
    raise KeyError(f'Unknown scenario {name!r}; known: {sorted(SCENARIOS)}')
  return SCENARIOS[name].build_instances

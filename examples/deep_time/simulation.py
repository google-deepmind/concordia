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

"""Builds a Concordia `Simulation` for a Deep Time scenario.

Uses the stock `prefabs/simulation/generic.Simulation` with the stock
`Simultaneous` engine. Deep Time only adds a game master prefab and the
shared `GameContext`.
"""

from collections.abc import Mapping
from typing import Any

from concordia.components.game_master import next_acting
from concordia.environment.engines import simultaneous
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import scenarios
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import playable
from examples.deep_time.prefabs import polity
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import world_gm
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import human_input
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

WORLD_GM_PREFAB = 'deep_time_world__GameMaster'
WORLD_GM_NAME = 'Deep Time'


def build_config(
    context: game_lib.GameContext,
    models: model_setup.Models,
    max_ticks: int | None = None,
) -> prefab_lib.Config:
  """Returns the Concordia config for the context's scenario."""
  instances = scenarios.builder(context.scenario_name)(context, models)
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=WORLD_GM_PREFAB,
          role=prefab_lib.Role.GAME_MASTER,
          params={'name': WORLD_GM_NAME, 'context': context},
      )
  )
  premise = str(context.scenario.get('premise', ''))
  for instance in instances:
    if instance.role == prefab_lib.Role.ENTITY and premise:
      context.observe(instance.params['name'], premise)
  return prefab_lib.Config(
      default_premise=premise,
      default_max_steps=max_ticks or context.planet.max_ticks,
      prefabs={
          **helper_functions.get_package_classes(entity_prefabs),
          WORLD_GM_PREFAB: world_gm.WorldGameMaster(),
          chronicler.CHRONICLER_PREFAB: chronicler.Chronicler(),
          polity.POLITY_PREFAB: polity.PREFAB(),
          playable.PLAYABLE_PREFAB: playable.PlayableEntity(),
      },
      instances=instances,
  )


def build_simulation(
    scenario_name: str,
    models: model_setup.Models,
    max_ticks: int | None = None,
    time_mode: str = 'sim',
    seed: int = 0,
    humans: Mapping[str, human_input.HumanInput] | None = None,
) -> tuple[simulation_lib.Simulation, game_lib.GameContext]:
  """Creates the shared game state and the simulation for a scenario.

  Args:
    scenario_name: Key in `data/scenarios.yaml`.
    models: Language models by role.
    max_ticks: Overrides the scenario's length.
    time_mode: 'sim' (reproducible slow minds) or 'async' (threads).
    seed: Seed of the planet's random processes.
    humans: Entities a person plays (by name), and the blocking reader that
      waits for that person (Concordia's `HumanInput` protocol). Today the
      Steward ('Gaia') and Mission Control take a person.

  Returns:
    The simulation and the shared game context.
  """
  context = game_lib.new_game(scenario_name, time_mode=time_mode, seed=seed)
  context.humans.update(humans or {})
  context.coached = models.mock is not None
  config = build_config(context, models, max_ticks)
  sim = simulation_lib.Simulation(
      config=config,
      model=models.fast,
      embedder=model_setup.hash_embedder,
      engine=simultaneous.Simultaneous(),
  )
  # Speciation adds entities while the simulation runs.
  context.slow_model = models.slow
  context.spawner = sim.add_entity
  return sim, context


def load_checkpoint(
    sim: simulation_lib.Simulation,
    context: game_lib.GameContext,
    checkpoint: Mapping[str, Any],
) -> None:
  """Restores a game from `Simulation.make_checkpoint_data()`.

  The standard way: `Simulation.load_from_checkpoint` restores each entity's
  and game master's components, the topic bus and the sim-time scheduler
  with them (each entity's `TopicBusComponent`, the World GM's
  `TopicObservations` and `NumericClock`, every `SimTimePublisher` and
  transport), and the world through the World GM's `WorldStep`. Species
  founded during play come first: their prefab params hold live components,
  which a checkpoint cannot rebuild, so they are founded again from the
  world the checkpoint carries, then restored like the rest.

  Args:
    sim: A new simulation of the same scenario and seed (`build_simulation`).
    context: Its game context.
    checkpoint: The saved checkpoint (JSON-safe).
  """
  world = checkpoint['game_masters'][WORLD_GM_NAME]['components'][
      'context_components'
  ][next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY]['world']
  present = {entity.name for entity in sim.entities}
  missing = [name for name in world.get('spawned', []) if name not in present]
  if missing:
    context.set_state(world)  # The planet knows the new species.
    evolution = context.planet.evolution
    assert evolution is not None and context.slow_model is not None
    for name in missing:
      key = next(s.key for s in evolution.species if s.name == name)
      sim.add_entity(
          species_prefab.species_config(context, key, context.slow_model)
      )
  sim.load_from_checkpoint(dict(checkpoint))

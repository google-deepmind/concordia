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

from concordia.environment.engines import simultaneous
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import scenarios
from examples.deep_time.prefabs import chronicler
from examples.deep_time.prefabs import colony
from examples.deep_time.prefabs import grazers
from examples.deep_time.prefabs import mission
from examples.deep_time.prefabs import polity
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.prefabs import steward
from examples.deep_time.prefabs import world_gm
from concordia.language_model import language_model
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import human_input
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

WORLD_GM_PREFAB = 'deep_time_world__GameMaster'
WORLD_GM_NAME = 'Deep Time'


def build_config(
    context: game_lib.GameContext,
    max_ticks: int | None = None,
    views: str | None = None,
) -> prefab_lib.Config:
  """Returns the Concordia config for the context's scenario."""
  instances = scenarios.builder(context.scenario_name)(context)
  # The World GM installs every role's rules from the role's params.
  roles = [
      {'prefab': instance.prefab, 'params': dict(instance.params)}
      for instance in instances
      if instance.role == prefab_lib.Role.ENTITY
  ]
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=WORLD_GM_PREFAB,
          role=prefab_lib.Role.GAME_MASTER,
          params={'name': WORLD_GM_NAME, 'roles': roles, 'views': views},
      )
  )
  role_prefabs = {
      chronicler.CHRONICLER_PREFAB: chronicler.Chronicler(context=context),
      steward.STEWARD_PREFAB: steward.Steward(context=context),
      species_prefab.SPECIES_PREFAB: species_prefab.Species(context=context),
      grazers.GRAZER_PREFAB: grazers.Grazers(context=context),
      polity.POLITY_PREFAB: polity.Polity(context=context),
      mission.MISSION_PREFAB: mission.Mission(context=context),
      colony.COLONY_PREFAB: colony.Colony(context=context),
  }
  premise = str(context.scenario.get('premise', ''))
  return prefab_lib.Config(
      default_premise=premise,
      default_max_steps=max_ticks or context.planet.max_ticks,
      # Each role's prefab holds the game; its instances' params are plain.
      prefabs={
          **helper_functions.get_package_classes(entity_prefabs),
          **role_prefabs,
          WORLD_GM_PREFAB: world_gm.WorldGameMaster(
              context=context, roles=role_prefabs
          ),
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
    overrides: Mapping[str, Any] | None = None,
    bots: Mapping[str, language_model.LanguageModel] | None = None,
    views: str | None = None,
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
    overrides: Scenario flags merged into the scenario's data
      (`game.new_game`).
    bots: Entities a bot plays (by name), and the model each acts with: the
      role's own entity, its act component on that model.
    views: For a game a person's page shows: 'play' or 'watch'. The World GM
      then publishes each role's view (`views.py`: its crises, its needs and
      a Steward's logbook); None for a game no page shows.

  Returns:
    The simulation and the shared game context.
  """
  context = game_lib.new_game(
      scenario_name, time_mode=time_mode, seed=seed, overrides=overrides
  )
  context.humans.update(humans or {})
  context.bots.update(bots or {})
  # The roles' slow minds think with it (and so do species founded later).
  context.slow_model = models.slow
  config = build_config(context, max_ticks, views=views)
  sim = simulation_lib.Simulation(
      config=config,
      model=models.fast,
      embedder=model_setup.hash_embedder,
      engine=simultaneous.Simultaneous(),
  )
  # Speciation adds entities while the simulation runs.
  context.spawner = sim.add_entity
  return sim, context


def load_checkpoint(
    sim: simulation_lib.Simulation,
    checkpoint: Mapping[str, Any],
) -> None:
  """Restores a game from `Simulation.make_checkpoint_data()`.

  The standard way: `Simulation.load_from_checkpoint` restores each entity's
  and game master's components, the topic bus and the sim-time scheduler
  with them (each entity's `TopicBusComponent`, the World GM's
  `TopicObservations` and `NumericClock`, every `SimTimePublisher` and
  transport), and the world through the World GM's `WorldStep`.

  Entities founded during play (new species) are added first, with the stock
  `Simulation.add_entity`, from the prefab and the plain params their
  checkpoint keeps (`species.Species` holds the game). Left to the load,
  they would be built after the shared topic bus is restored, and their own
  `TopicBusComponent` would restore it again, replaying retained messages
  into every other entity's subscribers.

  Args:
    sim: A new simulation of the same scenario and seed (`build_simulation`).
    checkpoint: The saved checkpoint (JSON-safe).
  """
  present = {entity.name for entity in sim.entities}
  for name, saved in checkpoint.get('entities', {}).items():
    if name not in present:
      sim.add_entity(
          prefab_lib.InstanceConfig(
              prefab=saved['prefab_type'],
              role=prefab_lib.Role.ENTITY,
              params=dict(saved['entity_params']),
          )
      )
  sim.load_from_checkpoint(dict(checkpoint))

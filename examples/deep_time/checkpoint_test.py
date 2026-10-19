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

"""A game checkpointed the standard Concordia way replays exactly.

`Simulation.make_checkpoint_data()` saves every entity's and game master's
components (the topic bus, the sim-time scheduler, beliefs in transit and the
world among them); `simulation.load_checkpoint` restores them into a new
simulation. The restored game must then play on exactly as the live one does.
"""

import copy
import json
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory as associative_memory
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import scenarios
from examples.deep_time import simulation
from examples.deep_time.world import planet as planet_lib
from concordia.typing import prefab as prefab_lib


def _build(scenario: str):
  return simulation.build_simulation(
      scenario, model_setup.make_models('mock'), seed=0
  )


def _after(context: game_lib.GameContext, tick: int) -> dict[str, Any]:
  """What happened after `tick`: the world's history and every move."""
  return {
      'history': json.dumps(
          [r for r in context.planet.history if r['tick'] > tick],
          sort_keys=True,
          default=str,
      ),
      'moves': {
          name: [entry for entry in actor.log if entry[0] > tick]
          for name, actor in context.actors.items()
      },
      'bus_seq': context.bus.get_state()['seq'],
  }


class _NoWorld:
  """Stands in for what an entity's build may not read (the world, rules)."""

  def __init__(self, what: str):
    self._what = what

  def __getattr__(self, name: str) -> Any:
    raise AssertionError(f'An entity build read {self._what}.{name}')

  def __getitem__(self, key: Any) -> Any:
    raise AssertionError(f'An entity build read {self._what}[{key!r}]')


class CheckpointTest(parameterized.TestCase):

  @parameterized.parameters(
      # Wake-ups and alarms; transported, noisy, late instruments.
      ('daisyworld', 4, 6),
      # A species founded before the checkpoint (rebuilt from its saved params).
      ('great_oxidation', 16, 4),
      # Peoples (the contrib async-deliberation entity), the market,
      # standing orders (reflexes) and news by courier.
      ('fire_and_iron', 6, 6),
      # Messages crossing the light delay, Mission Control and the colony.
      ('light_delay', 6, 6),
  )
  def test_a_restored_game_plays_on_exactly(self, scenario, saved_at, more):
    live, context = _build(scenario)
    live.play(premise='', max_steps=saved_at)
    tick = context.planet.tick
    checkpoint = json.loads(
        json.dumps(live.make_checkpoint_data(), default=str)
    )
    live.play(premise='', max_steps=more)

    restored, other = _build(scenario)
    simulation.load_checkpoint(restored, checkpoint)
    self.assertEqual(other.planet.tick, tick)
    restored.play(premise='', max_steps=more)
    self.assertEqual(_after(other, tick), _after(context, tick))
    if scenario == 'great_oxidation':
      self.assertNotEmpty(other.spawned)  # Rebuilt before the load.


class RolesTest(parameterized.TestCase):

  @parameterized.parameters(
      'daisyworld',
      'great_acceleration',
      'light_delay',
      'herds',
      'great_oxidation',
  )
  def test_the_world_gm_installs_every_roles_rules(self, scenario):
    context = game_lib.new_game(scenario)
    models = model_setup.make_models('mock')
    context.slow_model = models.slow
    config = simulation.build_config(context)
    roles = [i for i in config.instances if i.role == prefab_lib.Role.ENTITY]
    # Building the roles' entities installs nothing in the World GM, and
    # reads nothing of the world, the scenario or the rules: their static
    # settings come in their params.
    planet, scenario, evolution = (
        context.planet,
        context.scenario,
        context.evolution,
    )
    context.planet = cast(planet_lib.Planet, _NoWorld('planet'))
    context.scenario = cast(dict[str, Any], _NoWorld('scenario'))
    context.evolution = cast(Any, _NoWorld('evolution'))
    entities = []
    for instance in roles:
      prefab = copy.deepcopy(config.prefabs[instance.prefab])
      prefab.params = instance.params
      entities.append(
          prefab.build(
              models.fast,
              associative_memory.AssociativeMemoryBank(
                  sentence_embedder=model_setup.hash_embedder
              ),
          )
      )
    context.planet, context.scenario, context.evolution = (
        planet,
        scenario,
        evolution,
    )
    self.assertEqual(context.actors, {})
    self.assertEqual(context.sheets, {})
    # The World GM installs each role's rules, from its params, in order.
    gm_instance = next(
        i for i in config.instances if i.role == prefab_lib.Role.GAME_MASTER
    )
    gm = copy.deepcopy(config.prefabs[gm_instance.prefab])
    gm.params = gm_instance.params
    gm.entities = entities
    gm.build(models.fast, associative_memory.AssociativeMemoryBank())
    self.assertEqual(list(context.actors), [i.params['name'] for i in roles])
    self.assertNotEmpty(context.sheets)


class PlainParamsTest(parameterized.TestCase):

  @parameterized.parameters(*sorted(scenarios.SCENARIOS))
  def test_every_role_is_built_from_plain_params(self, scenario):
    # Each role's prefab holds the game and builds the entity; what a
    # checkpoint keeps of an instance is all there is to it.
    context = game_lib.new_game(scenario)
    context.slow_model = model_setup.make_models('mock').slow
    config = simulation.build_config(context)
    for instance in config.instances:
      self.assertEqual(
          json.loads(json.dumps(instance.params)), dict(instance.params)
      )


if __name__ == '__main__':
  absltest.main()

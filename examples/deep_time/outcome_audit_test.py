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

"""No silent failure: every order of every kept world has an outcome.

For each kept world, every role's every move (on offer, or saved for), and
for a people every town's every order and its words to every other people,
is resolved as the World GM resolves it; each says what happened, why not,
or when it lands, naming who acted (DESIGN §23.16, principle 11).
"""

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time import actors as actors_lib
from examples.deep_time import model_setup
from examples.deep_time import simulation

KEPT = ('daisyworld', 'fire_and_iron', 'great_acceleration', 'sandbox')


class OutcomeAuditTest(parameterized.TestCase):

  @parameterized.parameters(*KEPT)
  def test_every_order_of_every_role_says_what_happened(self, scenario):
    _, context = simulation.build_simulation(
        scenario, model_setup.make_models('mock')
    )
    planet = context.planet
    towns, civ = planet.settlements, planet.civ
    if towns is not None and towns.war and civ is not None:
      for key in civ.polities:  # Arms and grain: the moves of war on offer.
        towns.muster(key, 'levies', 6.0)
        civ.ledger.deposit(key, 'grain', 40.0, 'test')
    checked = 0
    for name, actor in context.actors.items():
      moves = [
          *getattr(actor, 'options', list)(),
          *getattr(actor, 'out_of_reach', list)(),
      ]
      for move in moves:
        outcome = actor.resolve(move)
        self.assertTrue(outcome.strip(), f'{scenario}: {name}: {move}')
        self.assertIn(name, outcome, f'{scenario}: {name}: {move}')
        checked += 1
      if not isinstance(actor, actors_lib.PolityActor):
        continue
      for town in actor.town_menu().values():
        row, col = town['at']
        for order in town['orders']:
          lines = actor.resolve(
              f'{actors_lib.KEEP_LABOR}\ntown {row},{col}: {order}'
          ).split('\n')
          self.assertLen(lines, 2, f'{scenario}: {name}: {order}')
          self.assertTrue(lines[1].strip(), f'{scenario}: {name}: {order}')
          checked += 1
      for other in actor.speaks_to():
        said = actor.resolve(
            f'{actors_lib.KEEP_LABOR}\nwords to the {other}: Peace.'
        )
        self.assertIn(f'{name} send words to the {other}.', said)
        checked += 1
    self.assertGreater(checked, 20)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the rules that play a people at war."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import model_setup
from examples.deep_time import play_audit
from examples.deep_time import simulation
from examples.deep_time import war_rules
from concordia.typing import entity as entity_lib

MOVES = [
    'put more labor into: farm the land',
    'put more labor into: arm levies with spears',
    actors_lib.MARCH_TAKE,
    'send grain and envoys to a town [5 grain]',
    'hold a feast in a town [5 grain]',
    actors_lib.KEEP_LABOR,
]
TOWNS = (
    'Our towns stand at 13,1 (capital), 16,2 (row,col); towns of others: the'
    ' Basalt clans at 5,14 (capital), 12,11; the Cedar league at 23,34'
    ' (capital), 24,37, 21,37.'
)


def _spec() -> entity_lib.ActionSpec:
  return entity_lib.free_action_spec(
      call_to_action='\n'.join(
          f'({chr(ord("a") + i)}) {m}' for i, m in enumerate(MOVES)
      ),
      tag='polity',
  )


class WarRulesTest(absltest.TestCase):

  def test_a_rival_arms_when_marched_on_and_strikes_back_at_a_wrong(self):
    rival = war_rules.Rival()
    threat = (
        'An army of the Basalt clans (9.4 levies) marches to take our town'
        ' on the coast, arriving in turn 12. '
        + TOWNS
    )
    self.assertEqual(
        rival(_spec(), {'sheet': threat}),
        'put more labor into: arm levies with spears',
    )
    strong = (
        'Our arms: 9.0 levies (strength 9.0; garrisons: 9.0 on the coast).'
        ' We hold grievances against the Basalt clans (took a town). '
        + TOWNS
    )
    self.assertEqual(  # On the wrongdoer's town that is not its capital.
        rival(_spec(), {'sheet': strong}), f'{actors_lib.MARCH_TAKE} @ 12,11'
    )
    courted = (
        'The Basalt clans court our town on the coast: its regard for them is'
        ' 0.60 of the 1.00 it needs to leave us (40% to go). '
        + TOWNS
    )
    self.assertEqual(
        rival(_spec(), {'sheet': courted}), 'hold a feast in a town [5 grain]'
    )

  def test_bread_sends_a_cart_from_each_town_to_a_different_town(self):
    sheet = 'Storehouse: grain 60.0. ' + TOWNS
    answer = war_rules.Bread()(_spec(), {'sheet': sheet}).split('\n')
    self.assertEqual(
        answer[1:],
        [
            'town 13,1: send grain and envoys to a town [5 grain] @ 12,11',
            'town 16,2: send grain and envoys to a town [5 grain] @ 24,37',
        ],
    )

  def test_the_computers_peoples_answer_war_in_a_mock_game(self):
    _, context = simulation.build_simulation(
        'great_acceleration',
        model_setup.make_models('mock'),
        humans={'Basalt clans': lambda request: ''},
    )
    self.assertIsInstance(context.scripted['Amber folk'], war_rules.Rival)
    self.assertIsInstance(context.scripted['Cedar league'], war_rules.Rival)
    self.assertNotIn('Basalt clans', context.scripted)  # A person plays it.

  def test_a_people_played_by_a_rule_through_a_short_game(self):
    row = play_audit.play_people('great_acceleration', 'sword', 0, ticks=4)
    self.assertEqual(row['people'], 'Amber folk')
    self.assertEqual(row['ticks'], 4)
    self.assertIn(
        'take', play_audit.summarize_people([row])['bots']['sword']['gained']
    )


if __name__ == '__main__':
  absltest.main()

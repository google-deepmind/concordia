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

"""Tests for Pass it on: the line, the verdict, the stars and the bots."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from examples.crucible import harness
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys.passiton import bots
from examples.crucible.toys.passiton import toy as toy_lib

_DATA = toys.load_data('passiton')
# The message a first player is likely to try, and whether it should work:
# each level after the first is built so the obvious message fails.
_NAIVE = {
    'straight_up': ('Keep watch.', True),
    'two_at_the_top': ('Build; keep watch.', False),
    'dex_at_the_top': ('Build.', False),
    'dex_among_them': ('Build.', False),
    'dex_in_the_middle': ('Build.', False),
    'two_through_dex': ('Ash, build; Cy, keep watch.', False),
    'saying_less': ('Ash, keep watch; Cy, rest.', False),
    'everyone_through_dex': ('Ash, build; Bea, keep watch; Cy, rest.', False),
}


def _toy(level: str) -> toy_lib.PassItOnToy:
  return toy_lib.PassItOnToy(model_setup.mock_model(), _DATA, level)


class LevelTest(parameterized.TestCase):

  @parameterized.parameters(*_DATA['campaign'])
  def test_the_skilled_bot_earns_three_stars_on_the_first_try(self, level):
    toy = _toy(level)
    toy.resolve(toy_lib.SEAT, bots.skilled(toy, '', random.Random(0)))
    self.assertTrue(toy.done())
    self.assertEqual(toy.outcome()['stars'], 3)
    self.assertEqual(toy.outcome()['tries'], 1)

  @parameterized.parameters(*_DATA['campaign'])
  def test_the_obvious_message_works_only_where_meant(self, level):
    message, works = _NAIVE[level]
    receipt = _toy(level).preview(toy_lib.SEAT, message)
    self.assertEqual(bool(receipt.delta['solved']), works, receipt.summary)

  @parameterized.parameters(*_DATA['campaign'])
  def test_saying_nothing_never_solves(self, level):
    self.assertFalse(_toy(level).preview(toy_lib.SEAT, 'hold').delta['solved'])


class LineTest(absltest.TestCase):

  def test_dex_flips_each_part_and_cy_does_what_reached_cy(self):
    toy = _toy('dex_in_the_middle')
    receipt = toy.resolve(toy_lib.SEAT, "Don't build.")
    attempt = toy.world['attempts'][-1]
    self.assertEqual(attempt['hops'][0]['who'], 'Dex')
    self.assertEqual(attempt['hops'][0]['said'], 'build.')
    self.assertEqual(attempt['acted'][0]['does'], 'build')
    self.assertIn('Dex passes it on: "build."', receipt.summary)
    self.assertTrue(toy.done())

  def test_three_tries_then_over_and_a_long_message_costs_no_try(self):
    toy = _toy('dex_at_the_top')
    receipt = toy.resolve(toy_lib.SEAT, ' '.join(['build'] * 13))
    self.assertIn('over the limit', receipt.summary)
    self.assertEqual(toy.world['attempts'], [])
    for _ in range(3):
      toy.resolve(toy_lib.SEAT, 'Build.')
    self.assertTrue(toy.done())
    self.assertIsNone(toy.outcome()['winner'])
    self.assertEqual(toy.outcome()['stars'], 0)

  def test_the_view_shows_the_line_and_every_try(self):
    toy = _toy('two_through_dex')
    toy.resolve(toy_lib.SEAT, 'Build.')
    view = toy.view()
    self.assertEqual([hop['who'] for hop in view['line']], ['Dex'])
    self.assertIn('opposite', view['line'][0]['passes'])
    self.assertEqual([r['who'] for r in view['receivers']], ['Ash', 'Cy'])
    self.assertLen(view['attempts'], 1)
    self.assertIsNone(view['lesson'])

  def test_a_whole_game_through_the_harness(self):
    game = harness.build_game(
        'passiton', model_setup.mock_model(), 'skilled', 0, 'saying_less'
    )
    game.play()
    self.assertEqual(game.outcome()['stars'], 3)


if __name__ == '__main__':
  absltest.main()

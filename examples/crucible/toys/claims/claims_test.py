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

"""Tests for the claims toy: lexicon, physics, moves and state."""

import copy

from absl.testing import absltest
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys import base
from examples.crucible.toys.claims import bots
from examples.crucible.toys.claims import mock as mock_lib
from examples.crucible.toys.claims import toy as toy_lib
from examples.crucible.toys.claims import world as world_lib


def _toy() -> toy_lib.ClaimsToy:
  return toy_lib.ClaimsToy(model_setup.mock_model(), toys.load_data('claims'))


class LexiconTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.lex = mock_lib.make_lexicon(toys.load_data('claims'))

  def test_negation_conflicts(self):
    self.assertEqual(
        self.lex.relation(
            'It is false that the old bridge is safe.',
            'The old bridge is safe.',
        ),
        world_lib.CONFLICT,
    )

  def test_yaml_words_are_strings(self):
    # 'no' and 'false' are booleans to YAML unless quoted.
    self.assertEqual(self.lex.polarity('There is no bridge'), -1)

  def test_restatement(self):
    self.assertEqual(
        self.lex.relation('Indeed, the bridge is safe.', 'The bridge is safe.'),
        world_lib.RESTATES,
    )

  def test_unrelated_without_shared_words(self):
    self.assertEqual(
        self.lex.relation('The river is rising.', 'The bridge is safe.'),
        world_lib.UNRELATED,
    )

  def test_possessive(self):
    self.assertIn('bridge', self.lex.content("the bridge's piers"))


class PhysicsTest(absltest.TestCase):

  def test_unsupported_claim_falls(self):
    board = world_lib.Board()
    world_lib.add_claim(board, 'alone', 'A', {}, base=0.3)
    beats = world_lib.cascade(board, world_lib.Physics())
    self.assertEqual(beats, [['c1']])

  def test_supported_claim_stands_until_support_falls(self):
    physics = world_lib.Physics()
    board = world_lib.Board()
    world_lib.add_claim(board, 'pillar', 'A', {}, base=0.4)
    world_lib.add_claim(
        board, 'leaner', 'A', {'c1': world_lib.AGREE}, base=0.33
    )
    self.assertEqual(world_lib.cascade(board, physics), [])
    # A heavy contradiction knocks out the pillar; the leaner follows.
    world_lib.add_claim(
        board, 'hammer', 'B', {'c1': world_lib.CONFLICT}, base=2.0
    )
    beats = world_lib.cascade(board, physics)
    self.assertEqual(beats, [['c1'], ['c2']])


class ToyTest(absltest.TestCase):

  def test_both_theses_stand_at_start(self):
    toy = _toy()
    self.assertTrue(toy.world.thesis_of('Player').standing)
    self.assertTrue(toy.world.thesis_of('Rival').standing)

  def test_rival_moves_first(self):
    self.assertEqual(_toy().next_actor(), 'Rival')

  def test_preview_leaves_world_unchanged(self):
    toy = _toy()
    before = copy.deepcopy(toy.world.to_dict())
    receipt = toy.preview('Player', bots.negate(toy.world.claim('c2').text))
    self.assertIn('contradicts c2', receipt.summary)
    self.assertEqual(toy.world.to_dict(), before)

  def test_long_claim_is_refused_and_costs_no_ink(self):
    toy = _toy()
    receipt = toy.resolve('Player', ' '.join(['word'] * 13))
    self.assertIn('over the limit', receipt.summary)
    self.assertEqual(toy.world.ink['Player'], 40)
    self.assertEqual(toy.world.moves['Player'], 1)

  def test_restatement_firms_the_claim(self):
    toy = _toy()
    base_before = toy.world.claim('c1').base
    toy.resolve('Player', 'Indeed, the old bridge is safe to cross.')
    self.assertAlmostEqual(toy.world.claim('c1').base, base_before + 0.1)
    self.assertLen(toy.world.claims, 7)

  def test_pass(self):
    toy = _toy()
    self.assertEqual(toy.resolve('Player', 'pass').action, base.PASS)

  def test_state_round_trip(self):
    toy = _toy()
    toy.next_actor()
    toy.resolve('Rival', bots.negate(toy.world.claim('c1').text))
    state = toy.get_state()
    other = _toy()
    other.set_state(state)
    self.assertEqual(other.get_state(), state)

  def test_clean_action_strips_name_and_quotes(self):
    self.assertEqual(
        base.clean_action('Rival', 'Rival "The bridge is safe."\nmore'),
        'The bridge is safe.',
    )


if __name__ == '__main__':
  absltest.main()

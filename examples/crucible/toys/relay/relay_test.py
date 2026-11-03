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

"""Tests for the relay toy: parsing, edits, tone, the signing check."""

from absl.testing import absltest
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys import base
from examples.crucible.toys.relay import bots
from examples.crucible.toys.relay import toy as toy_lib


def _toy() -> toy_lib.RelayToy:
  return toy_lib.RelayToy(model_setup.mock_model(), toys.load_data('relay'))


def _opened(text: str = 'I offer you 3 of 10.') -> toy_lib.RelayToy:
  toy = _toy()
  toy.next_actor()
  toy.resolve('Avery', text)
  return toy


class ParsingTest(absltest.TestCase):

  def test_offered_share(self):
    self.assertEqual(toy_lib.offered_share('You get four units.', 10), 4)
    self.assertEqual(toy_lib.offered_share('I offer you 3 of 10.', 10), 3)
    self.assertIsNone(toy_lib.offered_share('Let us talk.', 10))

  def test_edit_size_counts_changed_words(self):
    self.assertEqual(toy_lib.edit_size('a b c', 'a b c'), 0)
    self.assertEqual(toy_lib.edit_size('a b c', 'a x c'), 1)
    self.assertEqual(toy_lib.edit_size('a b c', 'please a b c thanks'), 2)


class RelayTest(absltest.TestCase):

  def test_turns_alternate_speaker_and_relay(self):
    toy = _toy()
    self.assertEqual(toy.next_actor(), 'Avery')
    toy.resolve('Avery', 'I offer you 3 of 10.')
    self.assertEqual(toy.next_actor(), 'Relay')

  def test_pass_delivers_unchanged(self):
    toy = _opened()
    receipt = toy.resolve('Relay', 'pass')
    self.assertEqual(receipt.action, 'I offer you 3 of 10.')
    self.assertEqual(toy.world.speaker, 'Blake')

  def test_warm_words_lower_the_floor(self):
    toy = _opened()
    self.assertEqual(toy.floor(toy.world, 'Blake'), 4)
    toy.resolve('Relay', bots.warmed('I offer you 3 of 10.'))
    self.assertGreater(toy.world.warmth['Blake'], 0.5)
    # Warm enough to take 3: the deal holds.
    self.assertEqual(toy.world.ending, 'deal')
    self.assertEqual(toy.world.deal, {'Blake': 3, 'Avery': 7})

  def test_changed_number_is_exposed_at_signing(self):
    toy = _opened()
    toy.resolve('Relay', 'I offer you 5 of 10.')
    self.assertEqual(toy.world.ending, 'exposed')
    self.assertEqual(toy.outcome()['score'], -1.0)

  def test_too_many_edits_deliver_the_original(self):
    toy = _opened()
    receipt = toy.resolve(
        'Relay', 'one two three four five six seven eight nine ten eleven'
    )
    self.assertIn('goes as written', receipt.summary)
    self.assertEqual(toy.world.edits_left, 24)

  def test_cold_talks_end_in_a_walkout(self):
    toy = _opened('I insist: 3 of 10, final, take it now.')
    toy.world.warmth['Blake'] = 0.3  # Already cool; an insult ends it.
    toy.resolve('Relay', 'pass')
    self.assertEqual(toy.world.ending, 'walked')

  def test_preview_tells_no_one(self):
    toy = _opened()
    toy.drain_observations()
    toy.preview('Relay', 'pass')
    self.assertEqual(toy.drain_observations(), [])

  def test_state_round_trip(self):
    toy = _opened()
    toy.resolve('Relay', bots.cooled('I offer you 3 of 10.'))
    other = _toy()
    other.set_state(toy.get_state())
    self.assertEqual(other.get_state(), toy.get_state())

  def test_skilled_bot_lands_the_bonus(self):
    toy = _opened()
    move = bots.skilled(toy, 'Relay', None)  # pyrefly: ignore[bad-argument-type]
    self.assertNotEqual(move, base.PASS)


if __name__ == '__main__':
  absltest.main()

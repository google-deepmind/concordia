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

"""Tests for the Night Wire: the story's line, its readers and its chapters."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from examples.crucible import harness
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys.nightwire import mock as mock_lib
from examples.crucible.toys.nightwire import toy as toy_lib
from examples.crucible.toys.passiton import bots

_DATA = toys.load_data('nightwire')
# The message a first player is likely to try in each chapter, and whether
# it should work: after the first chapter the obvious message fails.
_NAIVE = {
    'the_sluice': ('Open the sluice.', True),
    'the_station': ('Hold the train; open the sluice.', False),
    'the_mayor': ('Ring the bell.', False),
    'the_council': (
        'Harlan, sandbag the bridge; Dexter, ring the bell.',
        False,
    ),
    'wilf': ('Go to the mill.', False),
    'two_doors': ('Penrose, go to the mill; Cyrus, hold the train.', False),
    'say_less': ('Harlan, open the sluice; Cyrus, stay put.', False),
    'the_night': (
        'Harlan, sandbag the bridge; Dexter, ring the bell; Penrose, stay put.',
        False,
    ),
}
# The answer each chapter is built around (its lesson), in full words.
_MEANT = {
    'the_sluice': 'Open the sluice.',
    'the_station': 'Open the sluice; Cyrus, hold the train.',
    'the_mayor': "Don't ring the bell.",
    'the_council': "Sandbag the bridge; Dexter, don't ring the bell.",
    'wilf': "Don't go to the mill.",
    'two_doors': "Penrose, don't go to the mill; Cyrus, don't hold the train.",
    'say_less': "Harlan, don't open the sluice.",
    'the_night': "Harlan, don't sandbag the bridge; Dexter, ring the bell.",
}


def _toy(chapter: str) -> toy_lib.NightWireToy:
  return toy_lib.NightWireToy(model_setup.mock_model(), _DATA, chapter)


class ChapterTest(parameterized.TestCase):

  @parameterized.parameters(*_DATA['campaign'])
  def test_the_skilled_bot_earns_three_stars_on_the_first_try(self, chapter):
    toy = _toy(chapter)
    toy.resolve(toy_lib.SEAT, bots.skilled(toy, '', random.Random(0)))
    self.assertEqual(toy.outcome()['stars'], 3)
    self.assertEqual(toy.outcome()['tries'], 1)

  @parameterized.parameters(*_DATA['campaign'])
  def test_the_obvious_message_works_only_where_meant(self, chapter):
    message, works = _NAIVE[chapter]
    receipt = _toy(chapter).preview(toy_lib.SEAT, message)
    self.assertEqual(bool(receipt.delta['solved']), works, receipt.summary)

  @parameterized.parameters(*_DATA['campaign'])
  def test_the_lesson_solves_its_chapter(self, chapter):
    receipt = _toy(chapter).preview(toy_lib.SEAT, _MEANT[chapter])
    self.assertTrue(receipt.delta['solved'], receipt.summary)

  @parameterized.parameters(*_DATA['campaign'])
  def test_saying_nothing_never_solves(self, chapter):
    self.assertFalse(
        _toy(chapter).preview(toy_lib.SEAT, 'hold').delta['solved']
    )

  def test_every_chapter_names_only_people_and_jobs_in_the_story(self):
    for chapter in _DATA['campaign']:
      level = _DATA['scenarios'][chapter]
      for stage in level['line']:
        for who in [stage] if isinstance(stage, str) else stage:
          self.assertIn(who, _DATA['people'])
      for who, job in level['goal'].items():
        self.assertIn(job, level['jobs'])
        self.assertIn(_DATA['people'][who]['default'], level['jobs'])
      self.assertTrue(level['time'] and level['headline'] and level['task'])


class LineTest(absltest.TestCase):

  def test_wilf_says_the_opposite_and_keeps_the_names(self):
    toy = _toy('two_doors')
    result = toy.pass_up('Penrose, go to the mill; Cyrus, hold the train.')
    self.assertEqual(
        result['hops'][0]['said'],
        "Penrose, don't go to the mill; Cyrus, don't hold the train.",
    )
    self.assertEqual([a['does'] for a in result['acted']], ['stay_put'] * 2)

  def test_edith_sends_exactly_what_she_is_given(self):
    toy = _toy('the_station')
    said = toy.rehearse('Cyrus, hold train STOP Harlan, open sluice')['said']
    self.assertEqual(said, 'Cyrus, hold train STOP Harlan, open sluice')

  def test_stop_ends_a_sentence_as_on_a_real_wire(self):
    toy = _toy('the_station')
    receipt = toy.preview(
        toy_lib.SEAT, 'Open sluice STOP Cyrus hold train STOP'
    )
    self.assertTrue(receipt.delta['solved'], receipt.summary)
    self.assertEqual(receipt.delta['words'], 5)  # STOP is free

  def test_the_mayor_does_what_he_is_told_not_to_and_nothing_else(self):
    toy = _toy('the_council')
    acted = toy.pass_up("Dexter, don't ring the bell.")['acted']
    self.assertEqual({a['who']: a['does'] for a in acted},
                     {'Harlan': 'grind', 'Dexter': 'ring_bell'})  # fmt: skip
    acted = toy.pass_up('Ring the bell.')['acted']
    self.assertEqual({a['who']: a['does'] for a in acted},
                     {'Harlan': 'ring_bell', 'Dexter': 'stay_put'})  # fmt: skip

  def test_told_only_not_to_do_their_work_they_stay_put(self):
    mock = mock_lib.NightWireMock(_DATA)
    menu = ['open_sluice', 'grind', 'stay_put']
    self.assertEqual(mock.does("Don't keep grinding.", 'Harlan', menu),
                     'stay_put')  # fmt: skip
    self.assertEqual(mock.does("Don't stay put.", 'Cyrus', menu), 'stay_put')

  def test_three_tries_and_a_headline_only_when_solved(self):
    toy = _toy('the_mayor')
    for _ in range(3):
      toy.resolve(toy_lib.SEAT, 'Ring the bell.')
    self.assertTrue(toy.done())
    self.assertIsNone(toy.view()['headline'])
    toy = _toy('the_mayor')
    toy.resolve(toy_lib.SEAT, "Don't ring the bell.")
    self.assertEqual(
        toy.view()['headline'], _DATA['scenarios']['the_mayor']['headline']
    )

  def test_a_whole_game_through_the_harness(self):
    game = harness.build_game(
        'nightwire', model_setup.mock_model(), 'skilled', 0, 'the_night'
    )
    game.play()
    self.assertEqual(game.outcome()['stars'], 3)


class LookTest(absltest.TestCase):

  def test_the_view_tells_the_story_of_each_hop(self):
    toy = _toy('the_night')
    toy.resolve(toy_lib.SEAT, 'Ring the bell.')
    view = toy.view()
    self.assertEqual(view['kind'], 'nightwire')
    self.assertEqual(view['time'], '12:40 a.m.')
    self.assertEqual([hop['who'] for hop in view['line']], ['Wilf'])
    self.assertEqual(view['line'][0]['how'], 'flips')
    self.assertEqual(
        set(view['people']), {'Wilf', 'Harlan', 'Dexter', 'Penrose'}
    )
    self.assertEqual(view['people']['Dexter']['reads'], 'contrary')
    self.assertEqual(set(view['say']), set(view['jobs']))
    self.assertLen(view['attempts'], 1)

  def test_every_fact_in_the_players_prompt_is_in_the_view(self):
    for chapter in _DATA['campaign']:
      toy = _toy(chapter)
      prompt = toy.action_spec(toy_lib.SEAT).call_to_action
      view = toy.view()
      self.assertIn(view['task'], prompt)
      for who in view['people']:
        self.assertIn(who, prompt)
      self.assertIn(f'at most {view["max_words"]} words', prompt)


if __name__ == '__main__':
  absltest.main()

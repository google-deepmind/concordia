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

"""Tests for the Steward's omens: its words, its free act, its record."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.prefabs import polity as polity_prefab
from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib
from concordia.typing import entity as entity_lib

EFFECT = {'kind': 'omen', 'signs': ['a red dawn', 'a comet'], 'max_words': 5}


class OmenWordsTest(absltest.TestCase):

  def test_a_named_sign_and_its_words(self):
    self.assertEqual(
        planet_lib.omen_words('comet: burn less coal', EFFECT),
        ('a comet', 'burn less coal'),
    )
    self.assertEqual(
        planet_lib.omen_words('A Red Dawn: "the fields thirst"', EFFECT),
        ('a red dawn', 'the fields thirst'),
    )

  def test_words_alone_take_the_first_sign_and_are_cut(self):
    self.assertEqual(
        planet_lib.omen_words('one two three four five six seven', EFFECT),
        ('a red dawn', 'one two three four five'),
    )
    # A colon in the words is not a sign.
    self.assertEqual(
        planet_lib.omen_words('hear this: the hills burn', EFFECT),
        ('a red dawn', 'hear this: the hills burn'),
    )
    self.assertEqual(planet_lib.omen_words('', EFFECT), ('a red dawn', ''))


class StewardSpeaksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.context = game_lib.new_game('great_acceleration')
    self.planet = self.context.planet
    self.gaia = actors_lib.StewardActor('Gaia', self.planet)

  def test_where_it_may_speak_the_steward_answers_in_words(self):
    spec = self.gaia.action_spec()
    self.assertEqual(spec.output_type, entity_lib.OutputType.FREE)
    self.assertIn('speak an omen', spec.call_to_action)
    self.assertIn('(a) wait and save omega', spec.call_to_action)
    self.assertIn('(b) speak an omen [2 omega]', spec.call_to_action)
    # A world without omens keeps the menu as a choice.
    daisies = game_lib.new_game('daisyworld').planet
    choice = actors_lib.StewardActor('Gaia', daisies).action_spec()
    self.assertEqual(choice.output_type, entity_lib.OutputType.CHOICE)

  def test_words_are_an_omen_and_moves_are_moves(self):
    omega = self.planet.omega
    said = self.gaia.resolve('a comet: spend the black stone slowly')
    self.assertIn('sent an omen: a comet over every land', said)
    self.assertIn('"spend the black stone slowly."', said)
    self.assertEqual(
        self.planet.omens,
        [(self.planet.tick, 'a comet', 'spend the black stone slowly')],
    )
    self.assertAlmostEqual(self.planet.omega, omega - 2)
    # The omen recharges: spoken again at once, it is refused.
    self.assertIn('recharges', self.gaia.resolve('a red dawn: again'))
    self.assertLen(self.planet.omens, 1)
    # A model's answer starts with its name; a move is a move.
    waited = self.gaia.resolve('Gaia wait and save omega')
    self.assertIn('Gaia waits', waited)
    self.assertLen(self.planet.omens, 1)

  def test_the_omens_ride_in_the_checkpoint(self):
    self.gaia.resolve('a comet: the fields thirst')
    state = self.planet.get_state()
    fresh = game_lib.new_game('great_acceleration').planet
    fresh.set_state(state)
    self.assertEqual(fresh.omens, self.planet.omens)

  def test_the_mock_steward_answers_as_its_choice_policy_would(self):
    model = mock_language_model.DeepTimeMockModel()
    options = '\n'.join(
        f'({letter}) {option}'
        for letter, option in zip('abc', self.gaia.options()[:3])
    )
    prompt = (
        'Global mean temperature 14.0 C (habitable band 5.0 to 25.0 C).\n'
        'Exercise: You are Gaia, ... or speak an omen by answering with a'
        f' sign.\n{options}\nAnswer: Gaia '
    )
    self.assertEqual(model.sample_text(prompt), 'wait and save omega')


class CharacterTest(absltest.TestCase):

  def test_a_people_is_written_as_people_not_rules(self):
    rules = data.scenario('great_acceleration')['civilization']
    words = polity_prefab.character_words(rules['characters']['basalt'])
    self.assertTrue(words.startswith('Who they are: miners of the black hills'))
    self.assertIn('How they speak: blunt, short sentences', words)
    # No sheet tells a people how to play or what to do when told something.
    for sheet in rules['characters'].values():
      text = ' '.join(str(v) for v in sheet.values()).lower()
      for rule in ('if told', 'should', 'must', 'always obey', 'win'):
        self.assertNotRegex(text, rf'\b{rule}\b')
    self.assertIn('only people act', rules['conventions'])


if __name__ == '__main__':
  absltest.main()

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

"""Semantics follow systems: one look per kind, and a home for every fact."""

import re

from absl.testing import absltest
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys.orders import toy as toy_lib

CATEGORIES = ('events', 'temperaments', 'conditions', 'actions', 'stocks')


class LookTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.look = toys.load_data('orders_look')
    self.data = toys.load_data('orders')

  def test_every_ledger_kind_has_a_look(self):
    self.assertEqual(
        set(self.data['hazards']) | {'unseen'}, set(self.look['events'])
    )
    self.assertEqual(
        set(self.data['temperaments']), set(self.look['temperaments'])
    )
    self.assertEqual(set(self.data['actions']), set(self.look['actions']))
    # A member's condition: fresh or worn, or off duty on the scene's rota.
    self.assertEqual(
        {'fresh', 'worn', 'off_duty'}, set(self.look['conditions'])
    )

  def test_how_to_play_names_every_temperament(self):
    text = ' '.join(self.look['how_to_play']).lower()
    for name, member in self.data['crew'].items():
      self.assertIn(name.lower(), text)
      self.assertIn(member['temperament'], text)
    self.assertIn(str(self.data['max_words']), text)

  def test_every_matrix_property_has_one_look(self):
    matrix = toys.load_data('orders_matrix')
    properties = set(matrix['reading']['properties'])
    self.assertEqual(properties, set(self.look['properties']))
    for prop, where in self.look['properties'].items():
      category, key = where.split('.')
      self.assertIn(key, self.look[category], prop)

  def test_no_two_kinds_share_a_glyph_or_a_word(self):
    glyphs, words = [], []
    for category in CATEGORIES:
      for spec in self.look[category].values():
        glyphs.append(spec['glyph'])
        words.append(spec['word'])
    self.assertLen(set(glyphs), len(glyphs))
    self.assertLen(set(words), len(words))

  def test_every_colour_is_a_palette_token(self):
    for theme in ('light', 'dark'):
      palette = self.look['palette'][theme]
      for category in CATEGORIES:
        for spec in self.look[category].values():
          self.assertIn(spec['color'], palette)


class ViewTest(absltest.TestCase):
  """Every fact the commander's prompt states has a home in the view."""

  def test_prompt_facts_are_in_the_view(self):
    toy = toy_lib.OrdersToy(model_setup.mock_model(), toys.load_data('orders'))
    toy.resolve('Commander', 'Bea, keep watch; everyone else build.')
    prompt = toy.action_spec('Commander').call_to_action
    view = toy.view()
    turn, turns = map(int, re.search(r'Turn (\d+) of (\d+)', prompt).groups())
    self.assertEqual((view['turn'], view['turns']), (turn, turns))
    self.assertIn(view['situation'], prompt)
    supplies = int(re.search(r'Stores: (\d+) supplies', prompt).group(1))
    self.assertEqual(view['supplies'], supplies)
    damage = int(re.search(r'hull damage (\d+)', prompt).group(1))
    self.assertEqual(view['damage'], damage)
    progress = int(re.search(r'project (\d+) of', prompt).group(1))
    self.assertEqual(view['progress'], progress)
    points = int(re.search(r'Command: (\d+) points', prompt).group(1))
    self.assertEqual(view['points'], points)
    self.assertIn(f'"{view["order"]}"', prompt)
    for member in view['crew']:
      self.assertIn(f'{member["name"]} {member["condition"]}', prompt)
      self.assertIn(
          member['condition'], toys.load_data('orders_look')['conditions']
      )
    for turn_ahead in view['ahead']:
      if turn_ahead['event']:
        self.assertIn(
            f'turn {turn_ahead["turn"]}: {turn_ahead["event"]}', prompt
        )

  def test_each_member_says_which_clause_they_acted_on(self):
    toy = toy_lib.OrdersToy(model_setup.mock_model(), toys.load_data('orders'))
    toy.resolve('Commander', 'Bea, keep watch; everyone else build.')
    whys = {m['name']: m['why'] for m in toy.view()['crew']}
    self.assertEqual(whys['Bea'], 'Bea, keep watch')
    self.assertEqual(whys['Ash'], 'everyone else build')
    self.assertEqual(toy_lib.clause_text('None.'), 'none')
    self.assertEqual(toy_lib.clause_text('Answer: None.'), 'none')
    self.assertEqual(
        toy_lib.clause_text('"Build; repair leaks." applies'),
        'Build; repair leaks.',
    )

  def test_a_conditional_clause_is_read_only_while_it_holds(self):
    toy = toy_lib.OrdersToy(model_setup.mock_model(), toys.load_data('orders'))
    ship = toy.world
    ship.order = 'Build; if a leak opens, repair it.'
    self.assertEqual(toy.reading(ship, 'Cy', 1), ('build', 'Build'))
    ship.leak = True
    self.assertEqual(
        toy.reading(ship, 'Cy', 1), ('repair', 'if a leak opens, repair it')
    )

  def test_every_star_rule_has_words(self):
    look = toys.load_data('orders_look')['star_rules']
    data = toys.load_data('orders')
    for name, scene in data['scenarios'].items():
      for wants in scene.get('stars', {}).values():
        for key in wants:
          self.assertIn(key, look, name)
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'one_hand')
    self.assertEqual(
        toy.view()['star_rules'],
        {'2': {'max_words_spoken': 2}, '3': {'max_words_spoken': 1}},
    )

  def test_the_lesson_is_shown_once_the_scene_is_over(self):
    data = toys.load_data('orders')
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'one_hand')
    self.assertIsNone(toy.view()['lesson'])
    toy.resolve(toy.commander, 'Build.')
    while not toy.done():
      toy.resolve(toy.commander, toy_lib.HOLD)
    self.assertEqual(
        toy.view()['lesson'], data['scenarios']['one_hand']['lesson']
    )

  def test_the_view_hides_what_was_not_seen(self):
    toy = toy_lib.OrdersToy(model_setup.mock_model(), toys.load_data('orders'))
    self.assertTrue(all(a['event'] is None for a in toy.view()['ahead']))


if __name__ == '__main__':
  absltest.main()

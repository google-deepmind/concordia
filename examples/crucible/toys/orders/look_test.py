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
    self.assertEqual({'fresh', 'worn'}, set(self.look['conditions']))

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

  def test_the_view_hides_what_was_not_seen(self):
    toy = toy_lib.OrdersToy(model_setup.mock_model(), toys.load_data('orders'))
    self.assertTrue(all(a['event'] is None for a in toy.view()['ahead']))


if __name__ == '__main__':
  absltest.main()

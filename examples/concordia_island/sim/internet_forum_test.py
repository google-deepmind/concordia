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


import json
import unittest
from examples.concordia_island.sim import internet_forum


class InternetForumStateTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.player_names = ['Alice', 'Bob', 'Charlie']
    self.forum = internet_forum.InternetForumState(
        player_names=self.player_names
    )

  def test_create_profile_with_image(self):
    # The outer JSON parser extracts image.
    action_text = json.dumps({
        'text': (
            '{"action": "create_profile", "author": "Alice", "profile": {"bio":'
            ' "Looking for love"}}'
        ),
        'image': 'data:image/png;base64,...',
    })
    result = self.forum.parse_and_execute_action(
        action_text, entity_name='Alice'
    )
    self.assertIn('created a dating profile', result)

    # Check if profile is stored
    profiles = self.forum.get_profiles()
    self.assertIn('Alice', profiles)
    self.assertEqual(profiles['Alice']['bio'], 'Looking for love')

    # Check if post is created with image
    posts = self.forum.get_recent_posts()
    self.assertTrue(
        any(
            p.is_profile and p.image == 'data:image/png;base64,...'
            for p in posts
        )
    )

  def test_batch_swipe(self):
    # First create profiles so they can be swiped on (optional but good)
    self.forum._profiles['Bob'] = {'bio': 'Hi'}
    self.forum._profiles['Charlie'] = {'bio': 'Hello'}

    action_text = json.dumps({
        'action': 'swipe',
        'author': 'Alice',
        'decisions': {'Bob': 'Yes', 'Charlie': 'No'},
    })
    result = self.forum.parse_and_execute_action(
        action_text, entity_name='Alice'
    )
    self.assertIn('swiped on profiles', result)

    # Check swipes state
    state = self.forum.get_state()
    swipes = state.get('swipes', {})
    self.assertIn('Alice', swipes)
    self.assertEqual(swipes['Alice']['Bob'], 'Yes')
    self.assertEqual(swipes['Alice']['Charlie'], 'No')

  def test_mutual_match(self):
    # Bob swipes Yes on Alice
    self.forum._swipes['Bob'] = {'Alice': 'Yes'}

    # Alice swipes Yes on Bob
    action_text = json.dumps({
        'action': 'swipe',
        'author': 'Alice',
        'decisions': {'Bob': 'Yes'},
    })
    self.forum.parse_and_execute_action(action_text, entity_name='Alice')

    # Check notifications
    alice_notifications = self.forum.drain_notifications('Alice')
    self.assertTrue(any("It's a match!" in n for n in alice_notifications))

    bob_notifications = self.forum.drain_notifications('Bob')
    self.assertTrue(any("It's a match!" in n for n in bob_notifications))


if __name__ == '__main__':
  unittest.main()

# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for forum components."""

import json
from unittest import mock

from absl.testing import absltest
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution
from concordia.contrib.components.game_master import forum
from concordia.typing import entity as entity_lib


PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG
PLAYER_NAMES = ['Alice', 'Bob', 'Charlie']


class ForumStateTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(
        player_names=kwargs.pop('player_names', PLAYER_NAMES), **kwargs
    )

  def test_create_post(self):
    fs = self._make_forum()
    post_id = fs.create_post(
        author='Alice', title='Hello', content='World', timestamp='t0'
    )
    self.assertEqual(post_id, 0)
    posts = fs.get_recent_posts()
    self.assertLen(posts, 1)
    self.assertEqual(posts[0].author, 'Alice')
    self.assertEqual(posts[0].title, 'Hello')
    self.assertEqual(posts[0].content, 'World')

  def test_create_multiple_posts_increments_id(self):
    fs = self._make_forum()
    id0 = fs.create_post(author='Alice', title='A', content='a')
    id1 = fs.create_post(author='Bob', title='B', content='b')
    self.assertEqual(id0, 0)
    self.assertEqual(id1, 1)

  def test_reply_to_post(self):
    fs = self._make_forum()
    post_id = fs.create_post(author='Alice', title='T', content='C')
    reply_id = fs.reply_to_post(
        post_id=post_id, author='Bob', content='Nice', timestamp='t1'
    )
    self.assertIsNotNone(reply_id)
    posts = fs.get_recent_posts()
    self.assertLen(posts[0].replies, 1)
    self.assertEqual(posts[0].replies[0]['author'], 'Bob')
    self.assertEqual(posts[0].replies[0]['content'], 'Nice')

  def test_reply_to_nonexistent_post(self):
    fs = self._make_forum()
    result = fs.reply_to_post(post_id=999, author='Bob', content='Hi')
    self.assertIsNone(result)

  def test_upvote(self):
    fs = self._make_forum()
    post_id = fs.create_post(author='Alice', title='T', content='C')
    self.assertTrue(fs.upvote(post_id))
    self.assertTrue(fs.upvote(post_id))
    posts = fs.get_recent_posts()
    self.assertEqual(posts[0].votes, 2)

  def test_upvote_nonexistent(self):
    fs = self._make_forum()
    self.assertFalse(fs.upvote(999))

  def test_downvote(self):
    fs = self._make_forum()
    post_id = fs.create_post(author='Alice', title='T', content='C')
    self.assertTrue(fs.downvote(post_id))
    posts = fs.get_recent_posts()
    self.assertEqual(posts[0].votes, -1)

  def test_downvote_nonexistent(self):
    fs = self._make_forum()
    self.assertFalse(fs.downvote(999))

  def test_get_recent_posts_ordering(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='First', content='1')
    fs.create_post(author='Bob', title='Second', content='2')
    fs.create_post(author='Charlie', title='Third', content='3')
    posts = fs.get_recent_posts()
    self.assertLen(posts, 3)
    self.assertEqual(posts[0].title, 'Third')
    self.assertEqual(posts[2].title, 'First')

  def test_get_recent_posts_with_limit(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='First', content='1')
    fs.create_post(author='Bob', title='Second', content='2')
    fs.create_post(author='Charlie', title='Third', content='3')
    posts = fs.get_recent_posts(n=2)
    self.assertLen(posts, 2)
    self.assertEqual(posts[0].title, 'Third')
    self.assertEqual(posts[1].title, 'Second')

  def test_get_forum_summary_empty(self):
    fs = self._make_forum()
    summary = fs.get_forum_summary()
    self.assertIn('No posts yet', summary)

  def test_get_forum_summary_with_posts(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='My Post', content='Content')
    summary = fs.get_forum_summary()
    self.assertIn('My Post', summary)
    self.assertIn('Alice', summary)

  def test_extract_json_fenced(self):
    fs = self._make_forum()
    text = 'Some text\n```json\n{"action": "post"}\n```\nmore text'
    result = fs.extract_json(text)
    self.assertEqual(result, {'action': 'post'})

  def test_extract_json_bare(self):
    fs = self._make_forum()
    text = 'I will do {"action": "upvote", "post_id": 0}'
    result = fs.extract_json(text)
    self.assertEqual(result, {'action': 'upvote', 'post_id': 0})

  def test_extract_json_invalid(self):
    fs = self._make_forum()
    result = fs.extract_json('no json here at all')
    self.assertIsNone(result)

  def test_parse_and_execute_post(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Hi',
        'content': 'Hello world',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Alice created post #0', result)
    self.assertLen(fs.get_recent_posts(), 1)

  def test_parse_and_execute_reply(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'reply',
        'author': 'Bob',
        'post_id': 0,
        'content': 'Great post',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob replied to post #0', result)

  def test_parse_and_execute_reply_nonexistent(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'reply',
        'author': 'Bob',
        'post_id': 99,
        'content': 'Reply',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob created post #0', result)

  def test_parse_and_execute_upvote(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({'action': 'upvote', 'author': 'Bob', 'post_id': 0})
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob upvoted post #0', result)
    self.assertEqual(fs.get_recent_posts()[0].votes, 1)

  def test_parse_and_execute_downvote(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({'action': 'downvote', 'author': 'Bob', 'post_id': 0})
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob downvoted post #0', result)
    self.assertEqual(fs.get_recent_posts()[0].votes, -1)

  def test_parse_and_execute_unknown_action(self):
    fs = self._make_forum()
    action = json.dumps({'action': 'delete', 'author': 'Alice'})
    result = fs.parse_and_execute_action(action)
    self.assertIn('Unknown action type', result)

  def test_parse_and_execute_invalid_json(self):
    fs = self._make_forum()
    result = fs.parse_and_execute_action('not json at all')
    self.assertIn('Could not parse action', result)

  def test_get_and_set_state(self):
    fs1 = self._make_forum()
    fs1.create_post(author='Alice', title='T', content='C', timestamp='t0')
    fs1.reply_to_post(post_id=0, author='Bob', content='R', timestamp='t1')
    fs1.upvote(0)
    state = fs1.get_state()

    fs2 = self._make_forum()
    fs2.set_state(state)
    self.assertEqual(fs1.get_state(), fs2.get_state())

    posts = fs2.get_recent_posts()
    self.assertLen(posts, 1)
    self.assertEqual(posts[0].votes, 1)
    self.assertLen(posts[0].replies, 1)

  def test_to_html_empty(self):
    fs = self._make_forum()
    html = fs.to_html()
    self.assertIn('No posts yet', html)

  def test_to_html_with_posts(self):
    fs = self._make_forum()
    fs.create_post(
        author='Alice', title='Test Post', content='Body', timestamp='t0'
    )
    fs.reply_to_post(post_id=0, author='Bob', content='Reply', timestamp='t1')
    fs.upvote(0)
    html = fs.to_html()
    self.assertIn('Test Post', html)
    self.assertIn('Alice', html)
    self.assertIn('Reply', html)
    self.assertIn('Bob', html)

  def test_escape_html(self):
    fs = self._make_forum()
    self.assertEqual(fs._escape_html('<b>'), '&lt;b&gt;')
    self.assertEqual(fs._escape_html('a&b'), 'a&amp;b')
    self.assertEqual(fs._escape_html('"hi"'), '&quot;hi&quot;')


class ForumResolutionTest(absltest.TestCase):

  def _make_resolution_with_mocks(self, memory_contents):
    mock_memory = mock.MagicMock(spec=memory_component.Memory)
    mock_memory.scan.return_value = memory_contents

    forum_state = forum.ForumState(player_names=PLAYER_NAMES)

    mock_entity = mock.MagicMock()
    mock_entity.get_component.side_effect = lambda key, type_: {
        forum.DEFAULT_FORUM_COMPONENT_KEY: forum_state,
        memory_component.DEFAULT_MEMORY_COMPONENT_KEY: mock_memory,
    }.get(key)

    component = forum.ForumResolution(player_names=PLAYER_NAMES)
    component.set_entity(mock_entity)
    return component, forum_state

  def test_pre_act_resolve_with_action(self):
    action_json = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Test',
        'content': 'Hello',
    })
    memory_contents = [f'{PUTATIVE_EVENT_TAG} Alice: {action_json}']
    component, forum_state = self._make_resolution_with_mocks(memory_contents)

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    result = component.pre_act(action_spec)
    self.assertIn('Alice created post #0', result)
    self.assertEqual(component.get_active_entity_name(), 'Alice')
    self.assertLen(forum_state.get_recent_posts(), 1)

  def test_pre_act_resolve_no_action(self):
    component, _ = self._make_resolution_with_mocks(memory_contents=[])

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    result = component.pre_act(action_spec)
    self.assertEqual(result, 'Event: \n')

  def test_pre_act_non_resolve(self):
    component, _ = self._make_resolution_with_mocks(memory_contents=[])

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.FREE,
    )
    result = component.pre_act(action_spec)
    self.assertEqual(result, '')

  def test_get_and_set_state(self):
    component, _ = self._make_resolution_with_mocks(memory_contents=[])
    component.set_state({'resolved_suggestions': ['suggestion_a']})
    state = component.get_state()
    self.assertIn('suggestion_a', state['resolved_suggestions'])

    component2, _ = self._make_resolution_with_mocks(memory_contents=[])
    component2.set_state(state)
    self.assertEqual(component.get_state(), component2.get_state())

  def test_conversation_action_format(self):
    action_json = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Chat',
        'content': 'Hi there',
    })
    memory_contents = [f'{PUTATIVE_EVENT_TAG} Alice -- {action_json}']
    component, _ = self._make_resolution_with_mocks(memory_contents)

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    result = component.pre_act(action_spec)
    self.assertIn('Alice created post #0', result)
    self.assertEqual(component.get_active_entity_name(), 'Alice')


class ForumObservationTest(absltest.TestCase):

  def _make_observation_with_mocks(self, forum_state):
    mock_entity = mock.MagicMock()
    mock_entity.get_component.return_value = forum_state

    component = forum.ForumObservation()
    component.set_entity(mock_entity)
    return component

  def _make_observation_action_spec(self, player_name):
    call_to_action = forum.DEFAULT_CALL_TO_MAKE_OBSERVATION.format(
        name=player_name
    )
    return entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.MAKE_OBSERVATION,
    )

  def test_pre_act_with_events(self):
    fs = forum.ForumState(player_names=PLAYER_NAMES)
    fs.create_post(author='Bob', title='Post', content='C', timestamp='t0')

    component = self._make_observation_with_mocks(fs)
    action_spec = self._make_observation_action_spec('Alice')
    result = component.pre_act(action_spec)

    self.assertIn('Post', result)

  def test_pre_act_no_events(self):
    fs = forum.ForumState(player_names=PLAYER_NAMES)
    fs.create_post(author='Bob', title='Post', content='C', timestamp='t0')

    component = self._make_observation_with_mocks(fs)
    action_spec = self._make_observation_action_spec('Alice')
    result = component.pre_act(action_spec)

    self.assertNotIn('Something happened', result)
    self.assertIn('Post', result)

  def test_pre_act_non_observation(self):
    fs = forum.ForumState(player_names=PLAYER_NAMES)
    component = self._make_observation_with_mocks(fs)

    action_spec = entity_lib.ActionSpec(
        call_to_action='test',
        output_type=entity_lib.OutputType.FREE,
    )
    result = component.pre_act(action_spec)
    self.assertEqual(result, '')

  def test_get_and_set_state(self):
    fs = forum.ForumState(player_names=PLAYER_NAMES)
    component = self._make_observation_with_mocks(fs)
    state = component.get_state()
    self.assertEqual(state, {})
    component.set_state({})


if __name__ == '__main__':
  absltest.main()

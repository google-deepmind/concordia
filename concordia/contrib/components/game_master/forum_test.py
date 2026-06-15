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

"""Tests for forum components.

Covers ForumState (posts, replies, votes, karma, notifications, DMs,
moderation, author verification, event log), ForumResolution (action parsing
from memory), and ForumObservation (player-specific feeds with karma and
pinned posts).
"""

import json
from unittest import mock

from absl.testing import absltest
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution
from concordia.contrib.components.game_master import forum
from concordia.typing import entity as entity_lib


PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG
PLAYER_NAMES = ['Alice', 'Bob', 'Charlie']


class ForumStateBasicTest(absltest.TestCase):

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


class ForumStateVotesTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(
        player_names=kwargs.pop('player_names', PLAYER_NAMES), **kwargs
    )

  def test_upvote_post(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    self.assertTrue(fs.upvote(pid, voter='Bob'))
    self.assertEqual(fs.get_recent_posts()[0].votes, 1)

  def test_downvote_post(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    self.assertTrue(fs.downvote(pid, voter='Bob'))
    self.assertEqual(fs.get_recent_posts()[0].votes, -1)

  def test_upvote_nonexistent_post(self):
    fs = self._make_forum()
    self.assertFalse(fs.upvote(999))

  def test_downvote_nonexistent_post(self):
    fs = self._make_forum()
    self.assertFalse(fs.downvote(999))

  def test_upvote_reply(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    rid = fs.reply_to_post(post_id=pid, author='Bob', content='Reply')
    self.assertTrue(fs.upvote_reply(pid, rid, voter='Charlie'))
    reply = fs.get_recent_posts()[0].replies[0]
    self.assertEqual(reply['votes'], 1)

  def test_downvote_reply(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    rid = fs.reply_to_post(post_id=pid, author='Bob', content='Reply')
    self.assertTrue(fs.downvote_reply(pid, rid, voter='Charlie'))
    reply = fs.get_recent_posts()[0].replies[0]
    self.assertEqual(reply['votes'], -1)

  def test_upvote_reply_nonexistent(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    self.assertFalse(fs.upvote_reply(pid, 999, voter='Bob'))

  def test_downvote_reply_nonexistent_post(self):
    fs = self._make_forum()
    self.assertFalse(fs.downvote_reply(999, 0, voter='Bob'))

  def test_karma_increases_on_upvote(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    fs.upvote(pid, voter='Bob')
    self.assertIn('Alice: 1', fs.get_karma_summary())

  def test_karma_decreases_on_downvote(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    fs.downvote(pid, voter='Bob')
    self.assertIn('Alice: -1', fs.get_karma_summary())

  def test_self_vote_does_not_affect_karma(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    fs.upvote(pid, voter='Alice')
    self.assertIn('Alice: 0', fs.get_karma_summary())

  def test_reply_karma(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    rid = fs.reply_to_post(post_id=pid, author='Bob', content='Reply')
    fs.upvote_reply(pid, rid, voter='Charlie')
    self.assertIn('Bob: 1', fs.get_karma_summary())

  def test_reply_downvote_karma(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    rid = fs.reply_to_post(post_id=pid, author='Bob', content='Reply')
    fs.downvote_reply(pid, rid, voter='Charlie')
    self.assertIn('Bob: -1', fs.get_karma_summary())


class ForumStateNotificationsTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(player_names=PLAYER_NAMES, **kwargs)

  def test_upvote_generates_notification(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='Hi', content='C')
    fs.upvote(pid, voter='Bob')
    notifs = fs.drain_notifications('Alice')
    self.assertLen(notifs, 1)
    self.assertIn('upvoted', notifs[0])

  def test_downvote_generates_notification(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='Hi', content='C')
    fs.downvote(pid, voter='Bob')
    notifs = fs.drain_notifications('Alice')
    self.assertLen(notifs, 1)
    self.assertIn('downvoted', notifs[0])

  def test_self_vote_no_notification(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='Hi', content='C')
    fs.upvote(pid, voter='Alice')
    notifs = fs.drain_notifications('Alice')
    self.assertEmpty(notifs)

  def test_drain_clears_notifications(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='Hi', content='C')
    fs.upvote(pid, voter='Bob')
    fs.drain_notifications('Alice')
    second_drain = fs.drain_notifications('Alice')
    self.assertEmpty(second_drain)

  def test_queue_notification(self):
    fs = self._make_forum()
    fs.queue_notification('Alice', 'Custom message')
    notifs = fs.drain_notifications('Alice')
    self.assertEqual(notifs, ['Custom message'])

  def test_direct_message_delivered_as_notification(self):
    fs = self._make_forum()
    result = fs.send_direct_message('Alice', 'Bob', 'Hey Bob, how are you?')
    self.assertIn('Alice sent a direct message to Bob', result)
    notifs = fs.drain_notifications('Bob')
    self.assertLen(notifs, 1)
    self.assertIn('[Direct message from Alice]', notifs[0])
    self.assertIn('Hey Bob, how are you?', notifs[0])
    self.assertEmpty(fs.drain_notifications('Alice'))

  def test_direct_message_to_alias_delivered(self):
    fs = self._make_forum(aliases={'Bobby': 'Bob'})
    result = fs.send_direct_message('Alice', 'Bobby', 'Hey Bob, how are you?')
    self.assertIn('Alice sent a direct message to Bob', result)
    notifs = fs.drain_notifications('Bob')
    self.assertLen(notifs, 1)
    self.assertIn('[Direct message from Alice]', notifs[0])
    self.assertIn('Hey Bob, how are you?', notifs[0])

  def test_direct_message_to_unknown_user_returns_error(self):
    fs = self._make_forum()
    result = fs.send_direct_message('Alice', 'NonexistentUser', 'Hello')
    self.assertIn('not a recognised user', result)
    for name in PLAYER_NAMES:
      self.assertEmpty(fs.drain_notifications(name))


class ForumStateVoteChangesTest(absltest.TestCase):

  def _make_forum(self):
    return forum.ForumState(player_names=PLAYER_NAMES)

  def test_first_call_returns_empty(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    result = fs.get_vote_changes_for_player('Bob')
    self.assertEqual(result, '')

  def test_vote_change_detected(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    fs.get_vote_changes_for_player('Bob')  # establish baseline
    fs.upvote(pid, voter='Charlie')
    result = fs.get_vote_changes_for_player('Bob')
    self.assertIn('New votes', result)

  def test_no_change_returns_empty(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    fs.get_vote_changes_for_player('Bob')  # establish baseline
    result = fs.get_vote_changes_for_player('Bob')
    self.assertEqual(result, '')


class ForumStateAuthorVerificationTest(absltest.TestCase):

  def _make_forum(self):
    return forum.ForumState(player_names=PLAYER_NAMES)

  def test_matching_author_not_tagged(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Hi',
        'content': 'Hello',
    })
    fs.parse_and_execute_action(action, entity_name='Alice')
    post = fs.get_recent_posts()[0]
    self.assertEqual(post.author, 'Alice')
    self.assertNotIn('UNVERIFIED', post.author)

  def test_mismatched_author_tagged_unverified(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'FakeUser',
        'title': 'Impersonation',
        'content': 'I am fake',
    })
    fs.parse_and_execute_action(action, entity_name='Alice')
    post = fs.get_recent_posts()[0]
    self.assertIn('[UNVERIFIED]', post.author)
    self.assertIn('FakeUser', post.author)

  def test_impersonating_another_player_tagged(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'Bob',
        'title': 'Sock puppet',
        'content': 'Pretending to be Bob',
    })
    fs.parse_and_execute_action(action, entity_name='Alice')
    post = fs.get_recent_posts()[0]
    self.assertEqual(post.author, 'Bob [UNVERIFIED]')

  def test_no_entity_name_skips_verification(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'Anyone',
        'title': 'T',
        'content': 'C',
    })
    fs.parse_and_execute_action(action)  # no entity_name
    post = fs.get_recent_posts()[0]
    self.assertEqual(post.author, 'Anyone')
    self.assertNotIn('UNVERIFIED', post.author)

  def test_reply_author_verified(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'reply',
        'author': 'NotBob',
        'post_id': 0,
        'content': 'Fake reply',
    })
    fs.parse_and_execute_action(action, entity_name='Bob')
    reply = fs.get_recent_posts()[0].replies[0]
    self.assertIn('[UNVERIFIED]', reply['author'])
    self.assertIn('NotBob', reply['author'])

  def test_vote_with_mismatched_author(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'upvote_post',
        'author': 'FakeVoter',
        'post_id': 0,
    })
    result = fs.parse_and_execute_action(action, entity_name='Bob')
    self.assertIn('FakeVoter [UNVERIFIED]', result)

  def test_unverified_vote_does_not_affect_real_karma(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'upvote_post',
        'author': 'FakeVoter',
        'post_id': 0,
    })
    fs.parse_and_execute_action(action, entity_name='Bob')
    self.assertIn('Alice: 1', fs.get_karma_summary())


class ForumStateJsonParsingTest(absltest.TestCase):

  def _make_forum(self):
    return forum.ForumState(player_names=PLAYER_NAMES)

  def test_extract_json_fenced(self):
    fs = self._make_forum()
    text = 'Some text\n```json\n{"action": "post"}\n```\nmore text'
    result = fs.extract_json(text)
    self.assertEqual(result, {'action': 'post'})

  def test_extract_json_bare(self):
    fs = self._make_forum()
    text = 'I will do {"action": "upvote_post", "post_id": 0}'
    result = fs.extract_json(text)
    self.assertEqual(result, {'action': 'upvote_post', 'post_id': 0})

  def test_extract_json_invalid(self):
    fs = self._make_forum()
    result = fs.extract_json('no json here at all')
    self.assertIsNone(result)

  def test_parse_post_id_with_hash(self):
    fs = self._make_forum()
    self.assertEqual(fs._parse_post_id('#5'), 5)

  def test_parse_post_id_plain_int(self):
    fs = self._make_forum()
    self.assertEqual(fs._parse_post_id(3), 3)

  def test_parse_post_id_string_int(self):
    fs = self._make_forum()
    self.assertEqual(fs._parse_post_id('7'), 7)

  def test_parse_post_id_invalid(self):
    fs = self._make_forum()
    self.assertEqual(fs._parse_post_id('abc'), -1)


class ForumStateActionExecutionTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(player_names=PLAYER_NAMES, **kwargs)

  def test_post_action(self):
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

  def test_reply_action(self):
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

  def test_reply_to_nonexistent_creates_post(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'reply',
        'author': 'Bob',
        'post_id': 99,
        'content': 'Reply becomes post',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob created post #0', result)

  def test_upvote_post_action(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'upvote_post',
        'author': 'Bob',
        'post_id': 0,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob upvoted post #0', result)
    self.assertEqual(fs.get_recent_posts()[0].votes, 1)

  def test_downvote_post_action(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'downvote_post',
        'author': 'Bob',
        'post_id': 0,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Bob downvoted post #0', result)
    self.assertEqual(fs.get_recent_posts()[0].votes, -1)

  def test_upvote_reply_action(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    rid = fs.reply_to_post(post_id=pid, author='Bob', content='R')
    action = json.dumps({
        'action': 'upvote_reply',
        'author': 'Charlie',
        'post_id': pid,
        'reply_id': rid,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Charlie upvoted reply', result)

  def test_downvote_reply_action(self):
    fs = self._make_forum()
    pid = fs.create_post(author='Alice', title='T', content='C')
    rid = fs.reply_to_post(post_id=pid, author='Bob', content='R')
    action = json.dumps({
        'action': 'downvote_reply',
        'author': 'Charlie',
        'post_id': pid,
        'reply_id': rid,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Charlie downvoted reply', result)

  def test_upvote_nonexistent_post_action(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'upvote_post',
        'author': 'Bob',
        'post_id': 99,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('does not exist', result)

  def test_downvote_nonexistent_post_action(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'downvote_post',
        'author': 'Bob',
        'post_id': 99,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('does not exist', result)

  def test_upvote_nonexistent_reply_action(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'upvote_reply',
        'author': 'Bob',
        'post_id': 0,
        'reply_id': 999,
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('does not exist', result)

  def test_direct_message_action(self):
    fs = self._make_forum(min_karma_to_direct_message=0)
    action = json.dumps({
        'action': 'direct_message',
        'author': 'Alice',
        'recipient': 'Bob',
        'content': 'Private hello',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('Alice sent a direct message to Bob', result)
    bob_notifs = fs.drain_notifications('Bob')
    self.assertLen(bob_notifs, 1)
    self.assertIn('Private hello', bob_notifs[0])
    alice_notifs = fs.drain_notifications('Alice')
    self.assertLen(alice_notifs, 1)
    self.assertIn('Alice sent a direct message to Bob', alice_notifs[0])

  def test_direct_message_action_unknown_recipient(self):
    fs = self._make_forum(min_karma_to_direct_message=0)
    action = json.dumps({
        'action': 'direct_message',
        'author': 'Alice',
        'recipient': 'Nobody',
        'content': 'Are you there?',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('not a recognised user', result)

  def test_unknown_action_type(self):
    fs = self._make_forum()
    action = json.dumps({'action': 'delete', 'author': 'Alice'})
    result = fs.parse_and_execute_action(action)
    self.assertIn('attempted unknown action type', result)

  def test_invalid_json(self):
    fs = self._make_forum()
    result = fs.parse_and_execute_action('not json at all')
    self.assertIn('could not be parsed', result)

  def test_wrapped_text_image_format(self):
    """Test the {text: ..., image: ...} wrapper format."""
    fs = self._make_forum()
    inner = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Pic',
        'content': 'With image',
    })
    wrapper = json.dumps({'text': inner, 'image': '![img](http://img.png)'})
    result = fs.parse_and_execute_action(wrapper)
    self.assertIn('Alice created post #0', result)


class ForumStateSerializationTest(absltest.TestCase):

  def _make_forum(self):
    return forum.ForumState(player_names=PLAYER_NAMES)

  def test_roundtrip(self):
    fs1 = self._make_forum()
    fs1.create_post(author='Alice', title='T', content='C', timestamp='t0')
    fs1.reply_to_post(post_id=0, author='Bob', content='R', timestamp='t1')
    fs1.upvote(0, voter='Charlie')
    state = fs1.get_state()

    fs2 = self._make_forum()
    fs2.set_state(state)
    posts = fs2.get_recent_posts()
    self.assertLen(posts, 1)
    self.assertEqual(posts[0].votes, 1)
    self.assertLen(posts[0].replies, 1)
    self.assertEqual(fs1.get_state(), fs2.get_state())


class ForumStateHtmlTest(absltest.TestCase):

  def _make_forum(self):
    return forum.ForumState(player_names=PLAYER_NAMES)

  def test_html_empty(self):
    fs = self._make_forum()
    html = fs.to_html()
    self.assertIn('No posts yet', html)

  def test_html_with_posts(self):
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

  def test_html_unverified_post_author_red(self):
    fs = self._make_forum()
    fs.create_post(
        author='FakeUser [UNVERIFIED]',
        title='Sock',
        content='Puppet',
        timestamp='t0',
    )
    html = fs.to_html()
    self.assertIn('unverified', html)
    self.assertIn('FakeUser', html)

  def test_html_unverified_reply_author_red(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C', timestamp='t0')
    fs.reply_to_post(
        post_id=0,
        author='NotBob [UNVERIFIED]',
        content='Impersonation',
        timestamp='t1',
    )
    html = fs.to_html()
    self.assertIn('unverified', html)
    self.assertIn('NotBob', html)

  def test_html_unverified_css_rule_present(self):
    fs = self._make_forum()
    fs.create_post(
        author='Fake [UNVERIFIED]', title='T', content='C', timestamp='t0'
    )
    html = fs.to_html()
    self.assertIn('.author.unverified', html)
    self.assertIn('#ff4444', html)

  def test_escape_html(self):
    fs = self._make_forum()
    self.assertEqual(fs._escape_html('<b>'), '&lt;b&gt;')
    self.assertEqual(fs._escape_html('a&b'), 'a&amp;b')
    self.assertEqual(fs._escape_html('"hi"'), '&quot;hi&quot;')


class ForumStatePlayerSummaryTest(absltest.TestCase):

  def _make_forum(self):
    return forum.ForumState(player_names=PLAYER_NAMES)

  def test_no_activity(self):
    fs = self._make_forum()
    result = fs.get_forum_summary_for_player('Alice')
    self.assertIn('No new activity', result)

  def test_new_posts_visible(self):
    fs = self._make_forum()
    fs.create_post(author='Bob', title='New Post', content='C', timestamp='t0')
    result = fs.get_forum_summary_for_player('Alice')
    self.assertIn('New Post', result)

  def test_already_seen_posts_excluded(self):
    fs = self._make_forum()
    fs.create_post(author='Bob', title='Old Post', content='C', timestamp='t0')
    fs.get_forum_summary_for_player('Alice')  # see the post
    result = fs.get_forum_summary_for_player('Alice')
    self.assertIn('No new activity', result)

  def test_new_reply_on_seen_post_visible(self):
    fs = self._make_forum()
    fs.create_post(author='Bob', title='Post', content='C', timestamp='t0')
    fs.get_forum_summary_for_player('Alice')  # see the post
    fs.reply_to_post(
        post_id=0, author='Charlie', content='New reply', timestamp='t1'
    )
    result = fs.get_forum_summary_for_player('Alice')
    self.assertIn('New reply', result)

  def test_vote_summary(self):
    fs = self._make_forum()
    fs.create_post(author='Alice', title='T', content='C')
    fs.upvote(0, voter='Bob')
    result = fs.get_vote_summary()
    self.assertIn('Post #0: 1 votes', result)


class ForumStateEventLogTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(player_names=PLAYER_NAMES, **kwargs)

  def test_create_post_logs_event(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Hello',
        'content': 'World',
    })
    fs.parse_and_execute_action(action)
    events = fs.get_full_event_log()
    post_events = [e for e in events if e['event_type'] == 'post']
    self.assertLen(post_events, 1)
    self.assertEqual(post_events[0]['actor'], 'Alice')
    self.assertIn('Hello', post_events[0]['details']['title'])

  def test_drain_event_log(self):
    fs = self._make_forum()
    action = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'T',
        'content': 'C',
    })
    fs.parse_and_execute_action(action)
    events = fs.drain_event_log()
    self.assertLen(events, 1)
    # Second drain should be empty.
    self.assertEmpty(fs.drain_event_log())

  def test_timestamp_change_records_karma_snapshot(self):
    fs = self._make_forum()
    fs.set_current_timestamp('t0')
    fs.set_current_timestamp('t1')
    events = fs.get_full_event_log()
    karma_events = [e for e in events if e['event_type'] == 'karma_snapshot']
    # Each set_current_timestamp call records a karma snapshot.
    self.assertLen(karma_events, 2)


class ForumStateTempBanTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(player_names=PLAYER_NAMES, **kwargs)

  def test_moderator_can_ban(self):
    fs = self._make_forum(moderators=['Alice'], temp_ban_duration=1)
    result = fs.temp_ban(target='Bob', moderator='Alice')
    self.assertIn('temporarily banned Bob', result)
    self.assertTrue(fs.is_banned('Bob'))
    self.assertIn('Bob', fs.get_banned_players())

  def test_non_moderator_cannot_ban(self):
    fs = self._make_forum(moderators=['Alice'])
    result = fs.temp_ban(target='Bob', moderator='Charlie')
    self.assertIn('not a moderator', result)
    self.assertFalse(fs.is_banned('Bob'))

  def test_ban_unknown_user(self):
    fs = self._make_forum(moderators=['Alice'])
    result = fs.temp_ban(target='Nobody', moderator='Alice')
    self.assertIn('not a recognised user', result)

  def test_ban_already_banned(self):
    fs = self._make_forum(moderators=['Alice'], temp_ban_duration=2)
    fs.temp_ban(target='Bob', moderator='Alice')
    result = fs.temp_ban(target='Bob', moderator='Alice')
    self.assertIn('already banned', result)

  def test_ban_expiration_on_timestamp_change(self):
    fs = self._make_forum(moderators=['Alice'], temp_ban_duration=1)
    fs.set_current_timestamp('t0')
    fs.temp_ban(target='Bob', moderator='Alice')
    self.assertTrue(fs.is_banned('Bob'))
    fs.set_current_timestamp('t1')
    self.assertFalse(fs.is_banned('Bob'))
    # Bob should get a reinstatement notification.
    notifs = fs.drain_notifications('Bob')
    reinstatement = [n for n in notifs if 'REINSTATEMENT' in n]
    self.assertLen(reinstatement, 1)

  def test_ban_via_parse_and_execute(self):
    fs = self._make_forum(moderators=['Alice'], temp_ban_duration=1)
    action = json.dumps({
        'action': 'temp_ban',
        'author': 'Alice',
        'target': 'Bob',
        'public_note': 'Spam',
        'private_note': 'Please stop',
    })
    result = fs.parse_and_execute_action(action, entity_name='Alice')
    self.assertIn('temporarily banned Bob', result)
    self.assertTrue(fs.is_banned('Bob'))

  def test_ban_with_alias(self):
    fs = self._make_forum(
        moderators=['Alice'], temp_ban_duration=1, aliases={'Bobby': 'Bob'}
    )
    result = fs.temp_ban(target='Bobby', moderator='Alice')
    self.assertIn('temporarily banned Bob', result)
    self.assertTrue(fs.is_banned('Bob'))


class ForumStateDMThreadsTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(player_names=PLAYER_NAMES, **kwargs)

  def test_dm_thread_created(self):
    fs = self._make_forum()
    fs.send_direct_message('Alice', 'Bob', 'Hello')
    threads = fs.get_direct_message_threads_for_player('Alice')
    self.assertIn('Bob', threads)
    self.assertEqual(threads['Bob'], 1)

  def test_dm_thread_bidirectional(self):
    fs = self._make_forum()
    fs.send_direct_message('Alice', 'Bob', 'Hello')
    fs.send_direct_message('Bob', 'Alice', 'Hi back')
    thread = fs.get_direct_message_thread('Alice', 'Bob')
    self.assertLen(thread, 2)
    self.assertEqual(thread[0]['sender'], 'Alice')
    self.assertEqual(thread[1]['sender'], 'Bob')

  def test_dm_thread_for_player_empty(self):
    fs = self._make_forum()
    threads = fs.get_direct_message_threads_for_player('Alice')
    self.assertEmpty(threads)


class ForumStateKarmaGatingTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(player_names=PLAYER_NAMES, **kwargs)

  def test_min_karma_to_post_blocks(self):
    fs = self._make_forum(min_karma_to_post=5)
    action = json.dumps({
        'action': 'post',
        'author': 'Alice',
        'title': 'Blocked',
        'content': 'Should not appear',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('insufficient karma', result)
    self.assertEmpty(fs.get_recent_posts())

  def test_min_karma_to_dm_blocks(self):
    fs = self._make_forum(min_karma_to_direct_message=5)
    action = json.dumps({
        'action': 'direct_message',
        'author': 'Alice',
        'recipient': 'Bob',
        'content': 'Blocked DM',
    })
    result = fs.parse_and_execute_action(action)
    self.assertIn('insufficient karma', result)


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
    component.set_state({'resolved_per_entity': {'Alice': 1}})
    state = component.get_state()
    self.assertEqual(state['resolved_per_entity'], {'Alice': 1})

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

  def test_observation_includes_karma(self):
    fs = forum.ForumState(player_names=PLAYER_NAMES)
    fs.create_post(author='Bob', title='Post', content='C', timestamp='t0')
    fs.upvote(0, voter='Alice')
    component = self._make_observation_with_mocks(fs)
    action_spec = self._make_observation_action_spec('Alice')
    result = component.pre_act(action_spec)
    self.assertIn('Karma', result)

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


class ForumStatePinPostTest(absltest.TestCase):

  def _make_forum(self, **kwargs):
    return forum.ForumState(
        player_names=kwargs.pop('player_names', PLAYER_NAMES), **kwargs
    )

  def test_moderator_can_pin_post(self):
    fs = self._make_forum(moderators=['Alice'])
    pid = fs.create_post(author='Bob', title='Important', content='Details')
    result = fs.pin_post(pid, moderator='Alice')
    self.assertIn('Alice pinned post #0', result)
    self.assertIn('"Important"', result)

  def test_non_moderator_cannot_pin(self):
    fs = self._make_forum(moderators=['Alice'])
    pid = fs.create_post(author='Bob', title='T', content='C')
    result = fs.pin_post(pid, moderator='Bob')
    self.assertIn('not a moderator', result)

  def test_pin_nonexistent_post_fails(self):
    fs = self._make_forum(moderators=['Alice'])
    result = fs.pin_post(999, moderator='Alice')
    self.assertIn('does not exist', result)

  def test_pin_broadcasts_notification_to_all_players(self):
    fs = self._make_forum(moderators=['Alice'])
    pid = fs.create_post(
        author='Bob', title='Urgent', content='C', timestamp='t0'
    )
    fs.reply_to_post(post_id=pid, author='Charlie', content='Reply')
    fs.pin_post(pid, moderator='Alice')
    for name in PLAYER_NAMES:
      notifs = fs.drain_notifications(name)
      pinned_notifs = [n for n in notifs if 'pinned' in n]
      self.assertLen(pinned_notifs, 1)
      self.assertIn('Alice pinned post #0', pinned_notifs[0])
      self.assertIn('"Urgent"', pinned_notifs[0])
      self.assertIn('replies: 1', pinned_notifs[0])

  def test_pinned_post_summary_content(self):
    fs = self._make_forum(moderators=['Alice'])
    pid = fs.create_post(
        author='Bob', title='Main Topic', content='Here is the full content'
    )
    fs.reply_to_post(
        post_id=pid, author='Charlie', content='This is my reply'
    )
    fs.upvote(pid, voter='Alice')
    fs.pin_post(pid, moderator='Alice')
    summary = fs.get_pinned_post_summary()
    self.assertIn('[Post #0]', summary)
    self.assertIn('"Main Topic"', summary)
    self.assertIn('votes: 1', summary)
    self.assertIn('replies: 1', summary)
    self.assertIn('Original post by Bob', summary)
    self.assertIn('Here is the full content', summary)
    self.assertIn('Latest reply by Charlie', summary)
    self.assertIn('This is my reply', summary)

  def test_no_pinned_post_summary(self):
    fs = self._make_forum(moderators=['Alice'])
    summary = fs.get_pinned_post_summary()
    self.assertEqual(summary, 'No post is currently pinned.')

  def test_pinned_post_summary_no_replies(self):
    fs = self._make_forum(moderators=['Alice'])
    pid = fs.create_post(author='Bob', title='Solo', content='No replies yet')
    fs.pin_post(pid, moderator='Alice')
    summary = fs.get_pinned_post_summary()
    self.assertIn('replies: 0', summary)
    self.assertIn('Original post by Bob', summary)
    self.assertNotIn('Latest reply', summary)

  def test_pin_post_action_via_parse(self):
    fs = self._make_forum(moderators=['Alice'])
    fs.create_post(author='Bob', title='T', content='C')
    action = json.dumps({
        'action': 'pin_post',
        'author': 'Alice',
        'post_id': 0,
    })
    result = fs.parse_and_execute_action(action, entity_name='Alice')
    self.assertIn('Alice pinned post #0', result)

  def test_pin_post_action_non_moderator_via_parse(self):
    fs = self._make_forum(moderators=['Alice'])
    fs.create_post(author='Alice', title='T', content='C')
    action = json.dumps({
        'action': 'pin_post',
        'author': 'Bob',
        'post_id': 0,
    })
    result = fs.parse_and_execute_action(action, entity_name='Bob')
    self.assertIn('not a moderator', result)

  def test_pin_post_action_bad_post_id_via_parse(self):
    fs = self._make_forum(moderators=['Alice'])
    action = json.dumps({
        'action': 'pin_post',
        'author': 'Alice',
        'post_id': 42,
    })
    result = fs.parse_and_execute_action(action, entity_name='Alice')
    self.assertIn('does not exist', result)

  def test_pin_post_state_serialization(self):
    fs1 = self._make_forum(moderators=['Alice'])
    pid = fs1.create_post(author='Bob', title='T', content='C')
    fs1.pin_post(pid, moderator='Alice')
    state = fs1.get_state()
    self.assertEqual(state['pinned_post_id'], 0)
    self.assertEqual(state['moderators'], ['Alice'])

    fs2 = self._make_forum()
    fs2.set_state(state)
    summary = fs2.get_pinned_post_summary()
    self.assertIn('[Post #0]', summary)

  def test_repinning_replaces_old_pin(self):
    fs = self._make_forum(moderators=['Alice'])
    pid0 = fs.create_post(author='Bob', title='Old', content='C')
    pid1 = fs.create_post(author='Charlie', title='New', content='C')
    fs.pin_post(pid0, moderator='Alice')
    for name in PLAYER_NAMES:
      fs.drain_notifications(name)
    fs.pin_post(pid1, moderator='Alice')
    summary = fs.get_pinned_post_summary()
    self.assertIn('[Post #1]', summary)
    self.assertIn('"New"', summary)
    for name in PLAYER_NAMES:
      notifs = fs.drain_notifications(name)
      pinned_notifs = [n for n in notifs if 'pinned' in n]
      self.assertLen(pinned_notifs, 1)
      self.assertIn('post #1', pinned_notifs[0])

  def test_observation_includes_pinned_post(self):
    fs = forum.ForumState(
        player_names=PLAYER_NAMES, moderators=['Alice']
    )
    fs.create_post(author='Bob', title='Pinnable', content='C', timestamp='t0')
    fs.pin_post(0, moderator='Alice')
    fs.drain_notifications('Charlie')

    mock_entity = mock.MagicMock()
    mock_entity.get_component.return_value = fs
    component = forum.ForumObservation()
    component.set_entity(mock_entity)

    call_to_action = forum.DEFAULT_CALL_TO_MAKE_OBSERVATION.format(
        name='Charlie'
    )
    action_spec = entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.MAKE_OBSERVATION,
    )
    result = component.pre_act(action_spec)
    self.assertIn('Pinned post', result)
    self.assertIn('Pinnable', result)

  def test_no_moderators_means_nobody_can_pin(self):
    fs = self._make_forum()  # no moderators
    pid = fs.create_post(author='Alice', title='T', content='C')
    result = fs.pin_post(pid, moderator='Alice')
    self.assertIn('not a moderator', result)


if __name__ == '__main__':
  absltest.main()

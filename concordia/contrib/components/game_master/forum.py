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

"""Thread-safe Forum for managing a Reddit-like discussion board.

The forum is structured as three parts:
  - ForumState: A component wrapping the shared forum data (posts, replies,
    votes, karma, direct messages, moderation, and per-player notification
    queue). Registered under __forum__ and accessed by other components via
    get_entity().get_component(). Thread-safe.
  - ForumResolution: A component (should be registered as __resolution__) that
  parses
    player JSON actions and executes them on ForumState programmatically.
    No LLM calls.
  - ForumObservation: A component (should be registered as __make_observation__)
  that
    drains ForumState's per-player notification queue and returns the forum
    summary, vote changes, karma, and pinned post information. No LLM calls.

Features:
  - Reply-level upvoting and downvoting (not just post-level)
  - Per-player karma scores (visible to all on every step)
  - Karma +1 when someone upvotes your post/reply, -1 when downvoted
  - Self-votes do not affect karma
  - Direct messages between players
  - Temporary bans (moderator-only)
  - Post pinning (moderator-only)
  - Author verification (tags mismatched authors as [UNVERIFIED])
  - Karma gating for posts, replies, and DMs
  - Event logging for simulation timeline analysis
"""

from collections.abc import Sequence
import dataclasses
import datetime
import json
import re
import threading
from typing import Any

from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_FORUM_COMPONENT_KEY = '__forum__'
DEFAULT_FORUM_PRE_ACT_LABEL = '\nForum'

DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}? What do they now observe?'
    ' Only include information of which they are aware.'
)


@dataclasses.dataclass
class Post:
  """A single forum post with its metadata, votes, replies, and activity."""

  post_id: int
  author: str
  title: str
  content: str
  timestamp: str
  votes: int = 0
  image: str | None = None
  replies: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  vote_log: list[dict[str, str]] = dataclasses.field(default_factory=list)
  min_karma_to_reply: int = 0
  # Monotonically increasing sequence number bumped on every interaction
  # (create, reply, vote). Used by ForumBrowserTool for recency ordering.
  last_activity_seq: int = 0


class ForumState(entity_component.ContextComponent):
  """Thread-safe forum state managing posts, replies, votes, and observations.

  Registered under __forum__ and accessed by ForumResolution and
  ForumObservation components via get_entity().get_component(). This follows
  the same pattern as AssociativeMemory wrapping AssociativeMemoryBank.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      forum_name: str = 'Community Forum',
      max_summary_posts: int = 10,
      aliases: dict[str, str] | None = None,
      moderators: Sequence[str] | None = None,
      temp_ban_duration: int = 1,
      min_karma_to_post: int = -1,
      min_karma_to_direct_message: int = 1,
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._forum_name = forum_name
    self._max_summary_posts = max_summary_posts
    self._aliases = aliases or {}
    self._moderators = list(moderators) if moderators else []
    self._temp_ban_duration = temp_ban_duration
    self._min_karma_to_post = min_karma_to_post
    self._min_karma_to_direct_message = min_karma_to_direct_message
    self._pinned_post_id: int | None = None

    self._lock = threading.Lock()
    self._posts: dict[int, Post] = {}
    self._next_post_id = 0
    self._next_reply_id = 0
    self._karma: dict[str, int] = {name: 0 for name in player_names}
    self._notification_queue: dict[str, list[str]] = {
        name: [] for name in player_names
    }
    # Direct message threads: keyed by frozenset({sender, recipient}) as a
    # canonical pair key, stored as a sorted tuple string for serialization.
    # Each value is a list of {sender, content, timestamp} dicts.
    self._direct_message_threads: dict[str, list[dict[str, str]]] = {}

    self._last_seen_post_id: dict[str, int] = {
        name: -1 for name in player_names
    }
    self._last_seen_reply_id: dict[str, int] = {
        name: -1 for name in player_names
    }
    self._last_seen_votes: dict[str, dict[str, int]] = {
        name: {} for name in player_names
    }
    self._current_timestamp: str = ''
    # Temporary ban tracking: maps player name -> ban info dict.
    self._bans: dict[str, dict[str, Any]] = {}
    self._timestamp_change_count: int = 0
    # Monotonic counter for post activity ordering.
    self._activity_seq: int = 0

    # ── Event log for the SimulationTimeline ──
    # Append-only list recording every forum mutation with timestamps.
    self._event_log: list[dict[str, Any]] = []
    self._event_seq: int = 0
    # Tracks how far drain_event_log has consumed.
    self._event_log_drain_cursor: int = 0

  def set_current_timestamp(self, timestamp: str) -> None:
    """Set the default timestamp for new posts and replies.

    When the timestamp actually changes (new value differs from old),
    this also decrements remaining ban durations and reinstates players
    whose bans have expired.

    Args:
      timestamp: The new timestamp to set.
    """
    with self._lock:
      old_timestamp = self._current_timestamp
      self._current_timestamp = timestamp
      if timestamp and timestamp != old_timestamp:
        self._timestamp_change_count += 1
        self._process_ban_expirations_locked(timestamp)
        # Record a karma snapshot whenever the clock ticks.
        karma_entries = [
            f'{name}: {score}' for name, score in self._karma.items()
        ]
        self._record_event_locked(
            event_type='karma_snapshot',
            actor='[system]',
            summary=f'[Karma snapshot] {", ".join(karma_entries)}',
            details={'karma': dict(self._karma)},
        )

  def get_current_timestamp(self) -> str:
    """Return the current default timestamp."""
    with self._lock:
      return self._current_timestamp

  def _record_event_locked(
      self,
      event_type: str,
      actor: str,
      summary: str,
      details: dict[str, Any] | None = None,
  ) -> None:
    """Append an event to the log. Must be called while self._lock is held."""
    self._event_log.append({
        'timestamp': self._current_timestamp,
        'seq': self._event_seq,
        'event_type': event_type,
        'actor': actor,
        'summary': summary,
        'details': details or {},
    })
    self._event_seq += 1

  def _record_event(
      self,
      event_type: str,
      actor: str,
      summary: str,
      details: dict[str, Any] | None = None,
  ) -> None:
    """Thread-safe wrapper: append an event to the log."""
    with self._lock:
      self._record_event_locked(
          event_type=event_type,
          actor=actor,
          summary=summary,
          details=details,
      )

  def drain_event_log(self) -> list[dict[str, Any]]:
    """Atomically return all events since the last drain."""
    with self._lock:
      new_events = self._event_log[self._event_log_drain_cursor :]
      self._event_log_drain_cursor = len(self._event_log)
      return list(new_events)

  def get_full_event_log(self) -> list[dict[str, Any]]:
    """Return a copy of the complete event log (does not drain)."""
    with self._lock:
      return list(self._event_log)

  def _process_ban_expirations_locked(self, current_timestamp: str) -> None:
    """Decrement ban counters and reinstate expired bans.

    Must be called while self._lock is held.

    Args:
      current_timestamp: The current timestamp.
    """
    expired = []
    for player_name, ban_info in self._bans.items():
      ban_info['remaining_changes'] -= 1
      if ban_info['remaining_changes'] <= 0:
        expired.append(player_name)

    for player_name in expired:
      ban_info = self._bans.pop(player_name)
      banned_at = ban_info.get('banned_at_timestamp', 'unknown')
      reinstatement_msg = (
          f'{player_name} has been automatically reinstated'
          f' (banned at [{banned_at}], reinstated at'
          f' [{current_timestamp}]).'
      )
      self._record_event_locked(
          event_type='reinstatement',
          actor='[system]',
          summary=reinstatement_msg,
          details={
              'player': player_name,
              'banned_at': banned_at,
          },
      )
      if player_name in self._notification_queue:
        self._notification_queue[player_name].append(
            '[REINSTATEMENT] You have been automatically reinstated.'
            f' You were banned at [{banned_at}] and reinstated at'
            f' [{current_timestamp}]. You may now post again.'
        )

  def temp_ban(
      self,
      target: str,
      moderator: str,
      public_note: str = '',
      private_note: str = '',
  ) -> str:
    """Temporarily ban a player. Only moderators may issue bans.

    The ban lasts for ``self._temp_ban_duration`` timestamp changes.
    A public post is created on the forum with the public note, and a
    private notification with the private note is sent to the banned player.

    Args:
      target: The name of the player to ban.
      moderator: The name of the moderator issuing the ban.
      public_note: A note visible to all players (posted to the forum).
      private_note: A note sent privately to the banned player.

    Returns:
      A result string describing the outcome.
    """
    with self._lock:
      if moderator not in self._moderators:
        return (
            f'{moderator} attempted to temporarily ban {target} but they'
            ' are not a moderator. Only moderators can ban users.'
        )
      resolved_target = self._aliases.get(target, target)
      if resolved_target not in self._notification_queue:
        return (
            f'{moderator} attempted to temporarily ban "{target}" but'
            ' they are not a recognised user. Known users:'
            f' {self._player_names}'
        )
      if resolved_target in self._bans:
        return (
            f'{moderator} attempted to temporarily ban {resolved_target}'
            ' but they are already banned.'
        )
      ts = self._current_timestamp
      self._bans[resolved_target] = {
          'moderator': moderator,
          'banned_at_timestamp': ts,
          'remaining_changes': self._temp_ban_duration,
      }

    # Create a public post outside the lock (create_post acquires it).
    title = (
        f'[MODERATOR ACTION] {moderator} has temporarily banned'
        f' {resolved_target}'
    )
    content = (
        public_note or f'{moderator} has temporarily banned {resolved_target}.'
    )
    self.create_post(author=moderator, title=title, content=content)

    # Queue private notification to the banned player.
    with self._lock:
      ts = self._current_timestamp
      ts_str = f' [{ts}]' if ts else ''
      self._notification_queue[resolved_target].append(
          f'[BAN NOTICE]{ts_str} You have been temporarily banned by'
          f' {moderator}. You will be unable to post until the time'
          f' advances. Private message from {moderator}: {private_note}'
      )

    result = f'{moderator} temporarily banned {resolved_target}.'
    self._record_event(
        event_type='temp_ban',
        actor=moderator,
        summary=result,
        details={
            'moderator': moderator,
            'target': resolved_target,
            'public_note': public_note,
        },
    )
    return result

  def is_banned(self, player_name: str) -> bool:
    """Check whether a player is currently temporarily banned."""
    with self._lock:
      return player_name in self._bans

  def get_banned_players(self) -> set[str]:
    """Return the set of currently banned player names."""
    with self._lock:
      return set(self._bans.keys())

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    return ''

  def create_post(
      self,
      author: str,
      title: str,
      content: str,
      timestamp: str = '',
      image: str | None = None,
  ) -> int:
    with self._lock:
      post_id = self._next_post_id
      self._next_post_id += 1
      self._activity_seq += 1
      self._posts[post_id] = Post(
          post_id=post_id,
          author=author,
          title=title,
          content=content,
          timestamp=timestamp
          or self._current_timestamp
          or datetime.datetime.now().isoformat(),
          image=image,
          last_activity_seq=self._activity_seq,
      )
      return post_id

  def reply_to_post(
      self,
      post_id: int,
      author: str,
      content: str,
      timestamp: str = '',
      image: str | None = None,
  ) -> int | None:
    with self._lock:
      if post_id not in self._posts:
        return None
      reply_id = self._next_reply_id
      self._next_reply_id += 1
      self._activity_seq += 1
      self._posts[post_id].last_activity_seq = self._activity_seq
      self._posts[post_id].replies.append({
          'reply_id': reply_id,
          'author': author,
          'content': content,
          'timestamp': (
              timestamp
              or self._current_timestamp
              or datetime.datetime.now().isoformat()
          ),
          'image': image,
          'votes': 0,
      })
      return reply_id

  def upvote(self, post_id: int, voter: str = '') -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      self._activity_seq += 1
      self._posts[post_id].last_activity_seq = self._activity_seq
      self._posts[post_id].votes += 1
      self._posts[post_id].vote_log.append({'voter': voter, 'direction': 'up'})
      author = self._posts[post_id].author
      if voter and voter != author and author in self._karma:
        self._karma[author] += 1
      if voter and voter != author and author in self._notification_queue:
        title = self._posts[post_id].title
        ts = self._current_timestamp
        ts_str = f' [{ts}]' if ts else ''
        self._notification_queue[author].append(
            f'{voter} upvoted your post #{post_id} ("{title}").{ts_str}'
        )
      return True

  def downvote(self, post_id: int, voter: str = '') -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      self._activity_seq += 1
      self._posts[post_id].last_activity_seq = self._activity_seq
      self._posts[post_id].votes -= 1
      self._posts[post_id].vote_log.append(
          {'voter': voter, 'direction': 'down'}
      )
      author = self._posts[post_id].author
      if voter and voter != author and author in self._karma:
        self._karma[author] -= 1
      if voter and voter != author and author in self._notification_queue:
        title = self._posts[post_id].title
        ts = self._current_timestamp
        ts_str = f' [{ts}]' if ts else ''
        self._notification_queue[author].append(
            f'{voter} downvoted your post #{post_id} ("{title}").{ts_str}'
        )
      return True

  def upvote_reply(self, post_id: int, reply_id: int, voter: str = '') -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      for reply in self._posts[post_id].replies:
        if reply['reply_id'] == reply_id:
          self._activity_seq += 1
          self._posts[post_id].last_activity_seq = self._activity_seq
          reply['votes'] = reply.get('votes', 0) + 1
          if 'vote_log' not in reply:
            reply['vote_log'] = []
          reply['vote_log'].append({'voter': voter, 'direction': 'up'})  # pytype: disable=attribute-error
          author = str(reply['author'])
          if voter and voter != author and author in self._karma:
            self._karma[author] += 1
          if voter and voter != author and author in self._notification_queue:
            snippet = str(reply['content'])[:60]
            ts = self._current_timestamp
            ts_str = f' [{ts}]' if ts else ''
            self._notification_queue[author].append(
                f'{voter} upvoted your reply #{reply_id} on post'
                f' #{post_id} ("{snippet}...").{ts_str}'
            )
          return True
      return False

  def downvote_reply(
      self, post_id: int, reply_id: int, voter: str = ''
  ) -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      for reply in self._posts[post_id].replies:
        if reply['reply_id'] == reply_id:
          self._activity_seq += 1
          self._posts[post_id].last_activity_seq = self._activity_seq
          reply['votes'] = reply.get('votes', 0) - 1
          if 'vote_log' not in reply:
            reply['vote_log'] = []
          reply['vote_log'].append({'voter': voter, 'direction': 'down'})  # pytype: disable=attribute-error
          author = str(reply['author'])
          if voter and voter != author and author in self._karma:
            self._karma[author] -= 1
          if voter and voter != author and author in self._notification_queue:
            snippet = str(reply['content'])[:60]
            ts = self._current_timestamp
            ts_str = f' [{ts}]' if ts else ''
            self._notification_queue[author].append(
                f'{voter} downvoted your reply #{reply_id} on'
                f' post #{post_id} ("{snippet}...").{ts_str}'
            )
          return True
      return False

  def get_karma_summary(self) -> str:
    with self._lock:
      entries = [f'{name}: {score}' for name, score in self._karma.items()]
      return 'Karma scores: ' + ', '.join(entries)

  def get_karma(self) -> dict[str, int]:
    """Returns a copy of the current karma scores."""
    with self._lock:
      return dict(self._karma)

  def get_recent_posts(self, n: int | None = None) -> list[Post]:
    with self._lock:
      all_posts = sorted(
          self._posts.values(), key=lambda p: p.post_id, reverse=True
      )
      if n is not None:
        return all_posts[:n]
      return all_posts

  def send_direct_message(
      self, sender: str, recipient: str, content: str
  ) -> str:
    """Send a private direct message from one player to another.

    The message is delivered as a notification to the recipient, so they
    will see it on their next observation step.  It is *not* visible on
    the public forum or in the HTML visualisation.

    Args:
      sender: The name of the sending player.
      recipient: The name of the receiving player.
      content: The message body.

    Returns:
      A result string describing the outcome.
    """
    with self._lock:
      resolved_recipient = self._aliases.get(recipient, recipient)
      if resolved_recipient not in self._notification_queue:
        return (
            f'{sender} attempted to send direct message to "{recipient}" but'
            ' they are not a recognised user. Known users:'
            f' {self._player_names}'
        )
      ts = self._current_timestamp
      ts_str = f' [{ts}]' if ts else ''
      self._notification_queue[resolved_recipient].append(
          f'[Direct message from {sender}]{ts_str}: {content}'
      )
      # Persist in direct message thread history.
      pair_key = self._direct_message_pair_key(sender, resolved_recipient)
      if pair_key not in self._direct_message_threads:
        self._direct_message_threads[pair_key] = []
      self._direct_message_threads[pair_key].append({
          'sender': sender,
          'content': content,
          'timestamp': ts,
      })
      result = (
          f'{sender} sent a direct message to {resolved_recipient}: "{content}"'
      )
      self._record_event_locked(
          event_type='direct_message',
          actor=sender,
          summary=result,
          details={
              'sender': sender,
              'recipient': resolved_recipient,
              'content': content,
          },
      )
      return result

  @staticmethod
  def _direct_message_pair_key(a: str, b: str) -> str:
    """Canonical key for a direct message thread between two players."""
    return '|'.join(sorted([a, b]))

  def get_direct_message_threads_for_player(
      self, player_name: str
  ) -> dict[str, int]:
    """Returns direct message thread partners and message counts for a player.

    Args:
      player_name: The player whose direct message threads to list.

    Returns:
      A dict mapping partner name -> message count.
    """
    with self._lock:
      result: dict[str, int] = {}
      for pair_key, messages in self._direct_message_threads.items():
        parts = str(pair_key).split('|')
        if player_name in parts:
          partner = parts[0] if parts[1] == player_name else parts[1]
          result[partner] = len(messages)
      return result

  def get_direct_message_thread(
      self, player_a: str, player_b: str
  ) -> list[dict[str, str]]:
    """Returns the full direct message thread between two players.

    Args:
      player_a: One participant.
      player_b: The other participant.

    Returns:
      A list of message dicts with sender, content, and timestamp.
    """
    with self._lock:
      pair_key = self._direct_message_pair_key(player_a, player_b)
      return list(self._direct_message_threads.get(pair_key, []))

  def pin_post(self, post_id: int, moderator: str) -> str:
    """Pin a post. Only moderators may pin posts.

    Args:
      post_id: The ID of the post to pin.
      moderator: The name of the user attempting to pin.

    Returns:
      A result string describing the outcome.
    """
    with self._lock:
      if moderator not in self._moderators:
        return (
            f'{moderator} attempted to pin post #{post_id} but they are'
            ' not a moderator. Only moderators can pin posts.'
        )
      if post_id not in self._posts:
        available = sorted(self._posts.keys())
        return (
            f'{moderator} attempted to pin post #{post_id} but it does'
            f' not exist. Available posts are: {available}.'
        )
      self._pinned_post_id = post_id
      post = self._posts[post_id]
      reply_count = len(post.replies)
      notification = (
          f'{moderator} pinned post #{post_id}: "{post.title}"'
          f' (votes: {post.votes}, replies: {reply_count})'
      )
      for name in self._player_names:
        self._notification_queue[name].append(notification)
      self._record_event_locked(
          event_type='pin_post',
          actor=moderator,
          summary=notification,
          details={'post_id': post_id, 'title': post.title},
      )
      return notification

  def get_pinned_post_summary(self) -> str:
    """Return a summary of the currently pinned post.

    Includes post ID, title, vote total, reply count, original author
    and snippet, and latest reply author and snippet.

    Returns:
      A human-readable summary, or a message indicating no post is pinned.
    """
    with self._lock:
      if self._pinned_post_id is None:
        return 'No post is currently pinned.'
      post = self._posts.get(self._pinned_post_id)
      if post is None:
        return 'No post is currently pinned.'
      reply_count = len(post.replies)
      snippet = self._short_desc(post.content, max_len=80)
      lines = [
          (
              f'Pinned post: [Post #{post.post_id}] "{post.title}"'
              f' by {post.author} (votes: {post.votes},'
              f' replies: {reply_count})'
          ),
          f'  Original post by {post.author}: "{snippet}"',
      ]
      if post.replies:
        latest_reply = post.replies[-1]
        reply_snippet = self._short_desc(
            str(latest_reply['content']), max_len=80
        )
        lines.append(
            f'  Latest reply by {latest_reply["author"]}: "{reply_snippet}"'
        )
      return '\n'.join(lines)

  def get_pinned_post_id(self) -> int | None:
    """Return the ID of the currently pinned post, or None."""
    with self._lock:
      return self._pinned_post_id

  def queue_notification(self, player_name: str, message: str) -> None:
    with self._lock:
      if player_name in self._notification_queue:
        self._notification_queue[player_name].append(message)

  def drain_notifications(self, player_name: str) -> list[str]:
    with self._lock:
      messages = list(self._notification_queue.get(player_name, []))
      if player_name in self._notification_queue:
        self._notification_queue[player_name] = []
      return messages

  def _format_post_summary(self, post: Post) -> str:
    reply_count = len(post.replies)
    ts = post.timestamp
    ts_str = f' [{ts}]' if ts else ''
    summary = (
        f'[Post #{post.post_id}] "{post.title}" by {post.author}'
        f' (votes: {post.votes}, replies: {reply_count}){ts_str}'
    )
    if post.content and post.content != post.title:
      summary += f'\n  {post.content}'
    if post.replies:
      for reply in post.replies:
        reply_votes = reply.get('votes', 0)
        rid = reply['reply_id']
        r_ts = reply.get('timestamp', '')
        r_ts_str = f' [{r_ts}]' if r_ts else ''
        summary += (
            f'\n  [Reply #{rid}] by {reply["author"]}'
            f' (votes: {reply_votes}){r_ts_str}: "{reply["content"]}"'
        )
    return summary

  def get_forum_summary(self) -> str:
    posts = self.get_recent_posts(self._max_summary_posts)
    if not posts:
      return f'{self._forum_name}: No posts yet.'
    lines = [f'{self._forum_name} — Recent posts:']
    for post in posts:
      lines.append(self._format_post_summary(post))
    return '\n\n\n'.join(lines)

  def get_vote_summary(self) -> str:
    with self._lock:
      posts = sorted(self._posts.values(), key=lambda p: p.post_id)
      if not posts:
        return ''
      entries = [f'Post #{p.post_id}: {p.votes} votes' for p in posts]
      return f'{self._forum_name} vote counts: ' + ', '.join(entries)

  def _short_desc(self, text: str, max_len: int = 40) -> str:
    if len(text) <= max_len:
      return text
    return text[:max_len] + '...'

  def get_vote_changes_for_player(self, player_name: str) -> str:
    with self._lock:
      current_votes = {}
      for pid, post in self._posts.items():
        current_votes[f'post_{pid}'] = post.votes
        for reply in post.replies:
          rid = reply['reply_id']
          current_votes[f'reply_{pid}_{rid}'] = reply.get('votes', 0)

      prev = self._last_seen_votes.get(player_name, {})
      self._last_seen_votes[player_name] = dict(current_votes)

      if not prev:
        return ''

      changed_posts = {}
      for key, cur_val in current_votes.items():
        old_val = prev.get(key, 0)
        if cur_val != old_val:
          delta = cur_val - old_val
          changed_posts[key] = delta

      if not changed_posts:
        return ''

      by_post: dict[int, list[str]] = {}
      post_deltas: dict[int, int] = {}

      for key, delta in changed_posts.items():
        sign = '+' if delta > 0 else ''
        if key.startswith('post_'):
          pid = int(key.split('_')[1])
          post_deltas[pid] = delta
          if pid not in by_post:
            by_post[pid] = []
        elif key.startswith('reply_'):
          parts = key.split('_')
          pid = int(parts[1])
          rid = int(parts[2])
          if pid not in by_post:
            by_post[pid] = []
          post = self._posts.get(pid)
          if post:
            for r in post.replies:
              if r['reply_id'] == rid:
                desc = self._short_desc(r['content'])
                by_post[pid].append(
                    f'reply {rid} by {r["author"]} ({desc}): {sign}{delta}'
                )
                break

      lines = ['New votes:']
      for pid in sorted(by_post.keys()):
        post = self._posts.get(pid)
        if not post:
          continue
        post_desc = self._short_desc(post.title)
        parts_list = []
        if pid in post_deltas:
          d = post_deltas[pid]
          s = '+' if d > 0 else ''
          parts_list.append(f'post: {s}{d}')
        parts_list.extend(by_post[pid])
        lines.append(
            f'- post {pid} by {post.author} ({post_desc})'
            f' -- {", ".join(parts_list)}'
        )

      if len(lines) == 1:
        return ''
      return '\n'.join(lines)

  def get_forum_summary_for_player(self, player_name: str) -> str:
    with self._lock:
      last_post = self._last_seen_post_id.get(player_name, -1)
      last_reply = self._last_seen_reply_id.get(player_name, -1)
      new_posts = [p for p in self._posts.values() if p.post_id > last_post]
      updated_posts = [
          p
          for p in self._posts.values()
          if p.post_id <= last_post
          and any(r['reply_id'] > last_reply for r in p.replies)
      ]
      if new_posts:
        self._last_seen_post_id[player_name] = max(p.post_id for p in new_posts)
      all_new_reply_ids = []
      for p in self._posts.values():
        for r in p.replies:
          if r['reply_id'] > last_reply:
            all_new_reply_ids.append(r['reply_id'])
      if all_new_reply_ids:
        self._last_seen_reply_id[player_name] = int(max(all_new_reply_ids))
      if not new_posts and not updated_posts:
        return f'{self._forum_name}: No new activity.'
      lines = []
      if new_posts:
        new_posts.sort(key=lambda p: p.post_id, reverse=True)
        for post in new_posts:
          lines.append(self._format_post_summary(post))
      if updated_posts:
        updated_posts.sort(key=lambda p: p.post_id, reverse=True)
        for post in updated_posts:
          new_replies = [r for r in post.replies if r['reply_id'] > last_reply]
          for r in new_replies:
            reply_votes = r.get('votes', 0)
            rid = r['reply_id']
            r_ts = r.get('timestamp', '')
            r_ts_str = f' [{r_ts}]' if r_ts else ''
            lines.append(
                f'  New [reply #{rid}] to post #{post.post_id}'
                f' by {r["author"]}'
                f' (votes: {reply_votes}){r_ts_str}: "{r["content"]}"'
            )
      return '\n\n\n'.join(lines)

  def extract_json(self, text: str) -> dict[str, Any] | None:
    # Replace curly quotes
    text = (
        text.replace('\u201c', '"')
        .replace('\u201d', '"')
        .replace('\u2018', "'")
        .replace('\u2019', "'")
    )

    fence_match = re.search(
        r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL
    )
    if fence_match:
      try:
        return json.loads(fence_match.group(1))
      except json.JSONDecodeError:
        pass

    # Try to find the largest JSON-like substring
    brace_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if brace_match:
      try:
        return json.loads(brace_match.group(1))
      except json.JSONDecodeError:
        pass

    return None

  def _parse_post_id(self, raw_value: Any) -> int:
    try:
      return int(str(raw_value).strip().lstrip('#'))
    except (ValueError, TypeError):
      return -1

  def _get_reply_snippet(self, post_id: int, reply_id: int) -> str:
    """Return a short snippet of a reply's content for event summaries."""
    post = self._posts.get(post_id)
    if post is None:
      return '?'
    for reply in post.replies:
      if reply.get('reply_id') == reply_id:
        content = str(reply.get('content', ''))
        if len(content) > 80:
          return content[:80] + '...'
        return content
    return '?'

  def parse_and_execute_action(
      self, action_text: str, entity_name: str | None = None
  ) -> str:
    """Parse a JSON action and execute it on the forum.

    Args:
      action_text: The raw action text (JSON) from the player entity.
      entity_name: The account name of the player entity submitting the action.
        If provided and the author field in the action does not match, the
        author is tagged with [UNVERIFIED].

    Returns:
      A string describing the result of the action.
    """
    action = None
    image_data = None

    try:
      parsed = json.loads(action_text.strip())
      if isinstance(parsed, dict):
        if 'text' in parsed and 'image' in parsed:
          image_data = parsed.get('image')
          if image_data == 'FAILED TO MAKE AN IMAGE':
            image_data = None
          action = self.extract_json(parsed['text'])
        elif 'action' in parsed:
          action = parsed
    except (json.JSONDecodeError, ValueError):
      pass

    if action is None:
      action = self.extract_json(action_text)

    if action is None:
      actor_name = entity_name or 'Unknown'
      result = (
          f'{actor_name} attempted to act but it could not be parsed.'
          f' Expected valid JSON. Got: "{action_text}"'
      )
      self._record_event(
          event_type='parse_error',
          actor=actor_name,
          summary=result,
      )
      return result

    action_type = action.get('action', '')
    raw_author = action.get('author', 'Unknown')
    author = raw_author

    # Verify that the author field matches the entity's account name.
    # If it doesn't match, tag the author as unverified so that
    # impersonation attempts are visible in all views of this data.
    if entity_name is not None and author != entity_name:
      author = f'{author} [UNVERIFIED]'

    actor_name = entity_name or author

    # Karma lookup uses the raw author from the action JSON.
    # Unknown authors ("sock puppets") are treated as having karma 0.
    author_karma = self._karma.get(raw_author, 0)

    if action_type == 'post':
      if author_karma < self._min_karma_to_post:
        deficit = self._min_karma_to_post - author_karma
        result = (
            f'{raw_author} attempted to create a post but has insufficient'
            f' karma ({author_karma}). Creating a new thread requires at'
            f' least {self._min_karma_to_post} karma. {raw_author} needs'
            f' to gain {deficit} more karma point(s) to create threads.'
        )
        self.queue_notification(actor_name, result)
        return result
      title = action.get('title', '')
      content = action.get('content', '')
      post_id = self.create_post(
          author=author,
          title=title,
          content=content,
          image=image_data,
      )
      result = f'{author} created post #{post_id}: "{title}"'
      self.queue_notification(author, result)
      self._record_event(
          event_type='post',
          actor=actor_name,
          summary=result,
          details={'post_id': post_id, 'title': title, 'content': content},
      )
      return result
    elif action_type == 'reply':
      post_id = self._parse_post_id(action.get('post_id', -1))
      content = action.get('content', '')
      # Check per-post minimum karma to reply.
      with self._lock:
        target_post = self._posts.get(post_id)
      if target_post is not None:
        min_karma = target_post.min_karma_to_reply
        if author_karma < min_karma:
          deficit = min_karma - author_karma
          result = (
              f'{raw_author} attempted to reply to post #{post_id} but has'
              f' insufficient karma ({author_karma}). Replying to this'
              f' thread requires at least {min_karma} karma. {raw_author}'
              f' needs to gain {deficit} more karma point(s) to reply to'
              ' this thread.'
          )
          self.queue_notification(actor_name, result)
          return result
      reply_id = self.reply_to_post(
          post_id=post_id,
          author=author,
          content=content,
          image=image_data,
      )
      if reply_id is not None:
        result = f'{author} replied to post #{post_id}: "{content}"'
        self.queue_notification(author, result)
        self._record_event(
            event_type='reply',
            actor=actor_name,
            summary=result,
            details={
                'post_id': post_id,
                'reply_id': reply_id,
                'content': content,
            },
        )
        return result
      else:
        title = content[:80] + ('...' if len(content) > 80 else '')
        new_post_id = self.create_post(
            author=author, title=title, content=content
        )
        result = f'{author} created post #{new_post_id}: "{title}"'
        self.queue_notification(author, result)
        self._record_event(
            event_type='post',
            actor=actor_name,
            summary=result,
            details={
                'post_id': new_post_id,
                'title': title,
                'content': content,
                'fallback_from_reply': True,
            },
        )
        return result
    elif action_type == 'upvote_post':
      post_id = self._parse_post_id(action.get('post_id', -1))
      success = self.upvote(post_id, voter=author)
      if success:
        result = f'{author} upvoted post #{post_id}'
        self.queue_notification(author, result)
        post_title = self._posts.get(post_id, Post(0, '', '', '', '')).title
        self._record_event(
            event_type='upvote_post',
            actor=actor_name,
            summary=f'{result}: "{post_title}"',
            details={'post_id': post_id, 'title': post_title},
        )
        return result
      else:
        available = sorted(self._posts.keys())
        result = (
            f'{actor_name} attempted to upvote post #{post_id} but it does not'
            f' exist. Available posts are: {available}.'
        )
        self.queue_notification(author, result)
        return result
    elif action_type == 'downvote_post':
      post_id = self._parse_post_id(action.get('post_id', -1))
      success = self.downvote(post_id, voter=author)
      if success:
        result = f'{author} downvoted post #{post_id}'
        self.queue_notification(author, result)
        post_title = self._posts.get(post_id, Post(0, '', '', '', '')).title
        self._record_event(
            event_type='downvote_post',
            actor=actor_name,
            summary=f'{result}: "{post_title}"',
            details={'post_id': post_id, 'title': post_title},
        )
        return result
      else:
        available = sorted(self._posts.keys())
        result = (
            f'{actor_name} attempted to downvote post #{post_id} but it does'
            f' not exist. Available posts are: {available}.'
        )
        self.queue_notification(author, result)
        return result
    elif action_type == 'upvote_reply':
      post_id = self._parse_post_id(action.get('post_id', -1))
      reply_id = self._parse_post_id(action.get('reply_id', -1))
      success = self.upvote_reply(post_id, reply_id, voter=author)
      if success:
        result = f'{author} upvoted reply #{reply_id} on post #{post_id}'
        self.queue_notification(author, result)
        reply_snippet = self._get_reply_snippet(post_id, reply_id)
        self._record_event(
            event_type='upvote_reply',
            actor=actor_name,
            summary=f'{result}: "{reply_snippet}"',
            details={
                'post_id': post_id,
                'reply_id': reply_id,
                'reply_snippet': reply_snippet,
            },
        )
        return result
      else:
        result = (
            f'{actor_name} attempted to upvote reply #{reply_id} on post'
            f' #{post_id} but it does not exist.'
        )
        self.queue_notification(author, result)
        return result
    elif action_type == 'downvote_reply':
      post_id = self._parse_post_id(action.get('post_id', -1))
      reply_id = self._parse_post_id(action.get('reply_id', -1))
      success = self.downvote_reply(post_id, reply_id, voter=author)
      if success:
        result = f'{author} downvoted reply #{reply_id} on post #{post_id}'
        self.queue_notification(author, result)
        reply_snippet = self._get_reply_snippet(post_id, reply_id)
        self._record_event(
            event_type='downvote_reply',
            actor=actor_name,
            summary=f'{result}: "{reply_snippet}"',
            details={
                'post_id': post_id,
                'reply_id': reply_id,
                'reply_snippet': reply_snippet,
            },
        )
        return result
      else:
        result = (
            f'{actor_name} attempted to downvote reply #{reply_id} on post'
            f' #{post_id} but it does not exist.'
        )
        self.queue_notification(author, result)
        return result
    elif action_type == 'direct_message':
      if author_karma < self._min_karma_to_direct_message:
        deficit = self._min_karma_to_direct_message - author_karma
        result = (
            f'{raw_author} attempted to send a direct message but has'
            f' insufficient karma ({author_karma}). Sending direct messages'
            f' requires at least {self._min_karma_to_direct_message} karma.'
            f' {raw_author} needs to gain {deficit} more karma point(s) to send'
            ' direct messages.'
        )
        self.queue_notification(actor_name, result)
        return result
      recipient = action.get('recipient', '')
      content = action.get('content', '')
      result = self.send_direct_message(
          sender=actor_name, recipient=recipient, content=content
      )
      self.queue_notification(author, result)
      return result
    elif action_type == 'pin_post':
      post_id = self._parse_post_id(action.get('post_id', -1))
      result = self.pin_post(post_id=post_id, moderator=actor_name)
      self.queue_notification(author, result)
      return result
    elif action_type == 'temp_ban':
      target = action.get('target', '')
      public_note = action.get('public_note', '')
      private_note = action.get('private_note', '')
      result = self.temp_ban(
          target=target,
          moderator=actor_name,
          public_note=public_note,
          private_note=private_note,
      )
      self.queue_notification(author, result)
      return result
    else:
      result = (
          f'{actor_name} attempted unknown action type "{action_type}".'
          ' Expected one of: post, reply, upvote_post, downvote_post,'
          ' upvote_reply, downvote_reply, direct_message, pin_post,'
          ' temp_ban.'
      )
      self.queue_notification(author, result)
      self._record_event(
          event_type='unknown_action',
          actor=actor_name,
          summary=result,
          details={'action_type': action_type},
      )
      return result

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      posts_state = {}
      for pid, post in self._posts.items():
        posts_state[str(pid)] = dataclasses.asdict(post)
      return {
          'posts': posts_state,
          'next_post_id': self._next_post_id,
          'next_reply_id': self._next_reply_id,
          'karma': dict(self._karma),
          'last_seen_votes': {
              k: dict(v) for k, v in self._last_seen_votes.items()
          },
          'pinned_post_id': self._pinned_post_id,
          'moderators': list(self._moderators),
          'bans': {k: dict(v) for k, v in self._bans.items()},
          'timestamp_change_count': self._timestamp_change_count,
          'temp_ban_duration': self._temp_ban_duration,
          'direct_message_threads': {
              k: list(v) for k, v in self._direct_message_threads.items()
          },
          'min_karma_to_post': self._min_karma_to_post,
          'min_karma_to_direct_message': self._min_karma_to_direct_message,
          'event_log': list(self._event_log),
          'event_seq': self._event_seq,
          'event_log_drain_cursor': self._event_log_drain_cursor,
          'activity_seq': self._activity_seq,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._posts = {}
      posts_data = dict(state.get('posts', {}))  # pyrefly: ignore[no-matching-overload]
      for pid_str, post_data in posts_data.items():
        self._posts[int(pid_str)] = Post(**post_data)
      self._next_post_id = int(state.get('next_post_id', 0))  # pyrefly: ignore[bad-argument-type]
      self._next_reply_id = int(state.get('next_reply_id', 0))  # pyrefly: ignore[bad-argument-type]
      karma_data = state.get('karma', {})
      if karma_data:
        self._karma = {str(k): int(v) for k, v in dict(karma_data).items()}  # pyrefly: ignore[no-matching-overload]
      votes_data = state.get('last_seen_votes', {})
      if votes_data:
        self._last_seen_votes = {
            str(k): {str(vk): int(vv) for vk, vv in dict(v).items()}
            for k, v in dict(votes_data).items()  # pyrefly: ignore[no-matching-overload]
        }
      pinned = state.get('pinned_post_id', None)
      self._pinned_post_id = int(pinned) if pinned is not None else None  # pyrefly: ignore[bad-argument-type]
      moderators_data = state.get('moderators', [])
      if moderators_data:
        self._moderators = [str(m) for m in moderators_data]  # pyrefly: ignore[not-iterable]
      bans_data = state.get('bans', {})
      if bans_data:
        self._bans = {str(k): dict(v) for k, v in dict(bans_data).items()}  # pyrefly: ignore[no-matching-overload]
      else:
        self._bans = {}
      self._timestamp_change_count = int(state.get('timestamp_change_count', 0))  # pyrefly: ignore[bad-argument-type]
      direct_message_data = state.get('direct_message_threads', {})
      if direct_message_data:
        self._direct_message_threads = {
            str(k): [dict(m) for m in v]
            for k, v in dict(direct_message_data).items()  # pyrefly: ignore[no-matching-overload]
        }
      else:
        self._direct_message_threads = {}
      self._temp_ban_duration = int(state.get('temp_ban_duration', 1))  # pyrefly: ignore[bad-argument-type]
      self._min_karma_to_post = int(state.get('min_karma_to_post', -1))  # pyrefly: ignore[bad-argument-type]
      self._min_karma_to_direct_message = int(
          state.get('min_karma_to_direct_message', 1)  # pyrefly: ignore[bad-argument-type]
      )
      event_log_data = state.get('event_log', [])
      self._event_log = (
          [dict(e) for e in event_log_data] if event_log_data else []  # pyrefly: ignore[no-matching-overload, not-iterable]
      )
      self._event_seq = int(state.get('event_seq', len(self._event_log)))  # pyrefly: ignore[bad-argument-type]
      self._event_log_drain_cursor = int(state.get('event_log_drain_cursor', 0))  # pyrefly: ignore[bad-argument-type]
      self._activity_seq = int(state.get('activity_seq', 0))  # pyrefly: ignore[bad-argument-type]

  def _escape_html(self, text: str) -> str:
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace('\n', '<br>')
    )

  def _extract_image_src(self, image_markdown: str) -> str:
    match = re.search(r'!\[[^\]]*\]\(([^)]+)\)', image_markdown)
    if match:
      return match.group(1)
    return ''

  def _render_post_image(self, post: Post) -> str:
    if post.image and post.image.startswith('!['):
      src = self._extract_image_src(post.image)
      if src:
        return (
            f'<div class="post-image"><img src="{src}" alt="post image"></div>'
        )
    return ''

  def to_html(self, title: str = '') -> str:
    title = title or self._forum_name
    posts = self.get_recent_posts()
    posts_sorted = sorted(posts, key=lambda p: p.post_id)

    posts_html = ''
    if not posts_sorted:
      posts_html = (
          '<div class="empty">No posts yet. '
          'The forum is waiting for its first post!</div>'
      )
    else:
      for post in posts_sorted:
        vote_class = ''
        if post.votes > 0:
          vote_class = ' positive'
        elif post.votes < 0:
          vote_class = ' negative'

        replies_html = ''
        for reply in post.replies:
          reply_image_html = ''
          reply_image = reply.get('image')
          if reply_image and reply_image.startswith('!['):
            reply_image_html = (
                '<div class="post-image"><img src="'
                f'{self._extract_image_src(reply_image)}"'
                ' alt="reply image"></div>'
            )
          reply_votes = reply.get('votes', 0)
          rv_class = ''
          if reply_votes > 0:
            rv_class = ' positive'
          elif reply_votes < 0:
            rv_class = ' negative'
          replies_html += f"""
          <div class="reply">
            <div class="reply-vote-column">
              <span class="vote-arrow up">\u25b2</span>
              <span class="vote-count{rv_class}">{reply_votes}</span>
              <span class="vote-arrow down">\u25bc</span>
            </div>
            <div class="reply-body">
              <div class="reply-meta">
                <span class="author{' unverified' if '[UNVERIFIED]' in str(reply['author']) else ''}">{self._escape_html(reply['author'])}</span>
                <span class="timestamp">{self._escape_html(reply.get('timestamp', ''))}</span>
              </div>
              <div class="reply-content">{self._escape_html(reply['content'])}</div>
              {reply_image_html}
            </div>
          </div>"""

        reply_count = len(post.replies)
        reply_label = f'{reply_count} repl{"ies" if reply_count != 1 else "y"}'

        posts_html += f"""
        <div class="post">
          <div class="vote-column">
            <div class="vote-arrow up">\u25b2</div>
            <div class="vote-count{vote_class}">{post.votes}</div>
            <div class="vote-arrow down">\u25bc</div>
          </div>
          <div class="post-content">
            <div class="post-title">{self._escape_html(post.title)}</div>
            <div class="post-meta">
              Posted by <span class="author{' unverified' if '[UNVERIFIED]' in post.author else ''}">{self._escape_html(post.author)}</span>
              <span class="timestamp">{self._escape_html(post.timestamp)}</span>
            </div>
            <div class="post-body">{self._escape_html(post.content)}</div>
            {self._render_post_image(post)}
            <div class="post-actions">
              <span class="action-item">\U0001f4ac {reply_label}</span>
            </div>
            {f'<div class="replies">{replies_html}</div>' if post.replies else ''}
          </div>
        </div>"""

    stats_line = f'{len(posts_sorted)} posts'
    total_replies = sum(len(p.replies) for p in posts_sorted)
    if total_replies:
      stats_line += f' \u00b7 {total_replies} replies'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self._escape_html(title)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                   'Helvetica Neue', Arial, sans-serif;
      background: #1a1a1b;
      color: #d7dadc;
      line-height: 1.5;
    }}
    .header {{
      background: #1a1a2e;
      border-bottom: 3px solid #3b82f6;
      padding: 16px 0;
    }}
    .header-inner {{
      max-width: 800px;
      margin: 0 auto;
      padding: 0 16px;
    }}
    .header h1 {{
      font-size: 22px;
      color: #e0e0e0;
    }}
    .header .stats {{
      font-size: 13px;
      color: #818384;
      margin-top: 4px;
    }}
    .content {{
      max-width: 800px;
      margin: 20px auto;
      padding: 0 16px;
    }}
    .post {{
      display: flex;
      background: #272729;
      border: 1px solid #343536;
      border-radius: 4px;
      margin-bottom: 12px;
      overflow: hidden;
    }}
    .post:hover {{
      border-color: #4a4a4c;
    }}
    .vote-column {{
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 8px 10px;
      background: #1e1e20;
      min-width: 42px;
    }}
    .vote-arrow {{
      color: #555;
      font-size: 14px;
      cursor: default;
      line-height: 1;
    }}
    .vote-count {{
      font-size: 13px;
      font-weight: bold;
      color: #d7dadc;
      margin: 2px 0;
    }}
    .vote-count.positive {{ color: #ff8b60; }}
    .vote-count.negative {{ color: #7193ff; }}
    .post-content {{
      padding: 10px 14px;
      flex: 1;
      min-width: 0;
    }}
    .post-title {{
      font-size: 17px;
      font-weight: 600;
      color: #d7dadc;
      margin-bottom: 4px;
    }}
    .post-meta {{
      font-size: 12px;
      color: #818384;
      margin-bottom: 8px;
    }}
    .author {{
      color: #4fbcff;
      font-weight: 500;
    }}
    .author.unverified {{
      color: #ff4444;
      font-weight: 700;
    }}
    .timestamp {{
      margin-left: 6px;
      color: #555;
      font-size: 11px;
    }}
    .post-body {{
      font-size: 14px;
      color: #c8cbcd;
      margin-bottom: 8px;
      word-wrap: break-word;
    }}
    .post-actions {{
      font-size: 12px;
      color: #818384;
      font-weight: bold;
    }}
    .post-image {{
      margin: 8px 0;
    }}
    .post-image img {{
      max-width: 100%;
      max-height: 400px;
      border-radius: 4px;
      border: 1px solid #343536;
    }}
    .action-item {{
      padding: 4px 6px;
      border-radius: 3px;
    }}
    .replies {{
      margin-top: 10px;
      border-top: 1px solid #343536;
      padding-top: 8px;
    }}
    .reply {{
      display: flex;
      gap: 8px;
      padding: 8px 10px;
      margin: 4px 0 4px 16px;
      border-left: 2px solid #3b82f6;
      background: #1e1e20;
      border-radius: 0 4px 4px 0;
    }}
    .reply-vote-column {{
      display: flex;
      flex-direction: column;
      align-items: center;
      min-width: 24px;
      font-size: 11px;
      padding-top: 2px;
    }}
    .reply-body {{
      flex: 1;
      min-width: 0;
    }}
    .reply-meta {{
      font-size: 12px;
      color: #818384;
      margin-bottom: 4px;
    }}
    .reply-content {{
      font-size: 13px;
      color: #c8cbcd;
      word-wrap: break-word;
    }}
    .empty {{
      text-align: center;
      padding: 60px 20px;
      color: #818384;
      font-size: 15px;
      background: #272729;
      border: 1px solid #343536;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-inner">
      <h1>\U0001f4cb {self._escape_html(title)}</h1>
      <div class="stats">{stats_line}</div>
    </div>
  </div>
  <div class="content">
    {posts_html}
  </div>
</body>
</html>"""


class ForumResolution(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Resolves player actions on the forum programmatically (no LLM).

  Registered under the __resolution__ key so SwitchAct uses it for RESOLVE.
  Accesses ForumState and Memory via get_entity().get_component().
  """

  def __init__(
      self,
      player_names: Sequence[str],
      forum_component_key: str = DEFAULT_FORUM_COMPONENT_KEY,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = (event_resolution.DEFAULT_RESOLUTION_PRE_ACT_LABEL),
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._forum_component_key = forum_component_key
    self._memory_component_key = memory_component_key
    self._pre_act_label = pre_act_label
    self._resolved_per_entity: dict[str, int] = {}
    self._resolution_lock = threading.Lock()
    self._active_entity_name: str | None = None
    self._putative_action: str | None = None

  def _get_forum_state(self) -> ForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=ForumState
    )

  def _get_putative_action(self) -> tuple[str | None, str | None]:
    memory = self.get_entity().get_component(
        self._memory_component_key,
        type_=memory_component.Memory,
    )
    putative_event_tag = event_resolution.PUTATIVE_EVENT_TAG
    suggestions = memory.scan(selector_fn=lambda x: putative_event_tag in x)
    if not suggestions:
      return None, None

    # Determine which entity's thread is calling. In the async engine,
    # act() sets _active_capture_key to the entity name BEFORE
    # dispatching pre_act to worker threads (via _parallel_call_).
    # Since act() holds _control_lock for the entire duration, this
    # value is stable. We read it to restrict resolution to only the
    # calling entity's putative events, preventing cross-thread
    # contamination.
    thread_entity_name = None
    game_master = self.get_entity()
    if hasattr(game_master, '_active_capture_key'):
      capture_key = game_master._active_capture_key  # pylint: disable=protected-access
      if capture_key in self._player_names:
        thread_entity_name = capture_key

    with self._resolution_lock:
      names_to_check = (
          [thread_entity_name] if thread_entity_name else self._player_names
      )
      for name in names_to_check:
        prefix = f'{putative_event_tag} {name}'
        entity_suggestions = [s for s in suggestions if prefix in s]
        resolved = self._resolved_per_entity.get(name, 0)
        if len(entity_suggestions) > resolved:
          selected = entity_suggestions[resolved]
          self._resolved_per_entity[name] = resolved + 1

          putative_action = selected[
              selected.find(putative_event_tag) + len(putative_event_tag) :
          ]
          # Strip the entity name prefix and separator.
          entity_prefix = f' {name}'
          if putative_action.startswith(entity_prefix):
            remainder = putative_action[len(entity_prefix) :]
            if remainder.startswith(':'):
              remainder = remainder[1:]
            elif remainder.startswith(' --'):
              remainder = remainder[3:]
            putative_action = remainder.strip()

          return name, putative_action

    return None, None

  def get_active_entity_name(self) -> str | None:
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    return self._putative_action

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      active_entity_name, putative_action = self._get_putative_action()
      self._active_entity_name = active_entity_name
      self._putative_action = putative_action

      if putative_action is not None:
        forum_state = self._get_forum_state()
        result = forum_state.parse_and_execute_action(
            putative_action, entity_name=active_entity_name
        )
      else:
        result = ''

      result = f'{self._pre_act_label}: {result}\n'

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {
        'resolved_per_entity': dict(self._resolved_per_entity),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    raw = state.get('resolved_per_entity', {})
    if isinstance(raw, dict):
      self._resolved_per_entity = {str(k): int(v) for k, v in raw.items()}  # pyrefly: ignore[bad-argument-type]
    else:
      self._resolved_per_entity = {}


class ForumObservation(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Returns forum observations to players programmatically (no LLM).

  Registered under the __make_observation__ key so SwitchAct uses it for
  MAKE_OBSERVATION. Accesses ForumState via get_entity().get_component().
  """

  def __init__(
      self,
      forum_component_key: str = DEFAULT_FORUM_COMPONENT_KEY,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      pre_act_label: str = '\nPrompt',
  ):
    super().__init__()
    self._forum_component_key = forum_component_key
    self._call_to_make_observation = call_to_make_observation
    self._pre_act_label = pre_act_label

  def _get_forum_state(self) -> ForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=ForumState
    )

  def _get_active_entity_name_from_call_to_action(
      self, call_to_action: str
  ) -> str:
    prefix, suffix = self._call_to_make_observation.split('{name}')
    if not call_to_action.startswith(prefix):
      raise ValueError(
          f'Call to action {call_to_action} does not start with prefix'
          f' {prefix}. Check that call_to_make_observation is set correctly.'
      )
    if not call_to_action.endswith(suffix):
      raise ValueError(
          f'Call to action {call_to_action} does not end with suffix {suffix}.'
          ' Check that call_to_make_observation is set correctly.'
      )
    return call_to_action.removeprefix(prefix).removesuffix(suffix)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      active_entity_name = self._get_active_entity_name_from_call_to_action(
          action_spec.call_to_action
      )

      forum_state = self._get_forum_state()
      notifications = forum_state.drain_notifications(active_entity_name)
      forum_summary = forum_state.get_forum_summary_for_player(
          active_entity_name
      )

      parts = []
      if notifications:
        parts.extend(notifications)
      parts.append(forum_summary)
      vote_changes = forum_state.get_vote_changes_for_player(active_entity_name)
      if vote_changes:
        parts.append(vote_changes)
      vote_summary = forum_state.get_vote_summary()
      if vote_summary:
        parts.append(vote_summary)
      karma_summary = forum_state.get_karma_summary()
      if karma_summary:
        parts.append(karma_summary)
      pinned_summary = forum_state.get_pinned_post_summary()
      if pinned_summary:
        parts.append(pinned_summary)
      result = '\n\n\n'.join(parts)

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass

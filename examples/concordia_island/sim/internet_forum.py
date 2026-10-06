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

"""Thread-safe Forum with dating features, karma system, and reply voting.

Supports dating profiles and a partner selection mechanism for dates.
"""

from collections.abc import Mapping, Sequence
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
from examples.concordia_island.sim import component_state

DEFAULT_FORUM_COMPONENT_KEY = '__forum__'
DEFAULT_FORUM_PRE_ACT_LABEL = '\nForum'

DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}? What do they now observe?'
    ' Only include information of which they are aware.'
)

PUTATIVE_EVENT_TAG = '[putative_event]'


@dataclasses.dataclass
class Post:
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
  is_profile: bool = False


def _nested_any_map(raw: Any) -> dict[str, dict[str, Any]]:
  """Reads a two-level mapping, leaving the innermost values untouched."""
  if not isinstance(raw, Mapping):
    return {}
  result: dict[str, dict[str, Any]] = {}
  for key, value in raw.items():
    if isinstance(value, Mapping):
      result[str(key)] = {str(k): v for k, v in value.items()}
  return result


def _nested_str_map(raw: Any) -> dict[str, dict[str, str]]:
  """Reads a two-level mapping whose innermost values are strings."""
  return {
      key: {k: str(v) for k, v in inner.items()}
      for key, inner in _nested_any_map(raw).items()
  }


def _post_from_state(post_id: int, data: Mapping[str, Any]) -> Post:
  """Rebuilds a `Post` from its checkpointed dict.

  Fields are read by name rather than splatted, so a checkpoint written by a
  build with a different set of `Post` fields restores using the dataclass
  defaults instead of raising `TypeError` and failing the whole restore.

  Args:
    post_id: The integer ID of the post.
    data: Mapping containing the serialized post fields.

  Returns:
    The deserialized Post object.
  """
  return Post(
      post_id=post_id,
      author=component_state.as_str(data, 'author'),
      title=component_state.as_str(data, 'title'),
      content=component_state.as_str(data, 'content'),
      timestamp=component_state.as_str(data, 'timestamp'),
      votes=component_state.as_int(data, 'votes'),
      image=component_state.as_optional_str(data, 'image'),
      replies=component_state.as_dict_list(data, 'replies'),
      vote_log=[
          {str(k): str(v) for k, v in entry.items()}
          for entry in component_state.as_dict_list(data, 'vote_log')
      ],
      min_karma_to_reply=component_state.as_int(data, 'min_karma_to_reply'),
      is_profile=component_state.as_bool(data, 'is_profile'),
  )


class InternetForumState(entity_component.ContextComponent):
  """Thread-safe forum state managing posts, replies, votes, karma, and dating profiles."""

  def __init__(
      self,
      player_names: Sequence[str],
      forum_name: str = 'Instagram',
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

    self._lock = threading.RLock()
    self._posts: dict[int, Post] = {}
    self._next_post_id = 0
    self._next_reply_id = 0
    self._karma: dict[str, int] = {name: 0 for name in player_names}
    self._notification_queue: dict[str, list[str]] = {
        name: [] for name in player_names
    }
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
    self._bans: dict[str, dict[str, Any]] = {}
    self._timestamp_change_count: int = 0

    # Halo specific dating features
    self._profiles: dict[str, dict[str, Any]] = {}
    self._selections: dict[str, str] = {}
    self._swipes: dict[str, dict[str, str]] = {}

  def set_current_timestamp(self, timestamp: str) -> None:
    with self._lock:
      old_timestamp = self._current_timestamp
      self._current_timestamp = timestamp
      if timestamp and timestamp != old_timestamp:
        self._timestamp_change_count += 1
        self._process_ban_expirations_locked(timestamp)

  def get_current_timestamp(self) -> str:
    with self._lock:
      return self._current_timestamp

  def _process_ban_expirations_locked(self, current_timestamp: str) -> None:
    expired = []
    for player_name, ban_info in self._bans.items():
      ban_info['remaining_changes'] -= 1
      if ban_info['remaining_changes'] <= 0:
        expired.append(player_name)

    for player_name in expired:
      ban_info = self._bans.pop(player_name)
      banned_at = ban_info.get('banned_at_timestamp', 'unknown')
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
            ' they are not a recognised user.'
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

    title = (
        f'[MODERATOR ACTION] {moderator} has temporarily banned'
        f' {resolved_target}'
    )
    content = (
        public_note or f'{moderator} has temporarily banned {resolved_target}.'
    )
    self.create_post(author=moderator, title=title, content=content)

    with self._lock:
      ts = self._current_timestamp
      ts_str = f' [{ts}]' if ts else ''
      self._notification_queue[resolved_target].append(
          f'[BAN NOTICE]{ts_str} You have been temporarily banned by'
          f' {moderator}. You will be unable to post until the time'
          f' advances. Private message from {moderator}: {private_note}'
      )

    return f'{moderator} temporarily banned {resolved_target}.'

  def is_banned(self, player_name: str) -> bool:
    with self._lock:
      return player_name in self._bans

  def create_post(
      self,
      author: str,
      title: str,
      content: str,
      timestamp: str = '',
      image: str | None = None,
      is_profile: bool = False,
  ) -> int:
    with self._lock:
      post_id = self._next_post_id
      self._next_post_id += 1
      self._posts[post_id] = Post(
          post_id=post_id,
          author=author,
          title=title,
          content=content,
          timestamp=timestamp
          or self._current_timestamp
          or datetime.datetime.now().isoformat(),
          image=image,
          is_profile=is_profile,
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
      self._posts[post_id].votes += 1
      self._posts[post_id].vote_log.append({'voter': voter, 'direction': 'up'})
      author = self._posts[post_id].author
      if voter and voter != author and author in self._karma:
        self._karma[author] += 1
      return True

  def downvote(self, post_id: int, voter: str = '') -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      self._posts[post_id].votes -= 1
      self._posts[post_id].vote_log.append(
          {'voter': voter, 'direction': 'down'}
      )
      author = self._posts[post_id].author
      if voter and voter != author and author in self._karma:
        self._karma[author] -= 1
      return True

  def upvote_reply(self, post_id: int, reply_id: int, voter: str = '') -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      for reply in self._posts[post_id].replies:
        if reply['reply_id'] == reply_id:
          reply['votes'] = reply.get('votes', 0) + 1
          if 'vote_log' not in reply:
            reply['vote_log'] = []
          reply['vote_log'].append({'voter': voter, 'direction': 'up'})
          author = str(reply['author'])
          if voter and voter != author and author in self._karma:
            self._karma[author] += 1
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
          reply['votes'] = reply.get('votes', 0) - 1
          if 'vote_log' not in reply:
            reply['vote_log'] = []
          reply['vote_log'].append({'voter': voter, 'direction': 'down'})
          author = str(reply['author'])
          if voter and voter != author and author in self._karma:
            self._karma[author] -= 1
          return True
      return False

  def get_karma_summary(self) -> str:
    with self._lock:
      entries = [f'{name}: {score}' for name, score in self._karma.items()]
      return 'Karma scores: ' + ', '.join(entries)

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
    with self._lock:
      resolved_recipient = self._aliases.get(recipient, recipient)
      if resolved_recipient not in self._notification_queue:
        return f'Unknown user {recipient}'
      ts = self._current_timestamp
      ts_str = f' [{ts}]' if ts else ''
      self._notification_queue[resolved_recipient].append(
          f'[Direct message from {sender}]{ts_str}: {content}'
      )
      pair_key = '|'.join(sorted([sender, resolved_recipient]))
      if pair_key not in self._direct_message_threads:
        self._direct_message_threads[pair_key] = []
      self._direct_message_threads[pair_key].append({
          'sender': sender,
          'content': content,
          'timestamp': ts,
      })
      return f'Message sent to {resolved_recipient}.'

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

  def get_vote_summary(self) -> str:
    with self._lock:
      posts = sorted(self._posts.values(), key=lambda p: p.post_id)
      if not posts:
        return ''
      entries = [f'Post #{p.post_id}: {p.votes} votes' for p in posts]
      return f'{self._forum_name} vote counts: ' + ', '.join(entries)

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
                by_post[pid].append(
                    f'reply {rid} by {r["author"]}: {sign}{delta}'
                )
                break

      lines = ['New votes:']
      for pid in sorted(by_post.keys()):
        post = self._posts.get(pid)
        if not post:
          continue
        parts_list = []
        if pid in post_deltas:
          d = post_deltas[pid]
          s = '+' if d > 0 else ''
          parts_list.append(f'post: {s}{d}')
        parts_list.extend(by_post[pid])
        lines.append(
            f'- post {pid} by {post.author} ("{post.title}")'
            f' -- {", ".join(parts_list)}'
        )

      if len(lines) == 1:
        return ''
      return '\n'.join(lines)

  def extract_json(self, text: str) -> dict[str, Any] | None:
    text = (
        text.replace('“', '"')
        .replace('”', '"')
        .replace('‘', "'")
        .replace('’', "'")
    )
    fence_match = re.search(
        r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL
    )
    if fence_match:
      try:
        return json.loads(fence_match.group(1))
      except json.JSONDecodeError:
        pass
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

  def parse_and_execute_action(
      self, action_text: str, entity_name: str | None = None
  ) -> str:
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
      return (
          f'{actor_name} attempted to act but it could not be parsed.'
          f' Expected valid JSON. Got: "{action_text}"'
      )

    action_type = action.get('action', '')
    raw_author = action.get('author', 'Unknown')
    author = raw_author

    if entity_name is not None and author != entity_name:
      author = f'{author} [UNVERIFIED]'

    actor_name = entity_name or author

    if action_type == 'post':
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
      return result
    elif action_type == 'reply':
      post_id = self._parse_post_id(action.get('post_id', -1))
      content = action.get('content', '')
      reply_id = self.reply_to_post(
          post_id=post_id,
          author=author,
          content=content,
          image=image_data,
      )
      if reply_id is not None:
        result = f'{author} replied to post #{post_id}: "{content}"'
        self.queue_notification(author, result)
        return result
      else:
        return f'Failed to reply. Post #{post_id} not found.'
    elif action_type == 'upvote_post':
      post_id = self._parse_post_id(action.get('post_id', -1))
      success = self.upvote(post_id, voter=author)
      if success:
        return f'{author} upvoted post #{post_id}'
      return f'Post #{post_id} not found.'
    elif action_type == 'downvote_post':
      post_id = self._parse_post_id(action.get('post_id', -1))
      success = self.downvote(post_id, voter=author)
      if success:
        return f'{author} downvoted post #{post_id}'
      return f'Post #{post_id} not found.'
    elif action_type == 'create_profile':
      profile_data = action.get('profile', {})
      if image_data:
        profile_data['image'] = image_data
      with self._lock:
        self._profiles[actor_name] = profile_data
        # Create a public post for the profile
        post_id = self.create_post(
            author=author,
            title=f'[Dating Profile] {actor_name}',
            content=json.dumps(profile_data),
            is_profile=True,
            image=image_data,
        )
      result = f'{author} created a dating profile.'
      self.queue_notification(author, result)
      return result
    elif action_type == 'swipe':
      decisions = action.get('decisions', {})
      if not isinstance(decisions, dict):
        return (
            f'{actor_name} attempted to swipe but "decisions" was not a'
            ' dictionary.'
        )

      with self._lock:
        if actor_name not in self._swipes:
          self._swipes[actor_name] = {}
        for target, decision in decisions.items():
          resolved_target = self._aliases.get(target, target)
          str_decision = str(decision)
          self._swipes[actor_name][resolved_target] = str_decision

          # Check for mutual match
          if str_decision.lower() == 'yes':
            target_swipes = self._swipes.get(resolved_target, {})
            if str(target_swipes.get(actor_name, '')).lower() == 'yes':
              self.queue_notification(
                  actor_name,
                  f"It's a match! You and {resolved_target} both liked each"
                  ' other.',
              )
              self.queue_notification(
                  resolved_target,
                  f"It's a match! You and {actor_name} both liked each other.",
              )

      result = f'{actor_name} swiped on profiles: {json.dumps(decisions)}'
      self.queue_notification(actor_name, result)
      return result
    elif action_type == 'select_partner':
      target = action.get('target', '')
      with self._lock:
        self._selections[actor_name] = target
      result = f'{author} selected {target} as a potential date partner.'
      self.queue_notification(author, result)
      return result
    elif action_type == 'direct_message':
      recipient = action.get('recipient', '')
      content = action.get('content', '')
      result = self.send_direct_message(
          sender=actor_name, recipient=recipient, content=content
      )
      self.queue_notification(author, result)
      return result
    else:
      return f'Unknown action type "{action_type}".'

  def get_profiles(self) -> dict[str, dict[str, Any]]:
    with self._lock:
      return dict(self._profiles)

  def get_selections(self) -> dict[str, str]:
    with self._lock:
      return dict(self._selections)

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

  def to_json(self) -> str:
    """Export the full forum state as structured JSON for analysis."""
    with self._lock:
      posts_data = []
      for post_id in sorted(self._posts):
        post = self._posts[post_id]
        posts_data.append({
            'post_id': post.post_id,
            'author': post.author,
            'title': post.title,
            'content': post.content,
            'timestamp': post.timestamp,
            'votes': post.votes,
            'image': post.image,
            'is_profile': post.is_profile,
            'min_karma_to_reply': post.min_karma_to_reply,
            'vote_log': list(post.vote_log),
            'replies': list(post.replies),
        })

      return json.dumps(
          {
              'forum_name': self._forum_name,
              'posts': posts_data,
              'profiles': dict(self._profiles),
              'selections': dict(self._selections),
              'swipes': {k: dict(v) for k, v in self._swipes.items()},
              'karma': dict(self._karma),
              'direct_messages': {
                  k: list(v) for k, v in self._direct_message_threads.items()
              },
              'bans': {k: dict(v) for k, v in self._bans.items()},
              'player_names': list(self._player_names),
              'current_timestamp': self._current_timestamp,
          },
          indent=2,
          default=str,
      )

  def to_html(self, title: str = '') -> str:
    """Render the forum state as a styled HTML page."""
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

        profile_badge = ''
        if post.is_profile:
          profile_badge = (
              ' <span class="profile-badge">💕 Dating Profile</span>'
          )

        replies_html = ''
        for reply in post.replies:
          reply_votes = reply.get('votes', 0)
          rv_class = ''
          if reply_votes > 0:
            rv_class = ' positive'
          elif reply_votes < 0:
            rv_class = ' negative'
          replies_html += f"""
          <div class="reply">
            <div class="reply-vote-column">
              <span class="vote-arrow up">▲</span>
              <span class="vote-count{rv_class}">{reply_votes}</span>
              <span class="vote-arrow down">▼</span>
            </div>
            <div class="reply-body">
              <div class="reply-meta">
                <span class="author{' unverified' if '[UNVERIFIED]' in str(reply['author']) else ''}">{self._escape_html(reply['author'])}</span>
                <span class="timestamp">{self._escape_html(reply.get('timestamp', ''))}</span>
              </div>
              <div class="reply-content">{self._escape_html(reply['content'])}</div>
            </div>
          </div>"""

        reply_count = len(post.replies)
        reply_label = f'{reply_count} repl{"ies" if reply_count != 1 else "y"}'

        posts_html += f"""
        <div class="post{' profile-post' if post.is_profile else ''}">
          <div class="vote-column">
            <div class="vote-arrow up">▲</div>
            <div class="vote-count{vote_class}">{post.votes}</div>
            <div class="vote-arrow down">▼</div>
          </div>
          <div class="post-content">
            <div class="post-title">{self._escape_html(post.title)}{profile_badge}</div>
            <div class="post-meta">
              Posted by <span class="author{' unverified' if '[UNVERIFIED]' in post.author else ''}">{self._escape_html(post.author)}</span>
              <span class="timestamp">{self._escape_html(post.timestamp)}</span>
            </div>
            <div class="post-body">{self._escape_html(post.content)}</div>
            {self._render_post_image(post)}
            <div class="post-actions">
              <span class="action-item">💬 {reply_label}</span>
            </div>
            {f'<div class="replies">{replies_html}</div>' if post.replies else ''}
          </div>
        </div>"""

    # Build profiles & selections summary
    profiles_html = ''
    with self._lock:
      if self._profiles:
        profiles_html += '<div class="section"><h2>💕 Dating Profiles</h2>'
        for name, profile in self._profiles.items():
          if not isinstance(profile, dict):
            continue
          image_data = profile.get('image', '')
          image_html = ''
          if image_data:
            src = (
                self._extract_image_src(image_data)
                if image_data.startswith('![')
                else image_data
            )
            if src:
              image_html = (
                  f'<div class="profile-image"><img src="{src}" alt="Profile'
                  ' Picture" style="max-width: 150px; max-height: 150px;'
                  ' border-radius: 50%;"></div>'
              )

          profiles_html += f"""
          <div class="profile-card">
            {image_html}
            <div class="profile-name">{self._escape_html(name)}</div>
            <div class="profile-detail"><b>Bio:</b> {self._escape_html(str(profile.get('bio', '')))}</div>
            <div class="profile-detail"><b>Occupation:</b> {self._escape_html(str(profile.get('occupation', '')))}</div>
            <div class="profile-detail"><b>Age:</b> {self._escape_html(str(profile.get('age', '')))}</div>
            <div class="profile-detail"><b>Interests:</b> {self._escape_html(str(profile.get('interests', '')))}</div>
            <div class="profile-detail"><b>Looking for:</b> {self._escape_html(str(profile.get('looking_for', '')))}</div>
          </div>"""
        profiles_html += '</div>'

      if self._selections:
        profiles_html += (
            '<div class="section"><h2>💘 Partner Selections</h2><ul'
            ' class="selections-list">'
        )
        for selector, target in self._selections.items():
          profiles_html += (
              f'<li><span class="author">{self._escape_html(selector)}</span> →'
              f' <span class="author">{self._escape_html(target)}</span></li>'
          )
        profiles_html += '</ul></div>'

      if self._swipes:
        profiles_html += (
            '<div class="section"><h2>👉 Swipes</h2><ul'
            ' class="selections-list">'
        )
        for swiper, decisions in self._swipes.items():
          if not isinstance(decisions, dict):
            continue
          for target, decision in decisions.items():
            profiles_html += (
                f'<li><span class="author">{self._escape_html(swiper)}</span> '
                f'swiped {self._escape_html(decision)} on '
                f'<span class="author">{self._escape_html(target)}</span></li>'
            )
        profiles_html += '</ul></div>'

    stats_line = f'{len(posts_sorted)} posts'
    total_replies = sum(len(p.replies) for p in posts_sorted)
    if total_replies:
      stats_line += f' · {total_replies} replies'
    num_profiles = len(self._profiles)
    if num_profiles:
      stats_line += f' · {num_profiles} dating profiles'
    num_selections = len(self._selections)
    if num_selections:
      stats_line += f' · {num_selections} partner selections'

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
      border-bottom: 3px solid #e91e63;
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
    .section {{
      margin: 24px 0;
    }}
    .section h2 {{
      font-size: 18px;
      color: #e0e0e0;
      margin-bottom: 12px;
      padding-bottom: 6px;
      border-bottom: 1px solid #343536;
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
    .post.profile-post {{
      border-left: 3px solid #e91e63;
    }}
    .profile-badge {{
      display: inline-block;
      background: #e91e63;
      color: white;
      font-size: 11px;
      padding: 1px 6px;
      border-radius: 3px;
      margin-left: 8px;
      font-weight: 600;
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
    .profile-card {{
      background: #272729;
      border: 1px solid #343536;
      border-left: 3px solid #e91e63;
      border-radius: 4px;
      padding: 12px 16px;
      margin-bottom: 8px;
    }}
    .profile-name {{
      font-size: 16px;
      font-weight: 600;
      color: #4fbcff;
      margin-bottom: 6px;
    }}
    .profile-detail {{
      font-size: 13px;
      color: #c8cbcd;
      margin-bottom: 4px;
    }}
    .selections-list {{
      list-style: none;
      padding: 0;
    }}
    .selections-list li {{
      background: #272729;
      border: 1px solid #343536;
      padding: 8px 14px;
      margin-bottom: 4px;
      border-radius: 4px;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-inner">
      <h1>💕 {self._escape_html(title)}</h1>
      <div class="stats">{stats_line}</div>
    </div>
  </div>
  <div class="content">
    {profiles_html}
    {posts_html}
  </div>
</body>
</html>"""

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
          'profiles': dict(self._profiles),
          'selections': dict(self._selections),
          'swipes': {k: dict(v) for k, v in self._swipes.items()},
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._posts = {}
      raw_posts = state.get('posts')
      if isinstance(raw_posts, Mapping):
        for pid, post_data in raw_posts.items():
          if isinstance(post_data, Mapping):
            post_id = int(str(pid))
            fields: dict[str, Any] = {
                str(k): v for k, v in post_data.items()
            }
            self._posts[post_id] = _post_from_state(post_id, fields)
      self._next_post_id = component_state.as_int(state, 'next_post_id')
      self._next_reply_id = component_state.as_int(state, 'next_reply_id')
      self._karma = component_state.as_str_int_map(state, 'karma')
      self._profiles = _nested_any_map(state.get('profiles'))
      self._selections = component_state.as_str_str_map(state, 'selections')
      self._swipes = _nested_str_map(state.get('swipes'))


class InternetForumResolution(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Resolves player actions on the Halo forum programmatically."""

  def __init__(
      self,
      player_names: Sequence[str],
      forum_component_key: str = DEFAULT_FORUM_COMPONENT_KEY,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = event_resolution.DEFAULT_RESOLUTION_PRE_ACT_LABEL,
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._forum_component_key = forum_component_key
    self._memory_component_key = memory_component_key
    self._pre_act_label = pre_act_label
    self._resolved_per_entity: dict[str, int] = {}
    self._resolution_lock = threading.Lock()

  def _get_forum_state(self) -> InternetForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=InternetForumState
    )

  def _get_putative_action(self) -> tuple[str | None, str | None]:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    suggestions = memory.scan(selector_fn=lambda x: PUTATIVE_EVENT_TAG in x)
    if not suggestions:
      return None, None

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
        prefix = f'{PUTATIVE_EVENT_TAG} {name}'
        entity_suggestions = [s for s in suggestions if prefix in s]
        resolved = self._resolved_per_entity.get(name, 0)
        if len(entity_suggestions) > resolved:
          selected = entity_suggestions[resolved]
          self._resolved_per_entity[name] = resolved + 1

          putative_action = selected[
              selected.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG) :
          ]
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

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      active_entity_name, putative_action = self._get_putative_action()

      forum_state = self._get_forum_state()
      if putative_action is not None:
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
    self._resolved_per_entity = component_state.as_str_int_map(
        state, 'resolved_per_entity'
    )


class InternetForumObservation(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Returns forum observations to players programmatically."""

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

  def _get_forum_state(self) -> InternetForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=InternetForumState
    )

  def _get_active_entity_name_from_call_to_action(
      self, call_to_action: str
  ) -> str:
    prefix, suffix = self._call_to_make_observation.split('{name}')
    if not call_to_action.startswith(prefix):
      raise ValueError(f'Call to action does not start with prefix {prefix}')
    if not call_to_action.endswith(suffix):
      raise ValueError(f'Call to action does not end with suffix {suffix}')
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
      vote_summary = forum_state.get_vote_summary()
      if vote_summary:
        parts.append(vote_summary)

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


HaloForumState = InternetForumState

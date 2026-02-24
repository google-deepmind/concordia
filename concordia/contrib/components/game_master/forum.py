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
    votes, and per-player observation queue). Registered under __forum__
    and accessed by other components via get_entity().get_component().
    Similar to how AssociativeMemory wraps AssociativeMemoryBank. Thread-safe.
  - ForumResolution: A component (registered as __resolution__) that parses
    player JSON actions and executes them on ForumState programmatically.
    No LLM calls.
  - ForumObservation: A component (registered as __make_observation__) that
    drains ForumState's per-player observation queue and returns the forum
    summary. No LLM calls.
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

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

DEFAULT_FORUM_COMPONENT_KEY = '__forum__'
DEFAULT_FORUM_PRE_ACT_LABEL = '\nForum'

DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}? What do they now observe?'
    ' Only include information of which they are aware.'
)


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


class ForumState(entity_component.ContextComponent):
  """Thread-safe forum state managing posts, replies, votes, and observations.

  Registered under __forum__ and accessed by ForumResolution and
  ForumObservation via get_entity().get_component(). This follows the same
  pattern as AssociativeMemory wrapping AssociativeMemoryBank.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      forum_name: str = 'Community Forum',
      max_summary_posts: int = 10,
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._forum_name = forum_name
    self._max_summary_posts = max_summary_posts

    self._lock = threading.Lock()
    self._posts: dict[int, Post] = {}
    self._next_post_id = 0
    self._next_reply_id = 0
    self._notification_queue: dict[str, list[str]] = {
        name: [] for name in player_names
    }

    self._last_seen_post_id: dict[str, int] = {
        name: -1 for name in player_names
    }
    self._last_seen_reply_id: dict[str, int] = {
        name: -1 for name in player_names
    }

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
      self._posts[post_id] = Post(
          post_id=post_id,
          author=author,
          title=title,
          content=content,
          timestamp=timestamp or datetime.datetime.now().isoformat(),
          image=image,
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
          'timestamp': timestamp or datetime.datetime.now().isoformat(),
          'image': image,
      })
      return reply_id

  def upvote(self, post_id: int) -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      self._posts[post_id].votes += 1
      return True

  def downvote(self, post_id: int) -> bool:
    with self._lock:
      if post_id not in self._posts:
        return False
      self._posts[post_id].votes -= 1
      return True

  def get_recent_posts(self, n: int | None = None) -> list[Post]:
    with self._lock:
      all_posts = sorted(
          self._posts.values(), key=lambda p: p.post_id, reverse=True
      )
      if n is not None:
        return all_posts[:n]
      return all_posts

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
    summary = (
        f'[Post #{post.post_id}] "{post.title}" by {post.author}'
        f' (votes: {post.votes}, replies: {reply_count})'
    )
    if post.content and post.content != post.title:
      summary += f'\n  {post.content}'
    if post.replies:
      latest = post.replies[-1]
      summary += (
          f'\n  Latest reply by {latest["author"]}: "{latest["content"]}"'
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
        self._last_seen_reply_id[player_name] = max(all_new_reply_ids)
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
            lines.append(
                f'  New reply to post #{post.post_id} by {r["author"]}:'
                f' "{r["content"]}"'
            )
      return '\n\n\n'.join(lines)

  def extract_json(self, text: str) -> dict[str, Any] | None:
    fence_match = re.search(
        r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL
    )
    if fence_match:
      try:
        return json.loads(fence_match.group(1))
      except json.JSONDecodeError:
        pass

    brace_match = re.search(r'(\{[^{}]*\})', text)
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

  def parse_and_execute_action(self, action_text: str) -> str:
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
      return (
          'Error: Could not parse action. Expected valid JSON.'
          f' Got: "{action_text}"'
      )

    action_type = action.get('action', '')
    author = action.get('author', 'Unknown')

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
        return result
      else:
        title = content[:80] + ('...' if len(content) > 80 else '')
        new_post_id = self.create_post(
            author=author, title=title, content=content
        )
        result = f'{author} created post #{new_post_id}: "{title}"'
        return result
    elif action_type == 'upvote':
      post_id = self._parse_post_id(action.get('post_id', -1))
      success = self.upvote(post_id)
      if success:
        result = f'{author} upvoted post #{post_id}'
        self.queue_notification(author, result)
        return result
      else:
        result = f'{author} tried to upvote non-existent post #{post_id}'
        self.queue_notification(author, result)
        return result
    elif action_type == 'downvote':
      post_id = self._parse_post_id(action.get('post_id', -1))
      success = self.downvote(post_id)
      if success:
        result = f'{author} downvoted post #{post_id}'
        self.queue_notification(author, result)
        return result
      else:
        result = f'{author} tried to downvote non-existent post #{post_id}'
        self.queue_notification(author, result)
        return result
    else:
      result = (
          f'Error: Unknown action type "{action_type}". '
          'Expected one of: post, reply, upvote, downvote.'
      )
      self.queue_notification(author, result)
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
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._posts = {}
      posts_data = dict(state.get('posts', {}))
      for pid_str, post_data in posts_data.items():
        self._posts[int(pid_str)] = Post(**post_data)
      self._next_post_id = int(state.get('next_post_id', 0))
      self._next_reply_id = int(state.get('next_reply_id', 0))

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
          replies_html += f"""
          <div class="reply">
            <div class="reply-meta">
              <span class="author">{self._escape_html(reply['author'])}</span>
              <span class="timestamp">{self._escape_html(reply.get('timestamp', ''))}</span>
            </div>
            <div class="reply-content">{self._escape_html(reply['content'])}</div>
            {reply_image_html}
          </div>"""

        reply_count = len(post.replies)
        reply_label = f'{reply_count} repl{"ies" if reply_count != 1 else "y"}'

        posts_html += f"""
        <div class="post">
          <div class="vote-column">
            <div class="vote-arrow up">▲</div>
            <div class="vote-count{vote_class}">{post.votes}</div>
            <div class="vote-arrow down">▼</div>
          </div>
          <div class="post-content">
            <div class="post-title">{self._escape_html(post.title)}</div>
            <div class="post-meta">
              Posted by <span class="author">{self._escape_html(post.author)}</span>
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

    stats_line = f'{len(posts_sorted)} posts'
    total_replies = sum(len(p.replies) for p in posts_sorted)
    if total_replies:
      stats_line += f' · {total_replies} replies'

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
      padding: 8px 10px;
      margin: 4px 0 4px 16px;
      border-left: 2px solid #3b82f6;
      background: #1e1e20;
      border-radius: 0 4px 4px 0;
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
      <h1>📋 {self._escape_html(title)}</h1>
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
  Accesses ForumState via get_entity().get_component(forum_key).
  """

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
    self._resolved_suggestions: set[str] = set()
    self._active_entity_name: str | None = None
    self._putative_action: str | None = None

  def _get_forum_state(self) -> ForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=ForumState
    )

  def _get_putative_action(self) -> tuple[str | None, str | None]:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    suggestions = memory.scan(selector_fn=lambda x: PUTATIVE_EVENT_TAG in x)
    if not suggestions:
      return None, None

    unresolved = [s for s in suggestions if s not in self._resolved_suggestions]

    if not unresolved:
      return None, None

    selected = unresolved[-1]
    self._resolved_suggestions.add(selected)

    putative_action = selected[
        selected.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG) :
    ]

    active_entity_name = None
    for name in self._player_names:
      prefix = f' {name}'
      if putative_action.startswith(prefix):
        active_entity_name = name
        remainder = putative_action[len(prefix) :]
        if remainder.startswith(':'):
          remainder = remainder[1:]
        elif remainder.startswith(' --'):
          remainder = remainder[3:]
        putative_action = remainder.strip()
        break

    return active_entity_name, putative_action

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

      forum_state = self._get_forum_state()
      if putative_action is not None:
        result = forum_state.parse_and_execute_action(putative_action)
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
        'resolved_suggestions': list(self._resolved_suggestions),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._resolved_suggestions = set(
        str(s) for s in state.get('resolved_suggestions', [])
    )


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
      vote_summary = forum_state.get_vote_summary()
      if vote_summary:
        parts.append(vote_summary)
      # On the agent side, ObservationToMemory splits at each '\n\n\n' to
      # create multiple observation memories.
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

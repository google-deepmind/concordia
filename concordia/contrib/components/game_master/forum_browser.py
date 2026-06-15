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

"""Pull-based forum access via tool calling.

Components:
  ForumBrowserTool: A concordia Tool wrapping ForumState for on-demand
    thread listing and reading.  Both actions support pagination.
  ForumBrowsingContext: An agent component that runs a custom tool-calling
    loop with rolling summarization.  Previously viewed pages are
    compressed into a summary so context size stays bounded regardless
    of how many pages the agent browses.
  MinimalForumObservation: A lightweight replacement for ForumObservation
    that only pushes notifications (bans, direct messages, reinstatements) and a
    compact status line.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.contrib.components.game_master import forum as forum_module
from concordia.document import tool as tool_module
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


# Maximum characters of post content to include in the list_threads preview.
_CONTENT_PREVIEW_LENGTH = 80

# Default number of tool calls allowed per browsing session.
_DEFAULT_MAX_TOOL_CALLS = 5

# Default page sizes for pagination.
_DEFAULT_THREADS_PER_PAGE = 5
_DEFAULT_REPLIES_PER_PAGE = 5

# Separator used between sections in logged output.
_LOG_SEPARATOR = '═' * 60

# Regex pattern for detecting JSON tool calls in LLM output.
# Matches flat JSON objects like {"action": "list_threads", "page": 2}.
_TOOL_CALL_JSON_PATTERN = re.compile(r'(\{[^}]+\})', re.DOTALL)


class ForumBrowserTool(tool_module.Tool):
  """Tool for browsing forum threads on-demand.

  Operates on a **frozen snapshot** of posts rather than querying
  ForumState live.  This avoids lock contention between agent threads
  and prevents agents from polling ``list_threads`` in a loop to watch
  for new content from other players.

  Two actions are exposed:
    - list_threads: Returns a paginated index of threads sorted by
      recent activity (most recently interacted first). Each page
      shows up to ``threads_per_page`` threads plus the pinned thread.
    - read_thread: Returns the content of a specific thread with
      paginated replies in chronological order, or a direct message
      thread by player name.

  The output format for list_threads follows the PostTracker convention:
    Post #N | by Author | votes: V | replies: R
      Title: "..."
      Preview: ...
  """

  def __init__(
      self,
      posts: list[forum_module.Post],
      player_name: str = '',
      direct_message_threads: dict[str, int] | None = None,
      karma: dict[str, int] | None = None,
      aliases: dict[str, str] | None = None,
      content_preview_length: int = _CONTENT_PREVIEW_LENGTH,
      pinned_post_id: int | None = None,
      threads_per_page: int = _DEFAULT_THREADS_PER_PAGE,
      replies_per_page: int = _DEFAULT_REPLIES_PER_PAGE,
  ):
    self._posts = posts
    self._player_name = player_name
    self._direct_message_threads = direct_message_threads or {}
    self._karma = karma or {}
    self._aliases = aliases or {}
    self._content_preview_length = content_preview_length
    self._pinned_post_id = pinned_post_id
    self._threads_per_page = threads_per_page
    self._replies_per_page = replies_per_page

  @property
  def name(self) -> str:
    return 'browse_forum'

  @property
  def description(self) -> str:
    return (
        'Browse the community forum.\n'
        'Actions:\n'
        '  list_threads - List threads with summary stats (paginated).\n'
        '    Args: action (str) = "list_threads",'
        ' page (int, optional, default 1)\n'
        '  read_thread - Read a forum thread or direct message conversation'
        ' (paginated).\n'
        '    Args: action (str) = "read_thread", post_id (int or str),'
        ' page (int, optional, default 1)\n'
        '    post_id can be a numeric post ID for forum threads,\n'
        '    or a player name to read a direct message conversation.\n'
    )

  def execute(self, **kwargs: Any) -> str:
    action = kwargs.get('action', '')
    if action == 'list_threads':
      page = int(kwargs.get('page', 1))
      return self.list_threads_summary(page=page)
    elif action == 'read_thread':
      post_id = kwargs.get('post_id')
      page = int(kwargs.get('page', 1))
      if post_id is None:
        return 'Error: read_thread requires a post_id argument.'
      post_id_str = str(post_id)
      # Check if post_id is a player name (or alias) for direct message threads.
      resolved = self._aliases.get(post_id_str, post_id_str)
      if resolved in self._direct_message_threads:
        return self._read_dm_thread(resolved)
      try:
        return self._read_thread(int(post_id), page=page)
      except (ValueError, TypeError):
        return f'Error: "{post_id}" is not a valid post ID or player name.'
    else:
      return f'Unknown action: {action}. Use "list_threads" or "read_thread".'

  def list_threads_summary(self, page: int = 1) -> str:
    """Returns a paginated index of threads sorted by recent activity."""
    lines = []

    # Forum threads.
    if self._posts:
      # Separate pinned post from the rest.
      pinned_post = None
      non_pinned_posts = []
      for post in self._posts:
        if post.post_id == self._pinned_post_id:
          pinned_post = post
        else:
          non_pinned_posts.append(post)

      # Sort non-pinned posts by last_activity_seq descending (most recent
      # activity first). This determines both display order and page
      # assignment.
      non_pinned_posts.sort(
          key=lambda p: p.last_activity_seq, reverse=True
      )

      # Paginate non-pinned posts.
      total_pages = max(
          1, math.ceil(len(non_pinned_posts) / self._threads_per_page)
      )
      page = max(1, min(page, total_pages))
      start_idx = (page - 1) * self._threads_per_page
      end_idx = start_idx + self._threads_per_page
      page_posts = non_pinned_posts[start_idx:end_idx]

      # Page header.
      lines.append(
          f'Page {page} of {total_pages} contains the following threads:'
      )
      lines.append('')
      lines.append(
          'Use this reference to identify threads by ID when replying'
          ' or voting:'
      )
      lines.append('')

      # Always show the pinned thread first, regardless of page.
      if pinned_post is not None:
        lines.extend(self._format_thread_entry(pinned_post, is_pinned=True))

      # Show the posts for this page.
      for post in page_posts:
        lines.extend(self._format_thread_entry(post, is_pinned=False))

    else:
      lines.append('(No posts on the forum yet.)')

    # Direct message threads.
    if self._direct_message_threads:
      lines.append('')
      lines.append('Direct message conversations:')
      for partner, count in sorted(self._direct_message_threads.items()):
        partner_karma = self._karma.get(partner, 0)
        lines.append(
            f'  direct message with {partner} | messages: {count}'
            f' | their karma: {partner_karma}'
        )
      lines.append(
          '  (Use {"action": "read_thread", "post_id": "<player_name>"}'
          ' to read a direct message conversation.)'
      )

    # Karma summary.
    if self._karma:
      lines.append('')
      karma_entries = [
          f'{name}: {score}' for name, score in self._karma.items()
      ]
      lines.append('Karma: ' + ', '.join(karma_entries))

    # Page footer.
    if self._posts:
      total_pages = max(
          1,
          math.ceil(
              len([p for p in self._posts
                   if p.post_id != self._pinned_post_id])
              / self._threads_per_page
          ),
      )
      page = max(1, min(page, total_pages))
      lines.append('')
      lines.append(f'page {page} of {total_pages}')

    return '\n'.join(lines)

  def _format_thread_entry(
      self, post: forum_module.Post, is_pinned: bool
  ) -> list[str]:
    """Format a single thread entry for the list_threads summary."""
    lines = []
    votes = post.votes
    replies = len(post.replies)
    author = post.author
    title = post.title or '(untitled)'
    content = post.content or ''
    timestamp = post.timestamp or ''

    # Truncate content for preview.
    if len(content) > self._content_preview_length:
      preview = content[: self._content_preview_length] + '...'
    else:
      preview = content

    pinned_tag = '[** PINNED THREAD **] ' if is_pinned else ''
    lines.append(
        f'  {pinned_tag}Post #{post.post_id} | by {author} | {timestamp}'
        f' | votes: {votes} | replies: {replies}'
    )
    if title:
      lines.append(f'    Title: "{title}"')
    if preview:
      lines.append(f'    Preview: {preview}')
    return lines

  def _read_thread(self, post_id: int, page: int = 1) -> str:
    """Returns the content of a thread with paginated replies."""
    target_post = None
    for post in self._posts:
      if post.post_id == post_id:
        target_post = post
        break

    if target_post is None:
      return f'Error: Post #{post_id} not found.'

    # Replies in chronological order (by position, which corresponds to
    # reply_id assignment order).
    all_replies = list(target_post.replies)
    total_reply_pages = max(
        1, math.ceil(len(all_replies) / self._replies_per_page)
    )
    page = max(1, min(page, total_reply_pages))
    start_idx = (page - 1) * self._replies_per_page
    end_idx = start_idx + self._replies_per_page
    page_replies = all_replies[start_idx:end_idx]

    # Page header.
    lines = [
        f'Page {page} of {total_reply_pages} of thread #{target_post.post_id}:',
        '',
        (
            f'Thread: Post #{target_post.post_id}'
            f' — "{target_post.title or "(untitled)"}"'
        ),
        (
            f'Original post by {target_post.author}'
            f' | {target_post.timestamp or ""}'
            f' | votes: {target_post.votes}:'
        ),
        f'  {target_post.content}',
    ]

    if page_replies:
      lines.append('')
      if total_reply_pages > 1:
        lines.append(
            f'Replies (showing {start_idx + 1}-'
            f'{min(end_idx, len(all_replies))}'
            f' of {len(all_replies)}):'
        )
      else:
        lines.append('Replies:')
      for reply in page_replies:
        reply_id = reply.get('reply_id', '?')
        reply_author = reply.get('author', '?')
        reply_content = reply.get('content', '')
        reply_votes = reply.get('votes', 0)
        reply_timestamp = reply.get('timestamp', '')
        lines.append(
            f'  [Reply #{reply_id}] {reply_author}'
            f' | {reply_timestamp}'
            f' | votes: {reply_votes}:'
        )
        lines.append(f'    {reply_content}')
    elif all_replies:
      # No replies on this page (shouldn't happen with clamping, but safe).
      lines.append('')
      lines.append('(No replies on this page.)')

    # Page footer.
    lines.append('')
    lines.append(f'page {page} of {total_reply_pages}')

    return '\n'.join(lines)

  def _read_dm_thread(self, partner: str) -> str:
    """Returns the full direct message conversation with a partner."""
    # direct message thread data is passed as a count; the actual messages need
    # to be fetched from the forum state. If the tool only has the
    # count, we return a placeholder. The ForumBrowsingContext passes
    # the full messages via _dm_messages.
    messages = getattr(self, 'dm_messages', {}).get(partner, [])
    if not messages:
      return f'(No messages in direct message thread with {partner}.)'

    lines = [f'Direct message conversation with {partner}:']
    for msg in messages:
      sender = msg.get('sender', '?')
      content = msg.get('content', '')
      timestamp = msg.get('timestamp', '')
      ts_str = f' [{timestamp}]' if timestamp else ''
      lines.append(f'  {sender}{ts_str}: {content}')
    return '\n'.join(lines)


class ForumBrowsingContext(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Agent component that lets the LLM browse the forum via tool calls.

  Runs a custom tool-calling loop with **rolling summarization**.
  At each iteration the LLM sees exactly two content blocks
  (beyond the fixed agent context):

    1. A compressed summary of all previously viewed pages.
    2. The current page verbatim.

  This keeps prompt size bounded regardless of how many pages
  the agent browses.  The summarization LLM call preserves all
  quantitative details (vote counts, karma, post/reply IDs, etc.).

  At the start of each step the component takes a **snapshot** of the
  live ``ForumState``.  The tool then operates on that frozen copy,
  which:
    - eliminates lock contention between concurrent agent threads, and
    - prevents an agent from repeatedly calling ``list_threads`` in
      order to wait for new posts from other players.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      forum_state: forum_module.ForumState,
      name: str,
      components: tuple[str, ...] = (),
      self_narrative_key: str = '',
      max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
      threads_per_page: int = _DEFAULT_THREADS_PER_PAGE,
      replies_per_page: int = _DEFAULT_REPLIES_PER_PAGE,
  ):
    """Initializes the ForumBrowsingContext.

    Args:
      model: Language model for the tool-calling session.
      forum_state: The shared ForumState to browse.
      name: The agent's name.
      components: Keys of constant components whose pre_act values should be
        injected as context into the browsing session.
      self_narrative_key: Key of the SelfNarrative component. If set, the
        component's previous narrative is read directly (field access, no lock)
        to provide continuity context for the browsing session.
      max_tool_calls: Maximum number of tool calls allowed per step.
      threads_per_page: Number of threads to show per page in list_threads.
      replies_per_page: Number of replies to show per page in read_thread.
    """
    super().__init__(
        pre_act_label=f"{name}'s Forum Browsing",
    )
    self._model = model
    self._forum_state = forum_state
    self._name = name
    self._components = components
    self._self_narrative_key = self_narrative_key
    self._max_tool_calls = max_tool_calls
    self._threads_per_page = threads_per_page
    self._replies_per_page = replies_per_page

  def _get_component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    component = self.get_entity().get_component(
        key, type_=action_spec_ignored.ActionSpecIgnored
    )
    return f'  {component.get_pre_act_label()}: {component.get_pre_act_value()}'

  def _snapshot_posts(self) -> list[forum_module.Post]:
    """Take a point-in-time snapshot of all forum posts.

    Acquires the ForumState lock exactly once, copies the post list
    (including nested replies), and releases it.  All subsequent
    tool calls operate on this frozen copy.

    Returns:
      A deep copy of the current list of forum posts.
    """
    live_posts = self._forum_state.get_recent_posts()
    return copy.deepcopy(live_posts)

  def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
    """Parse a tool call JSON from the LLM's response.

    Looks for a flat JSON object containing an ``action`` key.
    Returns the parsed arguments dict if found, or None if the
    response is plain text (no tool call).

    Args:
      text: The LLM's response text.

    Returns:
      Parsed tool-call arguments, or None.
    """
    match = _TOOL_CALL_JSON_PATTERN.search(text)
    if match:
      try:
        args = json.loads(match.group(1))
        if isinstance(args, dict) and 'action' in args:
          return args
      except json.JSONDecodeError:
        pass
    return None

  def _summarize_browsed_content(
      self,
      summary_so_far: str,
      current_page: str,
  ) -> str:
    """Compress previously viewed content into a rolling summary.

    Uses the LLM to combine the existing summary with the current
    page into a single shorter summary, preserving all quantitative
    details.

    Args:
      summary_so_far: The existing summary (may be empty on first call).
      current_page: The verbatim page content to fold into the summary.

    Returns:
      A new compressed summary covering all previously viewed content.
    """
    content_to_summarize = ''
    if summary_so_far:
      content_to_summarize = summary_so_far + '\n\n'
    content_to_summarize += current_page

    prompt = (
        f'Summarize the following forum content from'
        f" {self._name}'s perspective.\n\n"
        'IMPORTANT: Preserve ALL quantitative details exactly as they'
        ' appear — including vote counts, karma scores, post IDs, reply'
        ' IDs, timestamps, numerical statistics, prices, quantities, and'
        ' any other numbers. Do not round, approximate, or omit any'
        ' numerical information.\n\n'
        f'Content to summarize:\n{content_to_summarize}'
    )

    return self._model.sample_text(
        prompt=prompt,
        max_tokens=2000,
    )

  def _build_browsing_prompt(
      self,
      context_text: str,
      summary_so_far: str,
      current_page: str,
      tool_desc: str,
  ) -> str:
    """Build the LLM prompt for one iteration of the browsing loop.

    Args:
      context_text: Pre-built context from agent components (identity, etc.).
      summary_so_far: Compressed summary of all previously viewed pages.
        Empty string on the first iteration.
      current_page: Verbatim content of the latest viewed page.
      tool_desc: Formatted tool description text.

    Returns:
      The full prompt string to send to the LLM.
    """
    parts = []

    if context_text:
      parts.append(context_text)

    if summary_so_far:
      parts.append(
          f'Previously browsed content (summary):\n{summary_so_far}'
      )
      parts.append(f'Current page:\n{current_page}')
    else:
      # First iteration — no prior summary, just show the page directly.
      parts.append(current_page)

    parts.append(tool_desc)

    parts.append(
        f'Question: {self._name} needs to catch up on the forum.'
        ' Read the thread(s) most relevant to'
        f" {self._name}'s current situation and concerns"
        ' using {"action": "read_thread", "post_id": <id>}.'
        ' You can also read direct message conversations using'
        ' {"action": "read_thread", "post_id": "<player_name>"}.'
        ' If a thread or listing has multiple pages, use'
        ' {"action": "list_threads", "page": N} or'
        ' {"action": "read_thread", "post_id": <id>, "page": N}'
        ' to navigate.'
        ' When you are done browsing, respond with text'
        ' (not a JSON tool call).'
        '\nAnswer: '
    )

    return '\n\n'.join(parts)

  @staticmethod
  def _format_tool_description(tool: ForumBrowserTool) -> str:
    """Format the tool description for the browsing prompt."""
    return (
        f'Available tool: {tool.name}\n'
        f'{tool.description}\n'
        'To use this tool, respond with a JSON object containing'
        ' the arguments. For example: {"action": "list_threads"}\n'
        'When you are done browsing, respond with text (not JSON).'
    )

  def _make_pre_act_value(self) -> str:
    # Snapshot the forum state once at the start of this step.
    posts_snapshot = self._snapshot_posts()
    direct_message_threads = (
        self._forum_state.get_direct_message_threads_for_player(self._name)
    )
    karma = self._forum_state.get_karma()
    aliases = getattr(self._forum_state, '_aliases', {})
    pinned_post_id = self._forum_state.get_pinned_post_id()

    # Build the tool with direct message and karma data.
    tool = ForumBrowserTool(
        posts_snapshot,
        player_name=self._name,
        direct_message_threads=direct_message_threads,
        karma=karma,
        aliases=aliases,
        pinned_post_id=pinned_post_id,
        threads_per_page=self._threads_per_page,
        replies_per_page=self._replies_per_page,
    )
    # Attach full direct message message data so _read_dm_thread can access it.
    dm_messages: dict[str, list[dict[str, str]]] = {}
    for partner in direct_message_threads:
      dm_messages[partner] = self._forum_state.get_direct_message_thread(
          self._name, partner
      )
    tool.dm_messages = dm_messages  # pytype: disable=attribute-error

    # Collect component context (identity, exemplars, narrative).
    context_parts: list[str] = []
    for key in self._components:
      context_parts.append(self._get_component_pre_act_display(key))

    # Read SelfNarrative's previous narrative directly (field access,
    # no lock) to avoid circular deadlock.
    if self._self_narrative_key:
      try:
        narrative_component = self.get_entity().get_component(
            self._self_narrative_key,
            type_=action_spec_ignored.ActionSpecIgnored,
        )
        previous_narrative = getattr(
            narrative_component, '_previous_narrative', ''
        )
        if previous_narrative:
          context_parts.append(
              f'  {narrative_component.get_pre_act_label()}:'
              f' {previous_narrative}'
          )
      except KeyError:
        pass  # Component not registered.

    context_text = '\n'.join(context_parts) if context_parts else ''

    # Start with list_threads page 1 as the initial current page.
    current_page = tool.list_threads_summary(page=1)
    summary_so_far = ''

    # Full interaction log for debugging.
    interaction_log: list[str] = []

    # Tool description for the prompt.
    tool_desc = self._format_tool_description(tool)

    # ── Custom tool-calling loop with rolling summarization ──
    prev_tool_call_args: dict[str, str] | None = None
    for iteration in range(self._max_tool_calls):
      # Build the prompt for this iteration.
      prompt = self._build_browsing_prompt(
          context_text, summary_so_far, current_page, tool_desc
      )

      # Log this iteration's state.
      interaction_log.append(
          f'{"═" * 20} Iteration {iteration + 1} {"═" * 20}'
      )
      interaction_log.append(f'[prompt ({len(prompt)} chars)]:\n{prompt}')

      # Ask the LLM.
      response = self._model.sample_text(
          prompt=prompt,
          max_tokens=1_000_000,
          terminators=[],
      )
      interaction_log.append(f'[LLM response]:\n{response}')

      # Try to parse a tool call from the response.
      tool_call_args = self._parse_tool_call(response)

      if tool_call_args is None:
        # No tool call — LLM is done browsing.
        interaction_log.append('[Done browsing — no tool call detected]')
        break

      if tool_call_args == prev_tool_call_args:
        # LLM is stuck repeating the same tool call.
        interaction_log.append(
            f'[Aborting — duplicate tool call: {json.dumps(tool_call_args)}]'
        )
        break

      # Execute the tool call.
      prev_tool_call_args = tool_call_args
      result = tool.execute(**tool_call_args)
      interaction_log.append(
          f'[tool call]: {json.dumps(tool_call_args)}'
      )
      interaction_log.append(f'[tool result]:\n{result}')

      # Summarize everything seen so far before moving on.
      summary_so_far = self._summarize_browsed_content(
          summary_so_far, current_page
      )
      interaction_log.append(f'[summary update]:\n{summary_so_far}')

      # Advance to the new page.
      current_page = result

    # ── Build the return value: summary_so_far + current_page ──
    if summary_so_far:
      full_result = (
          summary_so_far
          + '\n\n' + _LOG_SEPARATOR + '\n\n'
          + current_page
      )
    else:
      full_result = current_page

    # Log the full interaction trace.
    interaction_log.append(_LOG_SEPARATOR)
    interaction_log.append('[Final pre_act value]:')
    interaction_log.append(full_result)

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': full_result,
        'Interaction trace': interaction_log,
    })
    return full_result


class MinimalForumObservation(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Lightweight forum observation that only pushes notifications and status.

  Instead of dumping all forum content into the observation stream, this
  component only delivers:
    1. A timestamp from the clock component (if configured)
    2. Notifications (ban notices, direct messages, reinstatements, vote alerts)
    3. A compact status line telling the agent how many threads exist.
  """

  def __init__(
      self,
      forum_component_key: str = forum_module.DEFAULT_FORUM_COMPONENT_KEY,
      call_to_make_observation: str = (
          forum_module.DEFAULT_CALL_TO_MAKE_OBSERVATION
      ),
      clock_component_key: str = '',
  ):
    super().__init__()
    self._forum_component_key = forum_component_key
    self._call_to_make_observation = call_to_make_observation
    self._clock_component_key = clock_component_key

  def _get_forum_state(self) -> forum_module.ForumState:
    return self.get_entity().get_component(
        self._forum_component_key, type_=forum_module.ForumState
    )

  def _extract_name(self, call_to_action: str) -> str | None:
    """Extract the player name from a call_to_action string.

    Uses the same template-parsing approach as the original
    ForumObservation: splits the call_to_make_observation template on
    {name} and strips the prefix/suffix from the actual call_to_action.

    Args:
      call_to_action: The action string to parse.

    Returns:
      The extracted player name, or None if the format doesn't match.
    """
    prefix, suffix = self._call_to_make_observation.split('{name}')
    if not call_to_action.startswith(prefix):
      return None
    if not call_to_action.endswith(suffix):
      return None
    return call_to_action.removeprefix(prefix).removesuffix(suffix)

  def _get_clock_value(self) -> str:
    """Read the current time from the clock component."""
    if not self._clock_component_key:
      return ''
    try:
      clock = self.get_entity().get_component(
          self._clock_component_key,
          type_=action_spec_ignored.ActionSpecIgnored,
      )
      return clock.get_pre_act_value().strip()
    except KeyError:
      return ''

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      name = self._extract_name(action_spec.call_to_action)
      if name is None:
        return ''

      forum_state = self._get_forum_state()
      parts = []

      # Prepend the current clock time as a timestamp.
      clock_time = self._get_clock_value()
      if clock_time:
        parts.append(f'[{clock_time}]')

      # Push notifications (bans, direct messages, reinstatements, vote alerts).
      notifications = forum_state.drain_notifications(name)
      if notifications:
        parts.extend(notifications)

      # Compact status line.
      posts = forum_state.get_recent_posts()
      total_replies = sum(len(p.replies) for p in posts)
      parts.append(
          f'The forum currently has {len(posts)} threads and'
          f' {total_replies} total replies.'
      )

      result = '\n\n'.join(parts)

    self._logging_channel({
        'Key': 'MinimalForumObservation',
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass

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

"""Post-simulation forum metrics: toxicity, polarization, collaboration.

Computes three categories of metrics from ForumState data after a
simulation completes.  All metrics are derived from existing simulation
data (event logs, vote logs, karma scores, post/reply counts) and
require no additional LLM calls.

Usage:
  from examples.social_media import forum_metrics

  metrics = forum_metrics.compute_all_metrics(forum_state)
  report = forum_metrics.format_metrics_report(metrics)
  forum_metrics.save_metrics_report(metrics, '/path/to/output_metrics.json')
  forum_metrics.save_metrics_html_report(metrics, '/path/to/metrics.html')
"""

import collections
import json
import math
from typing import Any

from concordia.contrib.components.game_master import forum as forum_module  # pylint: disable=g-importing-member


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _gini_coefficient(values: list[float]) -> float:
  """Compute Gini coefficient measuring inequality in a distribution.

  Args:
    values: List of numeric values to measure inequality over.

  Returns:
    A value between 0.0 (perfect equality) and 1.0 (one item has
    everything).  Returns 0.0 for empty or all-zero inputs.
  """
  if not values or all(v == 0 for v in values):
    return 0.0
  sorted_vals = sorted(values)
  n = len(sorted_vals)
  total = sum(sorted_vals)
  if total == 0:
    return 0.0
  cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
  return (2 * cumulative) / (n * total) - (n + 1) / n


def _shannon_entropy(counts: dict[str, int]) -> float:
  """Compute Shannon entropy (in bits) of a count distribution."""
  total = sum(counts.values())
  if total == 0:
    return 0.0
  entropy = 0.0
  for count in counts.values():
    if count > 0:
      p = count / total
      entropy -= p * math.log2(p)
  return entropy


def _std_dev(values: list[float]) -> float:
  """Population standard deviation."""
  if len(values) < 2:
    return 0.0
  mean = sum(values) / len(values)
  variance = sum((v - mean) ** 2 for v in values) / len(values)
  return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Toxicity metrics
# ---------------------------------------------------------------------------


def compute_toxicity_metrics(
    event_log: list[dict[str, Any]],
    karma: dict[str, int],
) -> dict[str, Any]:
  """Compute toxicity metrics from forum data.

  Scans the event log counting events by type and inspects the final
  karma snapshot.  No derived statistics—purely event counting.

  Measures:
    downvote_ratio: total_downvotes / (total_upvotes + total_downvotes).
        Upvotes are counted from 'upvote_post' and 'upvote_reply' events;
        downvotes from 'downvote_post' and 'downvote_reply'.  Range 0–1;
        higher values indicate a more hostile voting culture.  Returns
        0.0 when there are no votes.
    ban_count: Count of 'temp_ban' events in the log.
    parse_error_count: Count of 'parse_error' events—agent actions that
        could not be parsed as valid forum JSON.
    negative_karma_count: Number of agents whose final karma score is
        strictly below zero.
    negative_karma_agents: Names of those agents.
    min_karma: The single lowest final karma score across all agents
        (0 if karma dict is empty).

  Also emits raw counts: total_upvotes, total_downvotes, total_votes.

  Args:
    event_log: Full event log from ForumState.get_full_event_log().
    karma: Final karma scores from ForumState.get_karma().

  Returns:
    Dict of toxicity metric values.
  """
  total_upvotes = 0
  total_downvotes = 0
  ban_count = 0
  parse_errors = 0

  for event in event_log:
    etype = event.get('event_type', '')
    if etype in ('upvote_post', 'upvote_reply'):
      total_upvotes += 1
    elif etype in ('downvote_post', 'downvote_reply'):
      total_downvotes += 1
    elif etype == 'temp_ban':
      ban_count += 1
    elif etype == 'parse_error':
      parse_errors += 1

  total_votes = total_upvotes + total_downvotes
  downvote_ratio = total_downvotes / total_votes if total_votes > 0 else 0.0

  negative_karma_agents = [
      name for name, score in karma.items() if score < 0
  ]
  min_karma = min(karma.values()) if karma else 0

  return {
      'downvote_ratio': round(downvote_ratio, 3),
      'total_upvotes': total_upvotes,
      'total_downvotes': total_downvotes,
      'total_votes': total_votes,
      'ban_count': ban_count,
      'parse_error_count': parse_errors,
      'negative_karma_agents': negative_karma_agents,
      'negative_karma_count': len(negative_karma_agents),
      'min_karma': min_karma,
  }


# ---------------------------------------------------------------------------
# Polarization metrics
# ---------------------------------------------------------------------------


def compute_polarization_metrics(
    posts: list[forum_module.Post],
    karma: dict[str, int],
) -> dict[str, Any]:
  """Compute polarization metrics from forum data.

  Builds a (voter, author) vote matrix from the vote logs attached to
  every post and reply, excluding self-votes.  From this matrix it
  derives per-pair and aggregate polarity scores.

  Measures:
    vote_polarity: For each (voter, author) pair, compute
        |upvotes − downvotes| / (upvotes + downvotes), yielding a
        per-pair polarity in [0, 1].  vote_polarity is the arithmetic
        mean of all per-pair polarities.  1.0 means every voter
        consistently votes the same direction for each author (fully
        polarized camps); 0.0 means voters mix upvotes and downvotes
        equally for every author.  Returns 0.0 when no vote pairs exist.
    karma_spread_std: Population standard deviation of the final karma
        values (σ = sqrt(Σ(xᵢ − μ)² / N)).  Higher values indicate
        some agents were heavily rewarded while others were punished.
    karma_range: Dict with 'min' and 'max' final karma values.
    final_karma: Copy of the full karma dict for per-agent inspection.
    voting_pairs: List of per-pair dicts (voter, author, upvotes,
        downvotes, net_polarity), sorted by net_polarity descending.

  Args:
    posts: All posts from ForumState.get_recent_posts().
    karma: Final karma scores from ForumState.get_karma().

  Returns:
    Dict of polarization metric values.
  """
  # Build voter→author vote counts from post and reply vote logs.
  vote_pairs: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(
      lambda: {'up': 0, 'down': 0}
  )

  for post in posts:
    author = post.author
    for vote in post.vote_log:
      voter = vote.get('voter', '')
      direction = vote.get('direction', '')
      if voter and voter != author and direction in ('up', 'down'):
        vote_pairs[(voter, author)][direction] += 1

    for reply in post.replies:
      reply_author = str(reply.get('author', ''))
      for vote in reply.get('vote_log', []):
        voter = vote.get('voter', '')
        direction = vote.get('direction', '')
        if voter and voter != reply_author and direction in ('up', 'down'):
          vote_pairs[(voter, reply_author)][direction] += 1

  # Compute polarity per pair.
  pair_details = []
  polarities = []
  for (voter, author), counts in vote_pairs.items():
    total = counts['up'] + counts['down']
    if total > 0:
      net_polarity = abs(counts['up'] - counts['down']) / total
      polarities.append(net_polarity)
      pair_details.append({
          'voter': voter,
          'author': author,
          'upvotes': counts['up'],
          'downvotes': counts['down'],
          'net_polarity': round(net_polarity, 3),
      })

  vote_polarity = (
      sum(polarities) / len(polarities) if polarities else 0.0
  )

  # Karma spread.
  karma_values = list(karma.values())
  karma_std = _std_dev([float(v) for v in karma_values])

  return {
      'vote_polarity': round(vote_polarity, 3),
      'karma_spread_std': round(karma_std, 3),
      'karma_range': {
          'min': min(karma_values) if karma_values else 0,
          'max': max(karma_values) if karma_values else 0,
      },
      'final_karma': dict(karma),
      'voting_pairs': sorted(
          pair_details, key=lambda x: x['net_polarity'], reverse=True
      ),
  }


# ---------------------------------------------------------------------------
# Collaboration metrics
# ---------------------------------------------------------------------------


def compute_collaboration_metrics(
    event_log: list[dict[str, Any]],
    posts: list[forum_module.Post],
    player_names: list[str],
) -> dict[str, Any]:
  """Compute constructive collaboration metrics from forum data.

  Combines event-log scanning (for action diversity) with post-level
  analysis (for reply depth and thread engagement).  System events
  ('karma_snapshot', 'reinstatement') and '[system]' actors are
  excluded from action counts.

  Measures:
    avg_replies_per_post: total_replies / total_posts.  Higher values
        indicate deeper, more engaged discussions.  0.0 if no posts.
    action_diversity_entropy: For each agent, count occurrences of
        each event_type and compute Shannon entropy
        H = −Σ pᵢ log₂(pᵢ) in bits.  The reported value is the
        arithmetic mean across all agents in player_names.  Higher
        values mean agents use a wider variety of actions.
    participation_equality: 1 − Gini(content_per_agent), where
        content_per_agent is each agent's (posts + replies) count.
        The Gini coefficient is computed as
        G = (2 Σᵢ (i+1)·xᵢ) / (n·Σxᵢ) − (n+1)/n  over the sorted
        values.  Result range: 1.0 = perfectly equal contribution,
        0.0 = one agent produces everything.
    content_ratio: content_events / total_actions, where content
        events are 'post' and 'reply' event types.  Range 0–1.
    avg_threads_engaged: For each agent, count distinct post_ids
        they authored or replied to.  Report the mean across agents.

  Also emits per-agent breakdowns: agent_content (posts/replies per
  agent), agent_action_summaries (action counts, entropy, total per
  agent), and threads_engaged_per_agent.

  Args:
    event_log: Full event log from ForumState.get_full_event_log().
    posts: All posts from ForumState.get_recent_posts().
    player_names: List of agent names.

  Returns:
    Dict of collaboration metric values.
  """
  # Reply depth.
  total_posts = len(posts)
  total_replies = sum(len(post.replies) for post in posts)
  avg_replies = total_replies / total_posts if total_posts > 0 else 0.0

  # Action diversity per agent.
  agent_actions: dict[str, dict[str, int]] = collections.defaultdict(
      lambda: collections.defaultdict(int)
  )
  content_events = 0
  vote_events = 0
  total_actions = 0

  _system_event_types = frozenset(
      {'karma_snapshot', 'reinstatement'}
  )

  for event in event_log:
    etype = event.get('event_type', '')
    actor = event.get('actor', '')
    if etype in _system_event_types or actor == '[system]':
      continue
    total_actions += 1
    agent_actions[actor][etype] += 1
    if etype in ('post', 'reply'):
      content_events += 1
    elif etype in (
        'upvote_post', 'downvote_post',
        'upvote_reply', 'downvote_reply',
    ):
      vote_events += 1

  # Average entropy across known agents.
  entropies = []
  agent_action_summaries = {}
  for name in player_names:
    actions = dict(agent_actions.get(name, {}))
    h = _shannon_entropy(actions)
    entropies.append(h)
    agent_action_summaries[name] = {
        'action_counts': actions,
        'entropy': round(h, 3),
        'total': sum(actions.values()),
    }

  avg_entropy = (
      sum(entropies) / len(entropies) if entropies else 0.0
  )

  # Participation equality (posts + replies per agent).
  content_per_agent = []
  agent_content = {}
  for name in player_names:
    post_count = sum(1 for post in posts if post.author == name)
    reply_count = 0
    for post in posts:
      for reply in post.replies:
        if reply.get('author') == name:
          reply_count += 1
    content_per_agent.append(post_count + reply_count)
    agent_content[name] = {
        'posts': post_count,
        'replies': reply_count,
        'total': post_count + reply_count,
    }

  gini = _gini_coefficient([float(c) for c in content_per_agent])
  participation_equality = round(1.0 - gini, 3)

  # Content ratio.
  content_ratio = (
      content_events / total_actions if total_actions > 0 else 0.0
  )

  # Threads engaged per agent.
  agent_threads: dict[str, set[int]] = collections.defaultdict(set)
  for post in posts:
    agent_threads[post.author].add(post.post_id)
    for reply in post.replies:
      reply_author = str(reply.get('author', ''))
      agent_threads[reply_author].add(post.post_id)

  threads_engaged = {
      name: len(agent_threads.get(name, set()))
      for name in player_names
  }
  avg_threads = (
      sum(threads_engaged.values()) / len(threads_engaged)
      if threads_engaged
      else 0.0
  )

  return {
      'total_posts': total_posts,
      'total_replies': total_replies,
      'avg_replies_per_post': round(avg_replies, 2),
      'action_diversity_entropy': round(avg_entropy, 3),
      'participation_equality': participation_equality,
      'content_ratio': round(content_ratio, 3),
      'total_content_events': content_events,
      'total_vote_events': vote_events,
      'total_actions': total_actions,
      'agent_content': agent_content,
      'agent_action_summaries': agent_action_summaries,
      'threads_engaged_per_agent': threads_engaged,
      'avg_threads_engaged': round(avg_threads, 2),
  }


# ---------------------------------------------------------------------------
# Aggregate and reporting
# ---------------------------------------------------------------------------


def compute_all_metrics(
    forum_state: forum_module.ForumState,
) -> dict[str, Any]:
  """Compute all metrics from a ForumState.

  Args:
    forum_state: The ForumState after simulation completes.

  Returns:
    Dict with 'toxicity', 'polarization', and 'collaboration' sub-dicts.
  """
  event_log = forum_state.get_full_event_log()
  posts = forum_state.get_recent_posts()
  karma = forum_state.get_karma()
  player_names = list(karma.keys())

  return {
      'toxicity': compute_toxicity_metrics(event_log, karma),
      'polarization': compute_polarization_metrics(posts, karma),
      'collaboration': compute_collaboration_metrics(
          event_log, posts, player_names
      ),
  }


def format_metrics_report(metrics: dict[str, Any]) -> str:
  """Format metrics as a human-readable text report.

  Args:
    metrics: Output from compute_all_metrics().

  Returns:
    Formatted multi-line string.
  """
  tox = metrics['toxicity']
  pol = metrics['polarization']
  col = metrics['collaboration']

  lines = []
  lines.append('=' * 62)
  lines.append('  FORUM SIMULATION METRICS REPORT')
  lines.append('=' * 62)

  # ── Toxicity ──
  lines.append('')
  lines.append('── TOXICITY ──')
  lines.append(
      f'  Downvote ratio:         {tox["downvote_ratio"]:.1%}'
      f'  ({tox["total_downvotes"]} down / {tox["total_votes"]} total votes)'
  )
  lines.append(f'  Ban count:              {tox["ban_count"]}')
  lines.append(f'  Parse errors:           {tox["parse_error_count"]}')
  lines.append(
      f'  Agents with neg. karma: {tox["negative_karma_count"]}'
      f'  {tox["negative_karma_agents"]}'
  )
  lines.append(f'  Lowest karma reached:   {tox["min_karma"]}')

  # ── Polarization ──
  lines.append('')
  lines.append('── POLARIZATION ──')
  lines.append(
      f'  Vote polarity:          {pol["vote_polarity"]:.3f}'
      '  (0=balanced, 1=fully polarized)'
  )
  lines.append(f'  Karma spread (σ):       {pol["karma_spread_std"]:.2f}')
  lines.append(
      f'  Karma range:            {pol["karma_range"]["min"]}'
      f' to {pol["karma_range"]["max"]}'
  )
  lines.append('  Final karma:')
  for name, score in sorted(pol['final_karma'].items(), key=lambda x: -x[1]):
    lines.append(f'    {name:>35s}:  {score:+d}')

  if pol['voting_pairs']:
    lines.append('  Voting patterns:')
    for pair in pol['voting_pairs']:
      direction_str = (
          f'↑{pair["upvotes"]} ↓{pair["downvotes"]}'
      )
      lines.append(
          f'    {pair["voter"]:>30s} → {pair["author"]:<25s}'
          f'  {direction_str:<10s}'
          f'  polarity={pair["net_polarity"]:.2f}'
      )

  # ── Collaboration ──
  lines.append('')
  lines.append('── COLLABORATION ──')
  lines.append(
      f'  Total posts:            {col["total_posts"]}'
  )
  lines.append(
      f'  Total replies:          {col["total_replies"]}'
  )
  lines.append(
      f'  Avg replies/post:       {col["avg_replies_per_post"]:.1f}'
  )
  lines.append(
      f'  Action diversity (H):   {col["action_diversity_entropy"]:.2f} bits'
  )
  lines.append(
      f'  Participation equality: {col["participation_equality"]:.3f}'
      '  (0=dominated, 1=equal)'
  )
  lines.append(
      f'  Content ratio:          {col["content_ratio"]:.1%}'
      f'  ({col["total_content_events"]} content /'
      f' {col["total_actions"]} total actions)'
  )
  lines.append(
      f'  Avg threads engaged:    {col["avg_threads_engaged"]:.1f}'
  )

  if col.get('agent_content'):
    lines.append('  Per-agent breakdown:')
    for name in sorted(col['agent_content'].keys()):
      c = col['agent_content'][name]
      threads = col['threads_engaged_per_agent'].get(name, 0)
      summary = col.get('agent_action_summaries', {}).get(name, {})
      total_acts = summary.get('total', 0)
      entropy = summary.get('entropy', 0.0)
      lines.append(
          f'    {name:>30s}:  {c["posts"]}p {c["replies"]}r'
          f'  | {total_acts} acts  H={entropy:.2f}'
          f'  | {threads} threads'
      )

  lines.append('')
  lines.append('=' * 62)

  return '\n'.join(lines)


def save_metrics_report(
    metrics: dict[str, Any],
    output_path: str,
) -> str:
  """Save metrics as JSON and print a text summary.

  Args:
    metrics: Output from compute_all_metrics().
    output_path: File path (with .json extension) to save the report.

  Returns:
    The output file path.
  """
  with open(output_path, 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

  report = format_metrics_report(metrics)
  print(f'Metrics saved to: {output_path}')
  print()
  print(report)

  return output_path


def format_metrics_html_report(metrics: dict[str, Any]) -> str:
  """Format metrics as a self-contained HTML report.

  Produces a styled HTML document with sections for toxicity, polarization,
  and collaboration metrics, including tables and color-coded indicators.

  Args:
    metrics: Output from compute_all_metrics().

  Returns:
    Complete HTML document string.
  """
  tox = metrics['toxicity']
  pol = metrics['polarization']
  col = metrics['collaboration']

  def _pct(value: float) -> str:
    return f'{value:.1%}'

  def _severity_class(value: float, thresholds: tuple[float, float]) -> str:
    """Return CSS class based on value severity."""
    if value <= thresholds[0]:
      return 'good'
    elif value <= thresholds[1]:
      return 'warn'
    return 'bad'

  # Build karma table rows.
  karma_rows = ''
  for name, score in sorted(
      pol['final_karma'].items(), key=lambda x: -x[1]
  ):
    css_class = 'good' if score > 0 else ('warn' if score == 0 else 'bad')
    karma_rows += (
        f'<tr><td>{name}</td>'
        f'<td class="{css_class}">{score:+d}</td></tr>\n'
    )

  # Build voting pairs table rows.
  voting_rows = ''
  for pair in pol.get('voting_pairs', []):
    polarity_class = _severity_class(
        pair['net_polarity'], (0.5, 0.8)
    )
    voting_rows += (
        f'<tr><td>{pair["voter"]}</td>'
        f'<td>{pair["author"]}</td>'
        f'<td>{pair["upvotes"]}</td>'
        f'<td>{pair["downvotes"]}</td>'
        f'<td class="{polarity_class}">{pair["net_polarity"]:.2f}</td></tr>\n'
    )

  # Build per-agent content table rows.
  agent_rows = ''
  if col.get('agent_content'):
    for name in sorted(col['agent_content'].keys()):
      c = col['agent_content'][name]
      threads = col['threads_engaged_per_agent'].get(name, 0)
      summary = col.get('agent_action_summaries', {}).get(name, {})
      total_acts = summary.get('total', 0)
      entropy = summary.get('entropy', 0.0)
      agent_rows += (
          f'<tr><td>{name}</td>'
          f'<td>{c["posts"]}</td>'
          f'<td>{c["replies"]}</td>'
          f'<td>{c["total"]}</td>'
          f'<td>{total_acts}</td>'
          f'<td>{entropy:.2f}</td>'
          f'<td>{threads}</td></tr>\n'
      )

  # Negative karma agents list.
  neg_karma_list = ', '.join(tox['negative_karma_agents']) or 'None'

  # Severity classes for key indicators.
  downvote_class = _severity_class(tox['downvote_ratio'], (0.3, 0.5))
  polarity_class = _severity_class(pol['vote_polarity'], (0.5, 0.8))
  equality_class = _severity_class(
      1.0 - col['participation_equality'], (0.3, 0.5)
  )

  html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forum Simulation Metrics Report</title>
<style>
  :root {{
    --bg: #fafafa;
    --card-bg: #ffffff;
    --border: #e0e0e0;
    --text: #212121;
    --text-secondary: #616161;
    --accent-tox: #c62828;
    --accent-pol: #e65100;
    --accent-col: #2e7d32;
    --good: #2e7d32;
    --warn: #f57f17;
    --bad: #c62828;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  h1 {{
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    border-bottom: 3px solid var(--text);
    padding-bottom: 0.5rem;
  }}
  .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; }}
  .section {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .section h2 {{
    font-size: 1.3rem;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--border);
  }}
  .section-toxicity h2 {{ border-bottom-color: var(--accent-tox); }}
  .section-polarization h2 {{ border-bottom-color: var(--accent-pol); }}
  .section-collaboration h2 {{ border-bottom-color: var(--accent-col); }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
  }}
  .metric-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }}
  .metric-card .label {{
    font-size: 0.8rem;
    text-transform: uppercase;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
  }}
  .metric-card .value {{
    font-size: 1.6rem;
    font-weight: 700;
  }}
  .metric-card .detail {{
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.2rem;
  }}
  .good {{ color: var(--good); }}
  .warn {{ color: var(--warn); }}
  .bad  {{ color: var(--bad); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.5rem 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: var(--bg);
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
  }}
  tr:last-child td {{ border-bottom: none; }}
  .footer {{
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-top: 2rem;
  }}
</style>
</head>
<body>

<h1>Forum Simulation Metrics Report</h1>
<p class="subtitle">Toxicity · Polarization · Collaboration</p>

<!-- ── Toxicity ── -->
<div class="section section-toxicity">
  <h2>Toxicity</h2>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="label">Downvote Ratio</div>
      <div class="value {downvote_class}">{_pct(tox['downvote_ratio'])}</div>
      <div class="detail">{tox['total_downvotes']} down / {tox['total_votes']} total</div>
    </div>
    <div class="metric-card">
      <div class="label">Ban Count</div>
      <div class="value">{tox['ban_count']}</div>
    </div>
    <div class="metric-card">
      <div class="label">Parse Errors</div>
      <div class="value">{tox['parse_error_count']}</div>
    </div>
    <div class="metric-card">
      <div class="label">Negative Karma Agents</div>
      <div class="value {'bad' if tox['negative_karma_count'] > 0 else 'good'}">{tox['negative_karma_count']}</div>
      <div class="detail">{neg_karma_list}</div>
    </div>
    <div class="metric-card">
      <div class="label">Lowest Karma</div>
      <div class="value {'bad' if tox['min_karma'] < 0 else 'good'}">{tox['min_karma']:+d}</div>
    </div>
    <div class="metric-card">
      <div class="label">Total Upvotes</div>
      <div class="value">{tox['total_upvotes']}</div>
    </div>
  </div>
</div>

<!-- ── Polarization ── -->
<div class="section section-polarization">
  <h2>Polarization</h2>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="label">Vote Polarity</div>
      <div class="value {polarity_class}">{pol['vote_polarity']:.3f}</div>
      <div class="detail">0 = balanced, 1 = fully polarized</div>
    </div>
    <div class="metric-card">
      <div class="label">Karma Spread (σ)</div>
      <div class="value">{pol['karma_spread_std']:.2f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Karma Range</div>
      <div class="value">{pol['karma_range']['min']} to {pol['karma_range']['max']}</div>
    </div>
  </div>

  <h3 style="margin: 1rem 0 0.5rem; font-size: 1rem;">Final Karma</h3>
  <table>
    <thead><tr><th>Agent</th><th>Karma</th></tr></thead>
    <tbody>
{karma_rows}    </tbody>
  </table>

  {'<h3 style="margin: 1rem 0 0.5rem; font-size: 1rem;">Voting Patterns</h3>' if voting_rows else ''}
  {'<table><thead><tr><th>Voter</th><th>Author</th><th>↑</th><th>↓</th><th>Polarity</th></tr></thead><tbody>' + voting_rows + '</tbody></table>' if voting_rows else ''}
</div>

<!-- ── Collaboration ── -->
<div class="section section-collaboration">
  <h2>Collaboration</h2>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="label">Total Posts</div>
      <div class="value">{col['total_posts']}</div>
    </div>
    <div class="metric-card">
      <div class="label">Total Replies</div>
      <div class="value">{col['total_replies']}</div>
    </div>
    <div class="metric-card">
      <div class="label">Avg Replies / Post</div>
      <div class="value">{col['avg_replies_per_post']:.1f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Action Diversity (H)</div>
      <div class="value">{col['action_diversity_entropy']:.2f}</div>
      <div class="detail">bits</div>
    </div>
    <div class="metric-card">
      <div class="label">Participation Equality</div>
      <div class="value {equality_class}">{col['participation_equality']:.3f}</div>
      <div class="detail">0 = dominated, 1 = equal</div>
    </div>
    <div class="metric-card">
      <div class="label">Content Ratio</div>
      <div class="value">{_pct(col['content_ratio'])}</div>
      <div class="detail">{col['total_content_events']} content / {col['total_actions']} actions</div>
    </div>
    <div class="metric-card">
      <div class="label">Avg Threads Engaged</div>
      <div class="value">{col['avg_threads_engaged']:.1f}</div>
    </div>
  </div>

  {'<h3 style="margin: 1rem 0 0.5rem; font-size: 1rem;">Per-Agent Breakdown</h3>' if agent_rows else ''}
  {'<table><thead><tr><th>Agent</th><th>Posts</th><th>Replies</th><th>Total Content</th><th>Total Actions</th><th>Entropy (H)</th><th>Threads</th></tr></thead><tbody>' + agent_rows + '</tbody></table>' if agent_rows else ''}
</div>

<div class="footer">
  Generated by forum_metrics · Concordia Social Media Simulation
</div>

</body>
</html>"""

  return html


def save_metrics_html_report(
    metrics: dict[str, Any],
    output_path: str,
) -> str:
  """Save metrics as an HTML report.

  Args:
    metrics: Output from compute_all_metrics().
    output_path: File path (with .html extension) to save the report.

  Returns:
    The output file path.
  """
  html = format_metrics_html_report(metrics)
  with open(output_path, 'w') as f:
    f.write(html)
  print(f'Metrics HTML report saved to: {output_path}')
  return output_path

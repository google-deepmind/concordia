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

"""Log printing functions for interrupt-driven scenarios."""

from collections.abc import Mapping, Sequence
import datetime
import textwrap
from typing import Any

from concordia.components.game_master import interrupt_response_parsing


def format_mask(mask_prefixes: list[str]) -> str:
  """Format an InterruptMask prefix list for human-readable display."""
  if mask_prefixes == ['']:
    return '[""]  (matches all events)'
  elif not mask_prefixes:
    return '[]  (matches no events, only timers interrupt)'
  else:
    return f'[{", ".join(mask_prefixes)}]'


def format_duration(td: datetime.timedelta) -> str:
  """Format a timedelta as a compact duration string like '1h30m'."""
  total_minutes = int(td.total_seconds()) // 60
  if total_minutes == 0:
    return '0m'
  hours, minutes = divmod(total_minutes, 60)
  parts = []
  if hours:
    parts.append(f'{hours}h')
  if minutes:
    parts.append(f'{minutes}m')
  return ''.join(parts)


def print_log(
    log: Sequence[Mapping[str, Any]],
    gm_name: str = 'interrupt_driven_rules',
) -> None:
  """Walk the structured engine log and print a readable trace."""
  for entry in log:
    step = entry.get('Step', '?')
    print(f'\n{"─── Step " + str(step) + " ":─<56}')

    gm_log = entry.get(gm_name, {})
    print_next_acting(gm_log)
    print_entity_actions(entry)
    print_resolution(gm_log)


def print_next_acting(gm_log: Mapping[str, Any]) -> None:
  """Print what event fired and who was polled."""
  na = gm_log.get('next_acting', {})
  for _, component_log in na.items():
    if not isinstance(component_log, dict):
      continue
    # Only process the _InterruptNextActing component.
    event_info = component_log.get('Event')
    if not event_info:
      continue

    # ── Initial events (only present on step 1) ─────────────────
    initial_events = component_log.get('Initial Events')
    if initial_events:
      print('  ── Initial event queue ──')
      for ev in initial_events:
        ts = ev.get('timestamp', '')
        time_part = ts.split('T')[1][:5] if 'T' in ts else ts
        print(
            f'    [{time_part}] {ev.get("tag", "?")} '
            f'(from {ev.get("source", "?")})'
        )
        desc = ev.get('description', '')
        if desc:
          wrapped = textwrap.fill(
              f'"{desc}"',
              width=48,
              initial_indent='           ',
              subsequent_indent='           ',
          )
          print(wrapped)
      print()

    # ── Initial entity states (only present on step 1) ──────────
    initial_states = component_log.get('Initial Entity States')
    if initial_states:
      print('  ── Initial entity states ──')
      for name, state in initial_states.items():
        mask_prefixes = state.get('mask', [''])
        mask_str = format_mask(mask_prefixes)
        timer_expiry = state.get('timer_expiry', '')
        timer_desc = state.get('timer_description', '')
        timer_str = ''
        if timer_expiry and 'T' in timer_expiry:
          timer_str = timer_expiry.split('T')[1][:5]
        parts = [f'    {name}:  mask={mask_str}']
        if timer_str:
          parts.append(f'timer={timer_str} ({timer_desc})')
        print('  '.join(parts))
      print()

    # ── Current step event ───────────────────────────────────────
    tag = event_info.get('tag', '?')
    source = event_info.get('source', '?')
    desc = event_info.get('description', '')
    timestamp = event_info.get('timestamp', '')
    time_str = ''
    if 'T' in timestamp:
      time_str = timestamp.split('T')[1][:5]
    print(f'  Time:   {time_str}')
    print(f'  Event:  {tag}  (from {source})')
    if desc:
      wrapped = textwrap.fill(
          f'"{desc}"',
          width=52,
          initial_indent='          ',
          subsequent_indent='          ',
      )
      print(wrapped)
    polled = component_log.get('Value', '')
    if polled:
      print(f'  Polled: {polled}')


def print_entity_actions(entry: Mapping[str, Any]) -> None:
  """Print what each entity said and its parsed mask/timer."""
  for key, value in entry.items():
    if not key.startswith('Entity ['):
      continue
    entity_name = key[len('Entity [') : -1]
    act_log = value.get('__act__', {}) if isinstance(value, dict) else {}
    raw = act_log.get('Value', '') if isinstance(act_log, dict) else ''
    if not raw:
      continue

    action_body = raw

    # Parse the JSON block for display.
    parsed = interrupt_response_parsing.parse_entity_response(action_body)
    mask_str = format_mask(list(parsed.mask.prefixes))
    timer_desc = parsed.timer_description or '(none)'
    if parsed.timer_absolute is not None:
      timer_time_str = f'until {parsed.timer_absolute}'
    else:
      timer_time_str = parsed.timer_duration_str

    print(f'  {entity_name} responded:')
    if parsed.action_text:
      wrapped = textwrap.fill(
          parsed.action_text,
          width=52,
          initial_indent='    action: ',
          subsequent_indent='            ',
      )
      print(wrapped)
    print(f'    mask:   {mask_str}')
    if parsed.event_tags:
      print(f'    tags:   [{", ".join(parsed.event_tags)}]')
    print(f'    timer:  {timer_time_str}  ({timer_desc})')


def print_resolution(gm_log: Mapping[str, Any]) -> None:
  """Print what the GM resolved."""
  res = gm_log.get('resolve', {})
  for _, component_log in res.items():
    if not isinstance(component_log, dict):
      continue
    resolved = component_log.get('Resolved Entities', [])
    if resolved:
      print(f'  Resolved: {", ".join(resolved)}')

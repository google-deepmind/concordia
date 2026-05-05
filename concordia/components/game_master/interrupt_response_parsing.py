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

"""Parsing utilities for interrupt-driven entity responses.

Entities in interrupt-driven simulations respond with free-text action
descriptions followed by a JSON block specifying attention mask, timer, and
optional event tags.  This module extracts and validates those structured
fields from raw response strings.
"""

import dataclasses
import json
from typing import Any

from concordia.components.game_master import interrupt_scheduling


def extract_json_block(text: str) -> tuple[str, dict[str, Any]]:
  """Extracts a trailing JSON object from entity response text.

  Finds the last balanced ``{...}`` block in *text*, parses it as JSON,
  and returns the preceding text (the action) together with the parsed
  dict.

  Args:
    text: The raw entity response string, potentially containing a trailing JSON
      object.

  Returns:
    A ``(action_text, data)`` tuple.  If no valid JSON object is found,
    returns ``(text.strip(), {})``.
  """
  end = text.rfind('}')
  if end == -1:
    return text.strip(), {}
  # Walk backward from `end`, counting braces to find matching '{'.
  depth = 0
  for i in range(end, -1, -1):
    if text[i] == '}':
      depth += 1
    elif text[i] == '{':
      depth -= 1
      if depth == 0:
        try:
          data = json.loads(text[i : end + 1])
          if isinstance(data, dict):
            return text[:i].strip(), data
        except json.JSONDecodeError:
          pass
        return text.strip(), {}
  return text.strip(), {}


@dataclasses.dataclass(frozen=True)
class ParsedResponse:
  """A parsed entity response.

  Attributes:
    action_text: What the entity did/said (may be empty).
    mask: The new interrupt mask.
    timer_duration_str: Raw duration string until the next timer fires (e.g.
      ``'30m'``, ``'2h'``). The ``TimeModel`` is responsible for interpreting
      this string into a timestamp offset.
    timer_description: Human-readable reason for the timer.
    event_tags: Extra event tags to broadcast for this action. Each tag produces
      a separate Event that other entities can match via their interrupt masks.
      The default ``action.<EntityName>`` event is always emitted in addition to
      these.
    timer_absolute: Optional absolute time string for the timer (e.g.
      ``'8:00 AM'``). Future hook — not yet acted on by the resolution
      component.
  """

  action_text: str
  mask: interrupt_scheduling.InterruptMask
  timer_duration_str: str | None = None
  timer_description: str = 'default timer'
  event_tags: tuple[str, ...] = ()
  timer_absolute: str | None = None


def parse_entity_response(response: str) -> ParsedResponse:
  """Parses an entity response into action text, mask, and timer.

  The entity response should be free-text action description followed by
  a JSON object with ``mask``, ``timer``, and optionally ``tags`` fields.
  If the JSON block is missing, defaults are applied (match-all mask,
  1-hour timer, no tags).

  Args:
    response: The raw response string from an entity.

  Returns:
    A ParsedResponse with the extracted fields.
  """
  action_text, data = extract_json_block(response)

  # Mask.
  mask = interrupt_scheduling.mask_from_prefixes(data.get('mask', ['']))

  # Timer.
  timer_data = data.get('timer', {})
  timer_duration_str = timer_data.get('time')  # None if not specified.
  timer_description = timer_data.get('reason', 'default timer')
  timer_absolute = timer_data.get('until')

  # Tags.
  event_tags = tuple(data.get('tags', []))

  return ParsedResponse(
      action_text=action_text,
      mask=mask,
      timer_duration_str=timer_duration_str,
      timer_description=timer_description,
      event_tags=event_tags,
      timer_absolute=timer_absolute,
  )

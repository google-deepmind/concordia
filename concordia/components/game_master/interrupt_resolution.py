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

"""Resolution component for interrupt-driven game master orchestration.

Resolves entity actions into environment mutations and scheduling updates
by parsing entity responses, updating masks and timers, and injecting
action events into the shared event queue.
"""

from collections.abc import Sequence

from absl import logging
from concordia.components import agent as actor_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import interrupt_response_parsing
from concordia.components.game_master import interrupt_scheduling
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG


class InterruptResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Resolves entity actions into environment mutations and scheduling updates.

  For each entity action in the putative event:
  1. Parses the response to extract action text, mask, and timer.
  2. Updates the entity's mask and timer in the scheduler.
  3. If the action text is non-empty, creates an Event and injects it.

  The resolved event description is returned for the engine to observe.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      event_tag_for_actions: str = 'action',
      scheduler_component_key: str = (
          interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY
      ),
      memory_component_key: str = (
          actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str = (
          event_resolution_components.DEFAULT_RESOLUTION_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._event_tag_for_actions = event_tag_for_actions
    self._scheduler_key = scheduler_component_key
    self._memory_component_key = memory_component_key
    self._pre_act_label = pre_act_label
    self._active_entity_name: str | None = None
    self._putative_action: str | None = None

  def _get_scheduler(self) -> interrupt_scheduling.EntityScheduler:
    return self.get_entity().get_component(
        self._scheduler_key, type_=interrupt_scheduling.EntityScheduler
    )

  def get_active_entity_name(self) -> str | None:
    """Returns the name of the last entity whose action was resolved."""
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    """Returns the last putative action text."""
    return self._putative_action

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''

    scheduler = self._get_scheduler()

    # Read putative events from memory (stored by the engine).
    memory = self.get_entity().get_component(
        self._memory_component_key,
        type_=actor_components.memory.Memory,
    )
    suggestions = memory.scan(selector_fn=lambda x: PUTATIVE_EVENT_TAG in x)

    if not suggestions:
      self._logging_channel({
          'Key': self._pre_act_label,
          'Summary': 'No putative events to resolve.',
          'Value': '',
      })
      return ''

    # Take the most recent suggestion.
    selected = suggestions[-1]
    putative_text = selected[
        selected.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG) :
    ].strip()

    # The putative text may contain multiple entity actions joined by \n.
    # Process each polled entity's action.  Non-polled entities may also
    # appear in the putative text (the Simultaneous engine acts all
    # entities every step), but their responses should be ignored.
    polled = set(scheduler.get_current_polled_entities())
    resolved_parts = []
    for entity_name in self._player_names:
      if entity_name not in polled:
        continue
      # Find this entity's action in the putative text.  The
      # "{name}:" prefix we look for here is added by the Simultaneous
      # engine (simultaneous.py), which unconditionally wraps each
      # entity's raw action as "{name}: {raw_action}" before joining
      # all actions into the putative event string.  This is unrelated
      # to the prefix_entity_name setting on acting components (which
      # must be False — see below — so that parse_entity_response
      # receives the raw response without a redundant name prefix).
      prefix = f'{entity_name}:'
      if prefix not in putative_text:
        continue

      # Extract this entity's response.
      start = putative_text.index(prefix)
      raw_response = putative_text[start + len(prefix) :]
      # Trim at the next entity's action (if any).
      # Note: the memory bank replaces newlines with spaces, so we
      # look for ' {name}:' rather than '\n{name}:'.
      for other_name in self._player_names:
        if other_name != entity_name:
          for sep in (f'\n{other_name}:', f' {other_name}:'):
            if sep in raw_response:
              end_idx = raw_response.index(sep)
              raw_response = raw_response[:end_idx]
              break

      raw_response = raw_response.strip()
      # Acting components used with this prefab must not prepend the entity
      # name to responses (e.g., ConcatActComponent with
      # prefix_entity_name=False).
      parsed = interrupt_response_parsing.parse_entity_response(raw_response)

      self._active_entity_name = entity_name
      self._putative_action = parsed.action_text

      # Update mask.
      scheduler.set_mask(entity_name, parsed.mask)

      # Set timer.
      now = scheduler.get_current_time()
      expiry = now  # Default: fire immediately.
      if parsed.timer_absolute is not None:
        try:
          expiry = scheduler.time_model.parse_absolute_time(
              parsed.timer_absolute, now
          )
        except (ValueError, NotImplementedError):
          logging.warning(
              'Failed to parse absolute time %r for entity %s;'
              ' falling back to duration.',
              parsed.timer_absolute,
              entity_name,
          )
          if parsed.timer_duration_str is not None:
            expiry = scheduler.time_model.add_duration(
                now, parsed.timer_duration_str
            )
      elif parsed.timer_duration_str is not None:
        expiry = scheduler.time_model.add_duration(
            now, parsed.timer_duration_str
        )
      timer = interrupt_scheduling.Timer(
          expiry=expiry,
          entity_name=entity_name,
          description=parsed.timer_description,
      )
      scheduler.set_timer(entity_name, timer)

      # Inject action event if the entity did something.
      if parsed.action_text:
        action_event = interrupt_scheduling.Event(
            timestamp=scheduler.get_current_time(),
            tag=f'{self._event_tag_for_actions}.{entity_name}',
            source=entity_name,
            description=f'{entity_name}: {parsed.action_text}',
        )
        scheduler.inject_event(action_event)

        # Inject extra tagged events.
        for extra_tag in parsed.event_tags:
          tagged_event = interrupt_scheduling.Event(
              timestamp=scheduler.get_current_time(),
              tag=extra_tag,
              source=entity_name,
              description=f'{entity_name}: {parsed.action_text}',
          )
          scheduler.inject_event(tagged_event)

        resolved_parts.append(f'{entity_name}: {parsed.action_text}')

    result = '\n'.join(resolved_parts) if resolved_parts else ''
    result_for_pre_act = f'{self._pre_act_label}: {result}\n' if result else ''

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'Resolved Entities': [p.split(':')[0] for p in resolved_parts],
    })
    return result_for_pre_act

  def get_state(self) -> entity_component.ComponentState:
    return {
        '_active_entity_name': self._active_entity_name,
        '_putative_action': self._putative_action,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    active_name = state.get('_active_entity_name')
    self._active_entity_name = (
        str(active_name) if active_name is not None else None
    )
    putative = state.get('_putative_action')
    self._putative_action = str(putative) if putative is not None else None

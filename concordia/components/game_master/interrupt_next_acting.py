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

"""Next-acting component for interrupt-driven game master orchestration.

Selects which entities to poll on each simulation step by inspecting
interrupt masks, timers, and the shared event queue.
"""

from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import next_acting as next_acting_components
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class InterruptNextActing(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Selects which entities to poll based on interrupt masks and timers.

  On each NEXT_ACTING call:
  1. Finds the next event to process (earliest of queued event vs timer).
  2. Advances simulated time to that event's timestamp.
  3. Determines which entities to poll (mask match or timer expiry).
  4. Calls poll_entity() for each (clears mask and timer).
  5. Adds the event to pending observations for non-polled entities.
  6. Returns comma-separated names of polled entities.
  """

  def __init__(
      self,
      scheduler_component_key: str = (
          interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY
      ),
      pre_act_label: str = (
          next_acting_components.DEFAULT_NEXT_ACTING_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._scheduler_key = scheduler_component_key
    self._pre_act_label = pre_act_label
    self._logged_initial_state: bool = False

  def _get_scheduler(self) -> interrupt_scheduling.EntityScheduler:
    return self.get_entity().get_component(
        self._scheduler_key, type_=interrupt_scheduling.EntityScheduler
    )

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''

    scheduler = self._get_scheduler()

    # On the first call, snapshot the event queue and entity states
    # before any events are consumed.  Because InterruptNextActing
    # overwrites its channel every step, these initial-state entries
    # will naturally disappear from the log after step 1 — no
    # downstream "print once" guard is needed.
    initial_events_snapshot: list[dict[str, str]] | None = None
    initial_entity_states: dict[str, dict[str, object]] | None = None
    if not self._logged_initial_state:
      self._logged_initial_state = True
      queue = scheduler.peek_event_queue()
      if queue:
        initial_events_snapshot = [
            {
                'timestamp': scheduler.time_model.serialize_time(e.timestamp),
                'tag': e.tag,
                'source': e.source,
                'description': e.description,
            }
            for e in queue
        ]
      # Capture each entity's mask and timer.
      entity_states: dict[str, dict[str, object]] = {}
      for name in scheduler.player_names:
        state: dict[str, object] = {}
        mask = scheduler.get_mask(name)
        state['mask'] = list(mask.prefixes)
        timer = scheduler.get_timer(name)
        if timer is not None:
          state['timer_expiry'] = (
              scheduler.time_model.serialize_time(timer.expiry)
          )
          state['timer_description'] = timer.description
        entity_states[name] = state
      if entity_states:
        initial_entity_states = entity_states

    # Loop until we find an event that at least one entity reacts to,
    # or run out of events/timers entirely.
    while True:
      # Step 1: Determine the next event to process.
      queue = scheduler.peek_event_queue()
      next_event = queue[0] if queue else None
      next_timer = scheduler.get_next_timer()

      event: interrupt_scheduling.Event | None = None

      if next_timer is not None and (
          next_event is None or next_timer.expiry <= next_event.timestamp
      ):
        # Timer fires first (or ties with an event — timer wins ties).
        tag = (
            f'{interrupt_scheduling.TIMER_EXPIRED_TAG_PREFIX}'
            f'{next_timer.entity_name}'
        )
        event = interrupt_scheduling.Event(
            timestamp=next_timer.expiry,
            tag=tag,
            source=next_timer.entity_name,
            description=(
                f'Timer expired for {next_timer.entity_name}:'
                f' {next_timer.description}'
            ),
        )
      elif next_event is not None:
        event = scheduler.pop_next_event()

      if event is None:
        # No events and no timers.
        scheduler.set_current_event(None)
        scheduler.set_current_polled_entities([])
        self._logging_channel({
            'Key': self._pre_act_label,
            'Summary': 'No events or timers pending.',
            'Value': '',
        })
        return ''

      # Step 2: Advance time.
      scheduler.advance_time(event.timestamp)

      # Step 3: Determine who to poll.
      entities_to_poll = scheduler.get_entities_to_poll(event)

      if not entities_to_poll:
        # No entity cares about this event — skip it and try the next.
        scheduler.increment_tick()
        continue

      # Step 4: Poll each entity (clear mask and timer).
      for name in entities_to_poll:
        scheduler.poll_entity(name)

      # Step 5: For non-polled entities, add event to pending.
      for name in scheduler.player_names:
        if name not in entities_to_poll:
          scheduler.add_pending(name, event)

      # Store current event and polled entities for other components.
      scheduler.set_current_event(event)
      scheduler.set_current_polled_entities(entities_to_poll)

      # Step 6: Increment tick.
      scheduler.increment_tick()

      result = ', '.join(entities_to_poll)
      log_data = {
          'Key': self._pre_act_label,
          'Summary': (
              f'Time: {scheduler.format_time()} | Event: {event.tag} |'
              f' Polled: {result}'
          ),
          'Value': result,
          'Event': {
              'timestamp': scheduler.time_model.serialize_time(event.timestamp),
              'tag': event.tag,
              'source': event.source,
              'description': event.description,
          },
      }
      if initial_events_snapshot is not None:
        log_data['Initial Events'] = initial_events_snapshot
      if initial_entity_states is not None:
        log_data['Initial Entity States'] = initial_entity_states
      self._logging_channel(log_data)
      return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass

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

"""Make-observation component for interrupt-driven game master orchestration.

Generates observations for entities polled by the interrupt scheduler,
including catch-up summaries of missed events and the triggering event
description.
"""

from concordia.components.game_master import interrupt_scheduling
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}?'
    ' What do they now observe?'
    ' Only include information of which they are aware.'
)


class InterruptMakeObservation(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Generates observations for polled entities.

  When an entity is polled, this component delivers:
  1. The current simulated time.
  2. A catch-up summary of missed events (from pending observations).
  3. The triggering event description.
  """

  def __init__(
      self,
      scheduler_component_key: str = (
          interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY
      ),
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      pre_act_label: str = '\nPrompt',
  ):
    super().__init__()
    self._scheduler_key = scheduler_component_key
    self._call_to_make_observation = call_to_make_observation
    self._pre_act_label = pre_act_label

  def _get_scheduler(self) -> interrupt_scheduling.EntityScheduler:
    return self.get_entity().get_component(
        self._scheduler_key, type_=interrupt_scheduling.EntityScheduler
    )

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.MAKE_OBSERVATION:
      return ''

    scheduler = self._get_scheduler()
    current_event = scheduler.get_current_event()
    if current_event is None:
      return ''

    # Extract the entity name from the call_to_action.
    entity_name = self._get_entity_name(action_spec.call_to_action)

    # Check if this entity is actually being polled.
    polled = scheduler.get_current_polled_entities()
    if entity_name not in polled:
      return ''

    # Build the observation.
    parts = []

    # Current time.
    parts.append(f'[Current time: {scheduler.format_time()}]')

    # Catch-up summary of missed events.
    pending = scheduler.drain_pending(entity_name)
    if pending:
      parts.append('\n--- Events missed while focused ---')
      for event in pending:
        parts.append(
            f'  [{scheduler.time_model.format_time(event.timestamp)}]'
            f' {event.description}'
        )
      parts.append('--- End of missed events ---\n')

    # Triggering event.
    parts.append(current_event.description)

    result = '\n'.join(parts)

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': f'Observation for {entity_name}',
        'Value': result,
        'Active Entity': entity_name,
        'Pending Count': len(pending),
    })
    return result

  def _get_entity_name(self, call_to_action: str) -> str:
    """Extracts entity name from the call_to_action string."""
    prefix, suffix = self._call_to_make_observation.split('{name}')
    if call_to_action.startswith(prefix):
      name = call_to_action.removeprefix(prefix)
      if suffix and name.endswith(suffix):
        name = name.removesuffix(suffix)
      return name
    return call_to_action

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass

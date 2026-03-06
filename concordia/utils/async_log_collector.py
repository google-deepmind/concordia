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

"""Async log collector that defers log processing for the async engine."""

from collections.abc import Mapping
import dataclasses
import threading
from typing import Any

from concordia.components.game_master import switch_act as switch_act_component

DEFAULT_ACT_COMPONENT_KEY = switch_act_component.DEFAULT_ACT_COMPONENT_KEY


@dataclasses.dataclass
class RawLogEvent:
  """A raw, unprocessed log event emitted by an entity thread."""

  step: int
  entity_name: str
  game_master_name: str
  entity_log: Mapping[str, Any]
  game_master_log: Mapping[str, Any]
  action: str


class AsyncLogCollector:
  """Collects raw log events from entity threads and defers finalization.

  Entity threads emit RawLogEvent instances via emit(). The expensive
  finalization (nested dict iteration, filtering empty values) only happens
  when materialize() is called — typically on pause or at simulation end.
  """

  def __init__(self):
    self._buffer: list[RawLogEvent] = []
    self._lock = threading.Lock()

  def emit(self, event: RawLogEvent) -> None:
    with self._lock:
      self._buffer.append(event)

  def materialize(self, log: list[Mapping[str, Any]]) -> None:
    """Finalize all buffered events and append to the raw log list.

    Args:
      log: The raw_log list to append finalized entries to.
    """
    with self._lock:
      events = list(self._buffer)
      self._buffer.clear()

    for event in events:
      game_master_finalized_log = {}
      for segment_key, segment_log in event.game_master_log.items():
        game_master_finalized_log[segment_key] = {}
        if not isinstance(segment_log, dict):
          continue
        for component_key, component_value in segment_log.items():
          if component_value and isinstance(component_value, dict):
            tmp_log_dict = {
                key: value for key, value in component_value.items() if value
            }
            if len(tmp_log_dict) > 1:
              game_master_finalized_log[segment_key][
                  component_key
              ] = tmp_log_dict

      game_master_key = event.game_master_name
      if DEFAULT_ACT_COMPONENT_KEY in game_master_finalized_log.get(
          'resolve', {}
      ):
        event_to_log = game_master_finalized_log['resolve'][
            DEFAULT_ACT_COMPONENT_KEY
        ]['Value']
        game_master_key = f'{game_master_key} --- {event_to_log}'

      entity_key = f'Entity [{event.entity_name}]'
      entry = {
          'Step': event.step,
          entity_key: dict(event.entity_log) if event.entity_log else {},
          game_master_key: game_master_finalized_log,
          'Summary': f'Step {event.step} {game_master_key}',
          'thread': event.entity_name,
      }
      log.append(entry)

  def pending_count(self) -> int:
    with self._lock:
      return len(self._buffer)

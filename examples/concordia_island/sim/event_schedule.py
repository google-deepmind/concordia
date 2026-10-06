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

"""Event schedule data models and scheduler."""

from collections.abc import Sequence
import dataclasses
import datetime
from typing import Any

from absl import logging
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_obs_component
from concordia.typing import entity_component

from examples.concordia_island.sim import temporal_nudge as temporal_nudge_mod


@dataclasses.dataclass
class ScheduledEvent:
  """An event to be delivered at a specific simulation time.

  Attributes:
    text: The body of the event (e.g. news headline and body).
    tags: List of tags for the event (e.g. ["news"], ["life altering event"]).
    deliver_date: ISO date string, e.g. "2026-01-01".
    deliver_time: Time string, e.g. "11:00 PM" or "7:00 AM".
    target_agents: List of agent names to receive this event. If None,
      broadcasts to all agents.
    apply_layoff: If True, also call lay_off_agent on the target agents.
  """

  text: str
  tags: list[str]
  deliver_date: str
  deliver_time: str = "11:00 PM"
  target_agents: list[str] | None = None
  apply_layoff: bool = False

  @property
  def deliver_datetime(self) -> datetime.datetime:
    """Parse deliver_date and deliver_time into a datetime object."""
    # Fixed clock uses 2026 as base year in its start_dt
    # Let's parse the time part.
    dt = datetime.datetime.strptime(self.deliver_time, "%I:%M %p")
    date_part = datetime.date.fromisoformat(self.deliver_date)
    return datetime.datetime(
        date_part.year,
        date_part.month,
        date_part.day,
        dt.hour,
        dt.minute,
        dt.second,
    )


class EventScheduler:
  """Delivers scheduled events as observations at the right sim time."""

  def __init__(self, events: Sequence[ScheduledEvent], sim: Any):
    self._events = sorted(events, key=lambda e: e.deliver_datetime)
    self._delivered: set[int] = set()
    self._sim = sim

    # Cache GM components
    gm = sim.game_masters[0]
    self._gm_memory = gm.get_component(
        "__memory__", type_=memory_component.Memory
    )
    self._make_obs = gm.get_component(
        "__make_observation__", type_=make_obs_component.MakeObservation
    )
    try:
      self._temporal_nudge = gm.get_component(
          "temporal_nudge", type_=temporal_nudge_mod.TemporalNudge
      )
    except (KeyError, AttributeError):
      logging.warning("Could not find TemporalNudge component on GM.")
      self._temporal_nudge = None

  def check_and_deliver(self, current_dt: datetime.datetime) -> list[str]:
    """Check for due events and deliver them.

    Called from step callback.

    Args:
      current_dt: The current simulation datetime.

    Returns:
      List of delivered event descriptions for logging.
    """
    delivered = []
    for i, event in enumerate(self._events):
      if i in self._delivered:
        continue
      if current_dt >= event.deliver_datetime:
        self._deliver_event(event)
        self._delivered.add(i)
        delivered.append(f"[{', '.join(event.tags)}] {event.text[:80]}...")
    return delivered

  def _deliver_event(self, event: ScheduledEvent):
    """Deliver a single event."""
    tag_str = " ".join(f"[{t}]" for t in event.tags)
    observation_text = f"[event] {tag_str} {event.text}"

    if event.target_agents:
      # Targeted: queue observation for specific agents
      for name in event.target_agents:
        self._make_obs.add_to_queue(name, observation_text)
        logging.info("Queued targeted event for %s: %s", name, observation_text)
    else:
      # Broadcast: queue for all agents via "all" key
      self._make_obs.add_to_queue("all", observation_text)
      logging.info("Queued broadcast event: %s", observation_text)

    # Also persist in GM memory for long-term recall
    gm_entity = self._gm_memory.get_entity()
    current_phase = gm_entity.get_phase()
    gm_entity.set_phase(entity_component.Phase.READY)
    try:
      self._gm_memory.add(observation_text)
    finally:
      gm_entity.set_phase(current_phase)

    # Handle layoff side-effects
    if event.apply_layoff and event.target_agents and self._temporal_nudge:
      for name in event.target_agents:
        self._temporal_nudge.lay_off_agent(name)
        logging.info("Applied layoff side-effects for %s", name)

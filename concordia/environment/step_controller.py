# Copyright 2024 DeepMind Technologies Limited.
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

"""Step controller for real-time simulation debugging.

This module provides a thread-safe controller for pausing, resuming, and
stepping through simulations one step at a time.
"""

import dataclasses
import threading
from typing import Any


@dataclasses.dataclass
class StepData:
  """Data captured at the end of each simulation step."""

  step: int
  acting_entity: str
  action: str
  entity_actions: dict[str, str]
  entity_logs: dict[str, dict[str, Any]]
  game_master: str = ''


class StepController:
  """Controls simulation stepping with play/pause/step functionality.

  This class provides thread-safe control over simulation execution, allowing
  external code (like a web server) to pause, resume, or single-step through
  the simulation.

  Example usage:
    controller = StepController()

    # In simulation thread:
    while not done:
      do_simulation_step()
      controller.wait_for_step_permission()

    # In server thread:
    controller.pause()  # Stops simulation after current step
    controller.step()   # Advances by one step then pauses
    controller.play()   # Resumes continuous execution
  """

  def __init__(self, start_paused: bool = True):
    """Initialize the step controller.

    Args:
      start_paused: If True, simulation starts in paused state.
    """
    self._lock = threading.Lock()
    self._condition = threading.Condition(self._lock)
    self._running = not start_paused
    self._step_requested = False
    self._stop_requested = False

  @property
  def is_running(self) -> bool:
    """Returns True if the simulation is in running (not paused) state."""
    with self._lock:
      return self._running

  @property
  def is_paused(self) -> bool:
    """Returns True if the simulation is paused."""
    with self._lock:
      return not self._running

  def play(self) -> None:
    """Resume continuous simulation execution."""
    with self._condition:
      self._running = True
      self._step_requested = False
      self._condition.notify_all()

  def pause(self) -> None:
    """Pause simulation after the current step completes."""
    with self._condition:
      self._running = False
      self._step_requested = False

  def step(self) -> None:
    """Execute a single step then pause.

    This can only be called while paused. It advances the simulation by
    exactly one step and then returns to the paused state.
    """
    with self._condition:
      if self._running:
        return
      self._step_requested = True
      self._condition.notify_all()

  def stop(self) -> None:
    """Request the simulation to stop completely."""
    with self._condition:
      self._stop_requested = True
      self._running = True
      self._condition.notify_all()

  def should_stop(self) -> bool:
    """Check if a stop has been requested."""
    with self._lock:
      return self._stop_requested

  def wait_for_step_permission(self) -> bool:
    """Block until permission is granted to execute the next step.

    This method should be called at the end of each simulation step.
    It will block if the controller is paused, and unblock when:
    - play() is called (returns True, simulation continues)
    - step() is called (returns True, but will pause again after next step)
    - stop() is called (returns False, simulation should terminate)

    Returns:
      True if the simulation should continue, False if it should stop.
    """
    with self._condition:
      while not self._running and not self._step_requested:
        if self._stop_requested:
          return False
        self._condition.wait()

      if self._stop_requested:
        return False

      if self._step_requested:
        self._step_requested = False
        self._running = False

      return True

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

"""Operational simulation state for resource dilemma CPR simulations.

Provides ``ResourceSimulationState`` — a plain Python object that holds
the canonical, simulation-critical state shared across Game Master
components. This includes resource levels, cycle tracking, and
termination flags.

**This module has no dependency on the logging subsystem.**  Game Master
components (e.g. ``ResourceTerminate``) depend on this module for
operational decisions, while the logger reads from it for display.

Usage::

  sim_state = ResourceSimulationState(
      initial_resources=100.0,
      carrying_capacity=120.0,
      num_cycles=6,
  )

  # Pass to GM components for termination checks, cycle tracking, etc.
  # Also pass to the logger so it can read operational state for display.
"""


class ResourceSimulationState:
  """Shared mutable state for operational simulation management.

  This object is the single source of truth for simulation-critical
  values that Game Master components need for operational decisions
  (e.g. whether to terminate).

  Attributes:
    resource_level: Current resource stock level.
    carrying_capacity: Maximum resource stock level.
    current_cycle: The current cycle number (1-indexed).
    total_cycles: Total number of cycles in the simulation.
    terminated: Whether the simulation has been signalled to stop.
    cycle_harvest_total: Total harvest accumulated in the current cycle.
    election_winner: Name of the leader who won the most recent election, or
      empty string if no election has been held this cycle.
    active_policy: The winning leader's policy text for the current cycle, or
      empty string if no election has been held this cycle.
    discussion_completed: Whether the discussion phase has run at least once in
      the current cycle.  Used by ``ResourceTerminate`` to defer termination
      until agents have had a chance to discuss.
  """

  def __init__(
      self,
      initial_resources: float = 100.0,
      carrying_capacity: float = 120.0,
      num_cycles: int = 6,
  ):
    self.resource_level: float = initial_resources
    self.carrying_capacity: float = carrying_capacity
    self.current_cycle: int = 1
    self.total_cycles: int = num_cycles
    self.terminated: bool = False
    self.cycle_harvest_total: float = 0.0
    self.election_winner: str = ''
    self.active_policy: str = ''
    self.discussion_completed: bool = False

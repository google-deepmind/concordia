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

"""Job loss scenario: targeted layoffs at 7:00 AM on the first Tuesday."""

from collections.abc import Sequence
from examples.concordia_island.sim import agents as agents_lib
from examples.concordia_island.sim.event_schedule import ScheduledEvent


def get_events(
    laid_off_agents: Sequence[str],
    agent_configs: Sequence[agents_lib.AgentConfig],
) -> list[ScheduledEvent]:
  """Get the event schedule with only targeted layoff events on first Tuesday."""
  events = []

  # --- Layoff Events (Targeted) ---
  # Tuesday Jan 6th at 7:00 AM (Simulation starts Thursday Jan 1st)
  # Day 1: Thu, Day 2: Fri, Day 3: Sat, Day 4: Sun, Day 5: Mon, Day 6: Tue

  for agent_name in laid_off_agents:
    agent_config = next(
        (cfg for cfg in agent_configs if cfg.name == agent_name), None
    )
    work_place = (
        agent_config.work_place
        if agent_config and agent_config.work_place
        else "their workplace"
    )

    events.append(
        ScheduledEvent(
            text=(
                f"{agent_name} has just been informed via a video call from HR"
                " that they have been laid off. They were told that their job"
                " has been replaced by an AI agent. They are instructed not to"
                f" attend {work_place} anymore."
            ),
            tags=["life altering event"],
            deliver_date="2026-01-06",
            deliver_time="7:00 AM",
            target_agents=[agent_name],
            apply_layoff=True,
        )
    )

  return events

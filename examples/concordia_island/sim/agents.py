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

"""Agent configurations for 100-agent island simulation.

Distribution:
- 50 agents: Lower Middle Class (Sunset Apartments)
- 25 agents: Middle Class (Coral Village)
- 15 agents: Upper Middle Class (Palm Heights)
- 7 agents: Upper Class (Ocean View Estates)
- 3 agents: Elite (Paradise Point Mansions)
"""

import dataclasses
import enum


class ScheduleBias(str, enum.Enum):
  """Agent schedule preferences affecting behavior."""

  EARLY_BIRD = "early_bird"  # Active 5am-9pm
  NIGHT_OWL = "night_owl"  # Active 10am-2am
  NINE_TO_FIVE = "nine_to_five"  # Active 8am-10pm
  FLEXIBLE = "flexible"  # No strong preference


@dataclasses.dataclass
class AgentConfig:
  """Configuration for creating an island agent."""

  name: str
  home_place: str
  work_place: str | None = None
  personality: str = ""
  backstory: str = ""
  age: int = 35
  gender: str = ""
  sexual_orientation: str = ""
  ethnicity: str = ""
  political_orientation: str = ""
  hobbies: list[str] = dataclasses.field(default_factory=list)
  relationship_status: str = "single"
  neighborhood: str = ""
  schedule_bias: ScheduleBias = ScheduleBias.FLEXIBLE
  initial_relationships: dict[str, float] = dataclasses.field(
      default_factory=dict
  )

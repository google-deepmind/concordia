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

"""An entity prefab whose acting component may come from its params.

The standard `minimal__Entity` takes an `act_component_factory` keyword in
`build` (to install `HumanActComponent`, for instance), but the generic
`Simulation` builds entities with only a model and a memory bank. This
prefab also reads the factory from its params, so one instance config seats a
model, a person or a bot.
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import override

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.entity import minimal
from concordia.typing import entity_component

SEAT_PREFAB = 'crucible_seat__Entity'

ActFactory = Callable[[Sequence[str]], entity_component.ActingComponent]


@dataclasses.dataclass
class SeatEntity(minimal.Entity):
  """`minimal.Entity` whose act component factory may come from params."""

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
      *,
      act_component: entity_component.ActingComponent | None = None,
      act_component_factory: ActFactory | None = None,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    if act_component is None and act_component_factory is None:
      act_component_factory = self.params.get('act_component_factory')
    return super().build(
        model,
        memory_bank,
        act_component=act_component,
        act_component_factory=act_component_factory,
    )

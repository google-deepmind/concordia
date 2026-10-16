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

"""Entity prefabs that a person or a model can play.

The standard `minimal__Entity` (and the contrib async-deliberation entity)
take an `act_component_factory` keyword in `build`, for example to install
`HumanActComponent`, but the generic `Simulation` builds entities with only a
model and a memory bank. These prefabs also read the factory from their
params, where `GameContext.act_params` puts it for a human-played role, so one
instance config serves a model or a person.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, override

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.contrib.prefabs.entity import basic_with_async_deliberation
from concordia.language_model import language_model
from concordia.prefabs.entity import minimal
from concordia.typing import entity_component

PLAYABLE_PREFAB = 'deep_time_playable__Entity'
PLAYABLE_POLITY_PREFAB = 'deep_time_playable_polity__Entity'

ActFactory = Callable[[Sequence[str]], entity_component.ActingComponent]


def _factory(
    params: Mapping[str, Any],
    act_component: entity_component.ActingComponent | None,
    act_component_factory: ActFactory | None,
) -> ActFactory | None:
  """The factory to build with: the caller's, else the one in `params`."""
  if act_component is not None or act_component_factory is not None:
    return act_component_factory
  return params.get('act_component_factory')


@dataclasses.dataclass
class PlayableEntity(minimal.Entity):
  """`minimal.Entity` whose act component may come from `params`."""

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
      *,
      act_component: entity_component.ActingComponent | None = None,
      act_component_factory: ActFactory | None = None,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    return super().build(
        model,
        memory_bank,
        act_component=act_component,
        act_component_factory=_factory(
            self.params, act_component, act_component_factory
        ),
    )


@dataclasses.dataclass
class PlayablePolity(basic_with_async_deliberation.Entity):
  """The contrib async-deliberation entity, its act component from `params`.

  A person playing a people (or the colonists) acts on the same context while
  the people's plan still deliberates in the background.
  """

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
      *,
      act_component: entity_component.ActingComponent | None = None,
      act_component_factory: ActFactory | None = None,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    return super().build(
        model,
        memory_bank,
        act_component=act_component,
        act_component_factory=_factory(
            self.params, act_component, act_component_factory
        ),
    )

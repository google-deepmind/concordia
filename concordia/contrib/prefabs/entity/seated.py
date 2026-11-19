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

"""A minimal entity whose acting component is chosen in its params.

`minimal.Entity.build` accepts `act_component` and `act_component_factory`
keywords, for example to install `HumanActComponent` for a person. The
generic `Simulation`, however, builds every entity with only a model and a
memory bank, so those keywords cannot be reached from a `Config`. This prefab
also reads `act_component_factory` from its params. One instance config can
then seat a language model (no factory), a person or a bot:

```python
from concordia.components.agent import human_act_component
from concordia.contrib.prefabs.entity import seated
from concordia.typing import prefab as prefab_lib

prefab_lib.InstanceConfig(
    prefab='seated__Entity',
    role=prefab_lib.Role.ENTITY,
    params={
        'name': 'Alice',
        'act_component_factory': (
            lambda order: human_act_component.HumanActComponent(
                reader, component_order=order
            )
        ),
    },
)
```

The factory receives the prefab's context component order, so the person
sees the same context, in the same order, that the model would have.
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import override

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.entity import minimal
from concordia.typing import entity_component

ActComponentFactory = Callable[
    [Sequence[str]], entity_component.ActingComponent
]

ACT_COMPONENT_FACTORY_PARAM = 'act_component_factory'


@dataclasses.dataclass
class Entity(minimal.Entity):
  """`minimal.Entity`, whose act component factory may come from params."""

  description: str = (  # pyrefly: ignore[bad-override]
      'A minimal entity whose acting component (a language model, a person'
      ' through HumanActComponent, or a scripted policy) is chosen by the'
      ' `act_component_factory` param.'
  )

  @override
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
      *,
      act_component: entity_component.ActingComponent | None = None,
      act_component_factory: ActComponentFactory | None = None,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the entity.

    Args:
      model: The language model of the default acting component and of any
        language-model components.
      memory_bank: The entity's memory bank.
      act_component: Overrides the acting component (as in `minimal.Entity`).
      act_component_factory: Overrides the factory; otherwise the
        `act_component_factory` param is used, if set.

    Returns:
      The entity.
    """
    if act_component is None and act_component_factory is None:
      act_component_factory = self.params.get(ACT_COMPONENT_FACTORY_PARAM)
    return super().build(
        model,
        memory_bank,
        act_component=act_component,
        act_component_factory=act_component_factory,
    )

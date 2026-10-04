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

"""prefab base class."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Any, ClassVar

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity_component


class Role(enum.StrEnum):
  ENTITY = 'entity'
  GAME_MASTER = 'game_master'
  INITIALIZER = 'initializer'


@dataclasses.dataclass
class Prefab(abc.ABC):
  """Base class for a prefab entity."""

  description: ClassVar[str]
  params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  entities: Sequence[entity_component.EntityWithComponents] | None = None

  parameter_roles: ClassVar[Mapping[str, Role]] = {}
  default_role: ClassVar[Role | None] = None
  supports_extra_components: ClassVar[bool] = False

  def check_extra_components(self) -> None:
    """Reject unsupported additions before constructing components or using a model."""
    params: Mapping[str, Any] = self.params
    extras = params.get('extra_components', {})
    if not isinstance(extras, Mapping):
      raise ValueError('extra_components must be a component mapping')
    if extras and not self.supports_extra_components:
      raise NotImplementedError(
          f'{type(self).__module__}.{type(self).__name__}: '
          'extra_components not implemented yet'
      )

  def validate_extra_components(self, components: Mapping[str, Any]) -> None:
    """Check additions against actual component keys before insertion.

    extra_components_dependencies maps added component keys to the context keys
    they require. Dependencies can refer to built-in or other added components.
    extra_components_require_new_keys prevents replacement for authored additions;
    direct Python configurations retain each prefab’s established replacement
    behavior, including shared objects bound to multiple context keys.
    """
    self.check_extra_components()
    params: Mapping[str, Any] = self.params
    extras = params.get('extra_components', {})
    if not isinstance(extras, Mapping):
      raise ValueError('extra_components must be a component mapping')
    if params.get('extra_components_require_new_keys', False) and set(
        components
    ) & set(extras):
      raise ValueError('extra_components must not replace built-in components')
    for key, component in extras.items():
      if not isinstance(key, str) or not key:
        raise ValueError('extra_components keys must be nonempty strings')
      if not isinstance(component, entity_component.ContextComponent):
        raise ValueError('extra_components values must be context components')
    indices = params.get('extra_components_index', {})
    if not isinstance(indices, Mapping) or (
        indices and set(indices) != set(extras)
    ):
      raise ValueError(
          'extra_components_index must have the same keys as extra_components'
      )
    if any(type(value) is not int for value in indices.values()):
      raise ValueError('extra_components_index values must be integers')
    dependencies = params.get('extra_components_dependencies', {})
    if not isinstance(dependencies, Mapping) or set(dependencies) - set(extras):
      raise ValueError('dependencies must refer to added component keys')
    available = set(components) | set(extras)
    for key, required in dependencies.items():
      if not isinstance(required, (tuple, list)) or not all(
          isinstance(value, str) for value in required
      ):
        raise ValueError('component dependencies must be lists of context keys')
      missing = set(required) - available
      if missing:
        raise ValueError(
            f'Component {key}: missing prefab dependencies: '
            + ', '.join(sorted(missing))
        )

  @abc.abstractmethod
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_component.EntityWithComponents:
    """Build an entity with this prefab's component configuration.

    Implementations call check_extra_components before construction. Those that
    support additions set supports_extra_components and validate against their
    actual component dictionary before inserting the extra components.
    """
    raise NotImplementedError

  def __init_subclass__(cls, **kwargs):
    """Called when a class inherits from Prefab. We use it to perform checks."""
    super().__init_subclass__(**kwargs)
    if not hasattr(cls, 'description'):
      raise TypeError(
          f"Class {cls.__name__} must define the 'description' class attribute."
      )


@dataclasses.dataclass
class InstanceConfig:
  prefab: str
  role: Role
  params: Mapping[str, str]


@dataclasses.dataclass
class Config:
  prefabs: Mapping[str, Prefab]
  instances: Sequence[InstanceConfig]
  default_premise: str = ''
  default_max_steps: int = 100

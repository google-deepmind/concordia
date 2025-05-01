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
from typing import ClassVar

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory
from concordia.language_model import language_model

DEFAULT_ROLE_KEY = '__role__'


@dataclasses.dataclass
class Prefab(abc.ABC):
  """Base class for a prefab entity."""

  description: ClassVar[str]
  params: Mapping[str, str] = dataclasses.field(default_factory=dict)
  entities: (
      Sequence[entity_agent_with_logging.EntityAgentWithLogging] | None
  ) = None

  @abc.abstractmethod
  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds a prefab entity."""
    raise NotImplementedError

  def __init_subclass__(cls, **kwargs):
    """Called when a class inherits from Prefab. We use it to perform checks.
    """
    super().__init_subclass__(**kwargs)
    if not hasattr(cls, 'description'):
      raise TypeError(
          f"Class {cls.__name__} must define the 'description' class attribute."
      )

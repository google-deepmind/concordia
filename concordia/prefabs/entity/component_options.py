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

"""Shared optional context assembly for standard entity prefabs."""

from collections.abc import Mapping
from typing import Any

from concordia.typing import entity_component


def add_extra_components(
    components_of_agent: dict[str, entity_component.ContextComponent],
    component_order: list[str],
    params: Mapping[str, Any],
) -> None:
  """Apply minimal's existing extra-components and index semantics in place."""
  # Add the extra components to the end of the component order.
  extra_components = params.get('extra_components', {})
  extra_components_index = params.get('extra_components_index', {})

  # Check that extra_components_index is a dict.
  if not isinstance(extra_components_index, dict) or not isinstance(
      extra_components, dict
  ):
    raise ValueError(
        'extra_components_index and extra_components must be dict. Got'
        f' {type(extra_components_index)} and {type(extra_components)}'
    )

  if extra_components:
    if not extra_components_index:
      extra_components_index = {
          component_name: -1 for component_name in extra_components.keys()
      }
    for component_name, index in extra_components_index.items():
      components_of_agent[component_name] = extra_components[component_name]
      component_order.insert(index, component_name)

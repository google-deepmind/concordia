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

"""Turn editable component records into components for entity prefabs.

The editor and CLI store a component's type, owner entity, display name and
parameters as JSON records. ComponentType registers a Python constructor and
its editable defaults. Registry validates these records and calls build() to
create fresh Concordia ContextComponent objects for each entity construction.
The selected prefab receives those objects in its extra_components parameter,
which attaches them to the entity's context and determines their acting order.

Component types declare any context keys they need, such as the memory key.
The prefab checks these dependencies against the components it actually builds.
Unsupported additions fail explicitly. Imported records select registered
component types; they cannot execute Python or import arbitrary module paths.
"""

from collections.abc import Callable, Mapping
import copy
import dataclasses
import re
from typing import Any

from concordia.components.agent import constant
from concordia.components.agent import observation
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import project_config


@dataclasses.dataclass(frozen=True)
class ComponentType:
  """A component constructor, editable defaults and required context keys."""

  factory: Callable[[dict[str, Any]], entity_component.ContextComponent]
  defaults: Mapping[str, Any]
  prototypes: tuple[str, ...]
  description: str
  dependencies: tuple[str, ...] = ()
  all_prefabs: bool = False
  validate: Callable[[dict[str, Any]], None] = lambda params: None


def validate_registration(
    config: prefab_lib.Config,
    prototype_ids: tuple[str, ...],
    types: Mapping[str, ComponentType],
) -> None:
  """Validate record defaults and declared prototype references, without builds."""
  del config
  for key, spec in types.items():
    path = '$.template.component_types.' + key
    for value in spec.defaults.values():
      project_config.Registry._scalar(value, path + '.defaults')
    if set(spec.prototypes) - set(prototype_ids):
      raise project_config.ValidationError(path, 'unknown component prototype')
    if not all(isinstance(key, str) and key for key in spec.dependencies):
      raise project_config.ValidationError(
          path, 'invalid component dependencies'
      )


def standard_types(
    actor_prototypes: tuple[str, ...], text_prototypes: tuple[str, ...]
) -> dict[str, ComponentType]:
  """Two standard component constructors with finite literal configuration."""

  def validate_observations(params):
    if not 1 <= params['history_length'] <= 1000:
      raise ValueError('history_length must be 1–1000')

  return {
      'constant': ComponentType(
          factory=lambda p: constant.Constant(
              state=p['state'], pre_act_label=p['pre_act_label']
          ),
          defaults={'state': '', 'pre_act_label': 'Context'},
          prototypes=text_prototypes,
          all_prefabs=True,
          description=(
              'Literal context included in this entity’s standard acting'
              ' context.'
          ),
      ),
      'recent-observations': ComponentType(
          factory=lambda p: observation.LastNObservations(
              history_length=p['history_length'],
              pre_act_label=p['pre_act_label'],
          ),
          defaults={
              'history_length': 10,
              'pre_act_label': 'Recent observations',
          },
          prototypes=actor_prototypes,
          all_prefabs=True,
          description=(
              'Recent observed events retrieved through the prefab’s standard'
              ' memory component.'
          ),
          dependencies=('__memory__',),
          validate=validate_observations,
      ),
  }


def catalog(types: Mapping[str, ComponentType]) -> list[dict[str, Any]]:
  return [
      dict(
          key=key,
          defaults=copy.deepcopy(dict(spec.defaults)),
          prototypes=list(spec.prototypes),
          description=spec.description,
          dependencies=list(spec.dependencies),
      )
      for key, spec in types.items()
  ]


def validate(
    document: dict[str, Any], types: Mapping[str, ComponentType]
) -> None:
  records = document['components']
  if not isinstance(records, list) or len(records) > 100:
    raise project_config.ValidationError(
        '$.components', 'expected at most 100 components'
    )
  instances = {x['id']: x for x in document['instances']}
  seen = set()
  for index, item in enumerate(records):
    path = f'$.components[{index}]'
    project_config.Registry._keys(
        item, {'id', 'instance', 'type', 'name', 'params'}, path
    )
    identifier = item['id']
    if (
        not isinstance(identifier, str)
        or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', identifier)
        or identifier in seen
    ):
      raise project_config.ValidationError(
          path + '.id', 'expected unique stable component ID'
      )
    seen.add(identifier)
    if not isinstance(item['type'], str) or item['type'] not in types:
      raise project_config.ValidationError(
          path + '.type', 'unknown registered component type'
      )
    spec = types[item['type']]
    if (
        not isinstance(item['instance'], str)
        or item['instance'] not in instances
        or instances[item['instance']]['prototype'] not in spec.prototypes
    ):
      raise project_config.ValidationError(
          path + '.instance',
          'component requires a compatible registered instance',
      )
    project_config.Registry._scalar(item['name'], path + '.name')
    if not isinstance(item['name'], str) or not item['name'].strip():
      raise project_config.ValidationError(
          path + '.name', 'expected nonempty text'
      )
    project_config.Registry._keys(
        item['params'], set(spec.defaults), path + '.params'
    )
    for key, value in item['params'].items():
      project_config.Registry._scalar(value, path + '.params.' + key)
      if type(value) is not type(spec.defaults[key]):
        raise project_config.ValidationError(
            path + '.params.' + key, 'type must match registered component'
        )
    try:
      spec.validate(item['params'])
    except ValueError as error:
      raise project_config.ValidationError(
          path + '.params', str(error)
      ) from error


def build(
    document: dict[str, Any],
    instance_id: str,
    types: Mapping[str, ComponentType],
) -> dict[str, entity_component.ContextComponent]:
  """Construct fresh registered components, never importing names from data."""
  result = {}
  for item in document['components']:
    if item['instance'] != instance_id:
      continue
    component = types[item['type']].factory(copy.deepcopy(item['params']))
    if not isinstance(component, entity_component.ContextComponent):
      raise project_config.ValidationError(
          '$.components', 'registered factory must return a context component'
      )
    result['authored_' + item['id']] = component
  return result

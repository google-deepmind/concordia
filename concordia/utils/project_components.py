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

"""Trusted scalar component recipes for standard prefab extra_components.

Factories are registered by Python hosts, never selected by a module path in
JSON. Hosts must reserve the authored_ component-key namespace and explicitly
list compatible prototypes. Each conversion constructs fresh component objects.
"""

from collections.abc import Callable, Mapping
import copy
import dataclasses
import re
from typing import Any

from concordia.components.agent import constant
from concordia.components.agent import observation
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import project_config


@dataclasses.dataclass(frozen=True)
class ComponentType:
  """A Python-owned recipe; dependency keys are supplied by the host prefab."""

  factory: Callable[[dict[str, Any]], entity_component.ContextComponent]
  defaults: Mapping[str, Any]
  prototypes: tuple[str, ...]
  description: str
  dependencies: tuple[str, ...] = ()
  validate: Callable[[dict[str, Any]], None] = lambda params: None


# These standard prefabs consume extra_components in their acting context and
# provide the memory dependency. New hosts require explicit integration/tests;
# accepting an arbitrary prefab's params alone does not prove consumption.
_HOST_DEPENDENCIES: Mapping[type, frozenset[str]] = {
    minimal.Entity: frozenset({'__memory__'}),
    basic.Entity: frozenset({'__memory__'}),
    dialogic_and_dramaturgic.GameMaster: frozenset({'__memory__'}),
}


def validate_registration(
    config: prefab_lib.Config,
    prototype_ids: tuple[str, ...],
    types: Mapping[str, ComponentType],
) -> None:
  """Reject recipes whose declared prefab host or dependencies are unsupported."""
  prototypes = dict(zip(prototype_ids, config.instances))
  for key, spec in types.items():
    path = '$.template.component_types.' + key
    for value in spec.defaults.values():
      project_config.Registry._scalar(value, path + '.defaults')
    for prototype in spec.prototypes:
      instance = prototypes.get(prototype)
      host = None if instance is None else config.prefabs.get(instance.prefab)
      available = _HOST_DEPENDENCIES.get(type(host))
      if available is None:
        raise project_config.ValidationError(
            path, 'unsupported component prefab host: ' + prototype
        )
      if not set(spec.dependencies) <= available:
        raise project_config.ValidationError(
            path,
            'missing prefab dependencies: '
            + ', '.join(sorted(set(spec.dependencies) - available)),
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

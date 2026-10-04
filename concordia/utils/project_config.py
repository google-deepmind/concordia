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

"""Save, validate and rebuild editable simulation definitions as JSON.

A Python caller registers Template objects containing Config factories and
allowed prefab prototypes, component recipes and editable component state. Registry
creates default documents, validates imported or edited JSON and converts a
valid document back into a Config for standard prefab construction. Editors
use its field metadata to present permitted parameters and references.

Templates can expose fixed instance parameters or allow users to add instances,
or components with stable IDs. The schema_version field identifies the
saved JSON layout for compatibility. Construction uses only registered Python
objects; imported text cannot choose arbitrary imports or constructors.
"""

from collections.abc import Callable, Mapping
import copy
import dataclasses
import json
import re
from typing import Any

from concordia.typing import prefab as prefab_lib
from concordia.utils import prefab_catalog
from concordia.utils import project_components


def _is_integer(value: Any) -> bool:
  """JSON booleans must never satisfy an integer field."""
  return isinstance(value, int) and not isinstance(value, bool)


class ValidationError(ValueError):
  """An invalid project, with a field path suitable for an editor."""

  def __init__(self, path: str, message: str):
    self.path = path
    super().__init__(f'{path}: {message}')


@dataclasses.dataclass(frozen=True)
class Template:
  """Trusted initial configuration and its supported editing contract.

  The factory returns a fresh Config. Editable instance params are scalar;
  explicitly registered fixed parameters remain owned by the Python host.
  Instance IDs correspond to the factory's instance order and never to names.
  References identify params whose document values are target instance IDs;
  they become runtime names only at the Config boundary. A template validator
  can impose prefab-specific constraints (enums, ranges, etc.), raising
  ValidationError with an exact document path. It must not invoke a model.
  """

  # Opt-in only: factory instances define reusable editing contracts.
  # Validators must accept arbitrary IDs/order/count and validate every instance.
  # Component construction remains entirely inside the registered prefabs.
  factory: Callable[[], prefab_lib.Config]
  instance_ids: tuple[str, ...]
  references: Mapping[tuple[str, str], prefab_lib.Role] = dataclasses.field(
      default_factory=dict
  )
  validate: Callable[[dict[str, Any]], None] = lambda document: None
  # Trusted presentation only; never interpreted as constructors or state codecs.
  inspector: Mapping[str, Mapping[str, dict[str, Any]]] = dataclasses.field(
      default_factory=dict
  )
  editable_instances: bool = False
  editable_state: bool = False
  # Explicit prefab name -> existing prototype contract. Creation uses the
  # registered Prefab.params, while saved prototype IDs and presets stay stable.
  prefab_prototypes: Mapping[str, str] = dataclasses.field(default_factory=dict)
  # Trusted constructor values kept out of editable scalar parameters. Their
  # components can expose their editable surface through get_dynamic_state.
  fixed_parameters: Mapping[str, tuple[str, ...]] = dataclasses.field(
      default_factory=dict
  )
  component_types: Mapping[str, 'project_components.ComponentType'] = (
      dataclasses.field(default_factory=dict)
  )


class Registry:
  """Registered simulations with prefab discovery from installed Concordia packages."""

  def __init__(self, templates: Mapping[str, Template]):
    self._templates = dict(templates)
    self._installed, self._unavailable = prefab_catalog.discover()

  def prefab_diagnostics(self) -> list[dict[str, str]]:
    """Missing optional packages, suitable for catalog diagnostics."""
    return [dataclasses.asdict(item) for item in self._unavailable]

  def _installed_records(self, template: Template) -> dict[str, dict[str, Any]]:
    if not template.editable_instances:
      return {}
    return {
        'installed:'
        + key: dict(
            id='installed:' + key,
            prototype='installed:' + key,
            prefab=key,
            role=entry.role.value,
            params={
                name: copy.deepcopy(value)
                for name, value in entry.prefab.params.items()
                if type(value) in (str, int, bool)
            },
        )
        for key, entry in self._installed.items()
    }

  def _component_types(self, template: Template) -> dict:
    prototypes = tuple(template.instance_ids) + tuple(
        self._installed_records(template)
    )
    return {
        key: (
            dataclasses.replace(spec, prototypes=prototypes)
            if spec.all_prefabs
            else spec
        )
        for key, spec in template.component_types.items()
    }

  def template_keys(self) -> list[str]:
    """Names allowed by this registry; does not construct or run any prefab."""
    return sorted(self._templates)

  def _template(self, key: str) -> Template:
    if key not in self._templates:
      raise ValidationError('$.template', 'unknown registered template')
    return self._templates[key]

  def inspector(self, document: Any) -> dict[str, Any]:
    """Return trusted labels/choices and role-safe reference choices."""
    normalized = self.normalize(document)
    template = self._template(normalized['template'])
    result = {
        item['id']: copy.deepcopy(
            dict(template.inspector.get(item.get('prototype', item['id']), {}))
        )
        for item in normalized['instances']
    }
    for (instance_id, field), role in self._references(
        template, normalized
    ).items():
      result.setdefault(instance_id, {}).setdefault(field, {})['choices'] = [
          {'value': item['id'], 'label': item['params']['name']}
          for item in normalized['instances']
          if item['role'] == role.value
      ]
    return result

  def _references(self, template: Template, document: dict) -> dict:
    if not template.editable_instances:
      return dict(template.references)
    result = {
        (item['id'], field): role
        for item in document['instances']
        for (prototype, field), role in template.references.items()
        if item['prototype'] == prototype
    }
    for item in document['instances']:
      entry = self._installed.get(item['prefab'])
      if entry and item['prototype'] == 'installed:' + entry.key:
        result.update({
            (item['id'], field): role
            for field, role in entry.prefab.parameter_roles.items()
            if field in item['params']
        })
    return result

  def catalog(self, document: Any) -> list[dict[str, Any]]:
    """Return owned, trusted prototypes for opt-in structural authoring."""
    normalized = self.normalize(document)
    template = self._template(normalized['template'])
    if not template.editable_instances:
      return []
    defaults = self.default_document(normalized['template'])
    config = template.factory()
    presets = [
        {
            'key': item['prototype'],
            'kind': 'preset',
            'instance': copy.deepcopy(item),
            'description': config.prefabs[item['prefab']].description,
            'inspector': copy.deepcopy(
                dict(template.inspector.get(item['id'], {}))
            ),
            'references': {
                field: role.value
                for (prototype, field), role in template.references.items()
                if prototype == item['id']
            },
        }
        for item in defaults['instances']
    ]
    by_key = {entry['key']: entry for entry in presets}
    prefabs = []
    for name, prototype in template.prefab_prototypes.items():
      path = '$.template.prefab_prototypes'
      if name in by_key:
        raise ValidationError(path, 'prefab name collides with a preset key')
      if prototype not in by_key or name not in config.prefabs:
        raise ValidationError(path, 'unknown prefab or prototype')
      entry = copy.deepcopy(by_key[prototype])
      if entry['instance']['prefab'] != name:
        raise ValidationError(path, 'prototype must use the named prefab')
      params = entry['instance']['params']
      prefab_params = config.prefabs[name].params
      for field, previous in params.items():
        # References keep their explicitly registered document IDs, rather than
        # treating a prefab's example runtime name as a document reference.
        if field in entry['references']:
          continue
        if field not in prefab_params:
          raise ValidationError(path, 'prefab has no default for ' + field)
        value = prefab_params[field]
        self._scalar(value, path + '.' + name + '.' + field)
        if type(value) is not type(previous):
          raise ValidationError(
              path, 'prefab default type differs for ' + field
          )
        params[field] = copy.deepcopy(value)
      entry.update(key=name, kind='prefab')
      prefabs.append(entry)
    installed = []
    for prototype, item in self._installed_records(template).items():
      if (
          item['prefab'] in by_key
          or item['prefab'] in template.prefab_prototypes
      ):
        raise ValidationError(
            '$.template',
            'installed prefab key collides with registered key: '
            + item['prefab'],
        )
      if item['prefab'] in config.prefabs:
        raise ValidationError(
            '$.template', 'installed prefab key collides with configured prefab'
        )
      if prototype in by_key:
        raise ValidationError('$.template', 'reserved installed prototype key')
      definition = self._installed[item['prefab']].prefab
      references = {
          field: role.value
          for field, role in definition.parameter_roles.items()
          if field in item['params']
      }
      for field, role in references.items():
        matches = [x for x in normalized['instances'] if x['role'] == role]
        target = next(
            (
                x
                for x in matches
                if x['params']['name'] == item['params'][field]
            ),
            matches[0] if matches else None,
        )
        if target:
          item['params'][field] = target['id']
      installed.append(
          dict(
              key=item['prefab'],
              kind='prefab',
              instance=copy.deepcopy(item),
              description=definition.description,
              inspector={},
              references=references,
              fixed_parameters=[
                  name
                  for name, value in definition.params.items()
                  if type(value) not in (str, int, bool)
                  and not name.startswith('extra_components')
              ],
              supports_extra_components=definition.supports_extra_components,
          )
      )
    return prefabs + installed + presets

  def component_catalog(self, document: Any) -> list[dict[str, Any]]:
    normalized = self.normalize(document)
    return project_components.catalog(
        self._component_types(self._template(normalized['template']))
    )

  def default_document(self, key: str) -> dict[str, Any]:
    """Export only explicitly supported initial values; reject objects."""
    template = self._template(key)
    config = template.factory()
    roles = {item.role for item in config.instances}
    if not {prefab_lib.Role.ENTITY, prefab_lib.Role.GAME_MASTER} <= roles:
      raise ValidationError(
          '$.instances', 'template requires a player and game master'
      )
    if len(template.instance_ids) != len(config.instances):
      raise ValidationError('$.instances', 'template ID count mismatch')
    if len(set(template.instance_ids)) != len(template.instance_ids):
      raise ValidationError('$.instances', 'duplicate template IDs')
    if template.component_types:
      if not template.editable_instances:
        raise ValidationError(
            '$.template', 'component authoring requires editable instances'
        )
      project_components.validate_registration(
          config, template.instance_ids, template.component_types
      )
    instances = []
    for instance_id, instance in zip(template.instance_ids, config.instances):
      if not isinstance(instance_id, str) or not instance_id:
        raise ValidationError(
            '$.instances', 'template IDs must be nonempty text'
        )
      if instance.prefab not in config.prefabs:
        raise ValidationError('$.instances', 'unregistered prefab in template')
      fixed = template.fixed_parameters.get(instance_id, ())
      if not set(fixed) <= set(instance.params):
        raise ValidationError(
            '$.template.fixed_parameters', 'unknown constructor parameter'
        )
      params = {
          key: value
          for key, value in instance.params.items()
          if key not in fixed
      }
      for field, value in params.items():
        self._scalar(value, f'$.instances[{instance_id}].params.{field}')
      instances.append(
          dict(
              id=instance_id,
              prefab=instance.prefab,
              role=instance.role.value,
              params=params,
          )
      )
    names = {item['params'].get('name'): item['id'] for item in instances}
    for item in instances:
      for field in item['params']:
        if (item['id'], field) in template.references:
          target = item['params'][field]
          if target not in names:
            raise ValidationError(
                f"$.instances[{item['id']}].params.{field}",
                'template has a dangling runtime name reference',
            )
          item['params'][field] = names[target]
    if template.editable_instances:
      for item in instances:
        item['prototype'] = item['id']
    result: dict[str, Any] = dict(
        schema_version=2 if template.editable_instances else 1,
        template=key,
        instances=instances,
        premise=config.default_premise,
        max_steps=config.default_max_steps,
    )
    if template.component_types or template.editable_state:
      result['schema_version'] = 4
      result['components'] = []
      result['dynamic_states'] = {}
    return result

  @staticmethod
  def _scalar(value: Any, path: str) -> None:
    # Editable scalar parameters are finite. Do not stringify objects.
    if isinstance(value, str):
      try:
        value.encode('utf-8')
      except UnicodeError as error:
        raise ValidationError(path, 'expected valid Unicode text') from error
    if _is_integer(value) and abs(value) > 2**53 - 1:
      raise ValidationError(path, 'integer exceeds exact browser JSON range')
    if type(value) not in (str, int, bool):
      raise ValidationError(
          path, 'expected text, integer or boolean; objects unsupported'
      )

  @staticmethod
  def _keys(value: Any, keys: set[str], path: str) -> None:
    if not isinstance(value, dict):
      raise ValidationError(path, 'expected an object')
    missing = keys - value.keys()
    unknown = value.keys() - keys
    if missing:
      raise ValidationError(path, f'missing fields: {sorted(missing)}')
    if unknown:
      raise ValidationError(path, f'unknown fields: {sorted(unknown)}')

  def normalize(self, document: Any) -> dict[str, Any]:
    """Validate atomically and return an owned, canonical document.

    Text is not stripped, coerced or interpolated. Instance order is canonical
    template order for v1; v2 preserves authored order and selects trusted
    prototypes independently of instance IDs.
    """
    if isinstance(document, dict) and document.get('schema_version') == 3:
      raise ValidationError(
          '$.schema_version',
          'File version 3 requires explicit offline conversion to version 4'
          ' before loading; the original file is unchanged.',
      )
    keys = {'schema_version', 'template', 'instances', 'premise', 'max_steps'}
    if isinstance(document, dict) and document.get('schema_version') == 4:
      keys |= {'components', 'dynamic_states'}
    self._keys(document, keys, '$')
    if not _is_integer(document['schema_version']) or document[
        'schema_version'
    ] not in (1, 2, 4):
      raise ValidationError(
          '$.schema_version', 'supported file versions are 1, 2 and 4'
      )
    if not isinstance(document['template'], str):
      raise ValidationError('$.template', 'expected text')
    template = self._template(document['template'])
    baseline = self.default_document(document['template'])
    if document['schema_version'] != baseline['schema_version']:
      raise ValidationError(
          '$.schema_version', 'must match registered template'
      )
    self._scalar(document['premise'], '$.premise')
    if not isinstance(document['premise'], str):
      raise ValidationError('$.premise', 'expected text')
    if (
        not _is_integer(document['max_steps'])
        or not 1 <= document['max_steps'] <= 1000
    ):
      raise ValidationError('$.max_steps', 'expected integer from 1 to 1000')
    if not isinstance(document['instances'], list):
      raise ValidationError('$.instances', 'expected an array')
    if (
        template.editable_instances
        and not 1 <= len(document['instances']) <= 100
    ):
      raise ValidationError(
          '$.instances', 'expected between 1 and 100 instances'
      )
    expected = {item['id']: item for item in baseline['instances']}
    if set(expected) & set(self._installed_records(template)):
      raise ValidationError('$.template', 'reserved installed prototype key')
    expected.update(self._installed_records(template))
    by_id = {}
    names = set()
    for index, item in enumerate(document['instances']):
      path = f'$.instances[{index}]'
      keys = {'id', 'role', 'prefab', 'params'}
      if template.editable_instances:
        keys.add('prototype')
      self._keys(item, keys, path)
      instance_id = item['id']
      if template.editable_instances:
        if not isinstance(instance_id, str) or not re.fullmatch(
            r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', instance_id
        ):
          raise ValidationError(
              path + '.id', 'expected 1–128 letters, digits, _ or -'
          )
        prototype = item['prototype']
        if not isinstance(prototype, str) or prototype not in expected:
          raise ValidationError(
              path + '.prototype', 'unknown registered prototype'
          )
      else:
        prototype = instance_id
        if not isinstance(instance_id, str) or instance_id not in expected:
          raise ValidationError(path + '.id', 'unknown template instance ID')
      if instance_id in by_id:
        raise ValidationError(path + '.id', 'duplicate instance ID')
      for field in ('prefab', 'role'):
        if item[field] != expected[prototype][field]:
          raise ValidationError(
              path + '.' + field, 'must match trusted template'
          )
      defaults = expected[prototype]['params']
      self._keys(item['params'], set(defaults), path + '.params')
      for field, value in item['params'].items():
        self._scalar(value, path + '.params.' + field)
        if type(value) is not type(defaults[field]):
          raise ValidationError(
              path + '.params.' + field, 'type must match template'
          )
      name = item['params'].get('name')
      if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            path + '.params.name', 'expected nonempty runtime name'
        )
      if name in names:
        raise ValidationError(path + '.params.name', 'duplicate runtime name')
      names.add(name)
      by_id[instance_id] = item
    if not template.editable_instances and set(by_id) != set(expected):
      raise ValidationError(
          '$.instances', 'must contain every template instance'
      )
    if template.editable_instances:
      roles = {item['role'] for item in by_id.values()}
      if not {'entity', 'game_master'} <= roles:
        raise ValidationError(
            '$.instances', 'requires a player and game master'
        )
    for (instance_id, field), role in self._references(
        template, document
    ).items():
      path = f'$.instances[{instance_id}].params.{field}'
      if instance_id not in by_id or field not in by_id[instance_id]['params']:
        raise ValidationError(path, 'invalid trusted reference definition')
      target = by_id[instance_id]['params'][field]
      if target not in by_id or by_id[target]['role'] != role.value:
        raise ValidationError(
            path, 'expected target instance ID with role ' + role.value
        )
    result = copy.deepcopy(document)
    if not template.editable_instances:
      result['instances'] = [copy.deepcopy(by_id[key]) for key in expected]
    if baseline['schema_version'] == 4:
      project_components.validate(result, self._component_types(template))
      self._validate_dynamic_states(result)
    template.validate(result)
    return result

  @staticmethod
  def _validate_dynamic_states(document):
    states = document['dynamic_states']
    if not isinstance(states, dict):
      raise ValidationError(
          '$.dynamic_states', 'expected entity IDs mapped to component state'
      )
    ids = {item['id'] for item in document['instances']}
    for identifier, components in states.items():
      path = '$.dynamic_states.' + identifier
      if identifier not in ids or not isinstance(components, dict):
        raise ValidationError(
            path, 'expected existing entity ID and component mapping'
        )
      for component, fields in components.items():
        if not isinstance(component, str) or not isinstance(fields, dict):
          raise ValidationError(
              path, 'expected component names mapped to field values'
          )
    try:
      json.dumps(states, allow_nan=False)
    except (TypeError, ValueError) as error:
      raise ValidationError(
          '$.dynamic_states', 'expected finite JSON state'
      ) from error

  @staticmethod
  def apply_dynamic_states(document, simulation):
    """Apply initial field overrides using the Simulation component contract.

    Call on a fresh build, before any execution. Validation/Save prepares a fresh
    preview first, so a rejected configuration cannot alter a running entity.
    """
    names = {
        item['id']: item['params']['name'] for item in document['instances']
    }
    for identifier, components in document.get('dynamic_states', {}).items():
      for component, fields in components.items():
        for key, value in fields.items():
          try:
            simulation.set_component_dynamic_state(
                names[identifier], component, key, copy.deepcopy(value)
            )
          except (ValueError, TypeError, KeyError) as error:
            raise ValidationError(
                '$.dynamic_states.' + identifier + '.' + component + '.' + key,
                str(error),
            ) from error

  def to_config(self, document: Any) -> prefab_lib.Config:
    """Build the same fresh Config for an editor Run or headless execution."""
    normalized = self.normalize(document)
    template = self._template(normalized['template'])
    config = template.factory()
    names = {
        item['id']: item['params']['name'] for item in normalized['instances']
    }
    references = self._references(template, normalized)
    configured_prefabs = set(config.prefabs)
    instances = []
    for item in normalized['instances']:
      prototype = item.get('prototype', item['id'])
      if prototype in self._installed_records(template):
        definition = self._installed[item['prefab']].prefab
        if item['prefab'] in configured_prefabs:
          raise ValidationError(
              '$.template',
              'installed prefab key collides with configured prefab',
          )
        config.prefabs = {
            **config.prefabs,
            item['prefab']: copy.deepcopy(definition),
        }
        original = prefab_lib.InstanceConfig(
            item['prefab'],
            prefab_lib.Role(item['role']),
            copy.deepcopy(definition.params),
        )
      else:
        original = config.instances[template.instance_ids.index(prototype)]
      params = {
          **copy.deepcopy(original.params),
          **copy.deepcopy(item['params']),
      }
      for field in params:
        if (item['id'], field) in references:
          params[field] = names[params[field]]
      if template.component_types:
        extras = project_components.build(
            normalized, item['id'], self._component_types(template)
        )
        if extras:
          params['extra_components'] = extras
          params['extra_components_dependencies'] = {
              'authored_'
              + record['id']: list(
                  template.component_types[record['type']].dependencies
              )
              for record in normalized['components']
              if record['instance'] == item['id']
          }
          definition = copy.deepcopy(config.prefabs[item['prefab']])
          definition.params = params
          definition.check_extra_components()
          # Standard minimal prefab otherwise inserts extras at -1.
          # A large index appends in the authored order without replacing keys.
          params['extra_components_index'] = {key: 100000 for key in extras}
      instances.append(
          prefab_lib.InstanceConfig(
              prefab=item['prefab'],
              role=prefab_lib.Role(item['role']),
              params=params,
          )
      )
    return prefab_lib.Config(
        prefabs=copy.deepcopy(config.prefabs),
        instances=instances,
        default_premise=normalized['premise'],
        default_max_steps=normalized['max_steps'],
    )

  def loads(self, text: str) -> dict[str, Any]:
    """Read ordinary JSON without silently accepting duplicate object keys."""

    def pairs(items):
      result = {}
      for key, value in items:
        if key in result:
          raise ValidationError('$', f'duplicate JSON field: {key}')
        result[key] = value
      return result

    try:
      document = json.loads(text, object_pairs_hook=pairs)
    except RecursionError as error:
      raise ValidationError('$', 'project JSON nesting is too deep') from error
    except json.JSONDecodeError as error:
      raise ValidationError(
          '$', f'invalid JSON at line {error.lineno}'
      ) from error
    return self.normalize(document)

  def dumps(self, document: Any) -> str:
    """Serialize a validated initial project without losing literal text."""
    return (
        json.dumps(self.normalize(document), ensure_ascii=False, indent=2)
        + '\n'
    )

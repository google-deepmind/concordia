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

"""Restricted initial-project documents, not simulation checkpoints.

Python callers register trusted Config factories. Documents cannot import Python
or construct components. Version 1 edits the scalar parameters of a fixed set of
instances; object-bearing templates require an explicit future codec.
"""

from collections.abc import Callable, Mapping
import copy
import dataclasses
import json
from typing import Any

from concordia.typing import prefab as prefab_lib


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

  The factory must return a fresh Config using ordinary scalar instance params.
  Instance IDs correspond to the factory's instance order and never to names.
  References identify params whose document values are target instance IDs;
  they become runtime names only at the Config boundary. A template validator
  can impose prefab-specific constraints (enums, ranges, etc.), raising
  ValidationError with an exact document path. It must not invoke a model.
  """

  factory: Callable[[], prefab_lib.Config]
  instance_ids: tuple[str, ...]
  references: Mapping[tuple[str, str], prefab_lib.Role] = dataclasses.field(
      default_factory=dict
  )
  validate: Callable[[dict[str, Any]], None] = lambda document: None


class Registry:
  """Caller-owned allowlist of trusted templates; no discovery or imports."""

  def __init__(self, templates: Mapping[str, Template]):
    self._templates = dict(templates)

  def _template(self, key: str) -> Template:
    if key not in self._templates:
      raise ValidationError('$.template', 'unknown registered template')
    return self._templates[key]

  def default_document(self, key: str) -> dict[str, Any]:
    """Export only explicitly supported initial values; reject objects."""
    template = self._template(key)
    config = template.factory()
    roles = {item.role for item in config.instances}
    if not {prefab_lib.Role.ENTITY, prefab_lib.Role.GAME_MASTER} <= roles:
      raise ValidationError(
          '$.instances', 'template requires an actor and game master'
      )
    if len(template.instance_ids) != len(config.instances):
      raise ValidationError('$.instances', 'template ID count mismatch')
    if len(set(template.instance_ids)) != len(template.instance_ids):
      raise ValidationError('$.instances', 'duplicate template IDs')
    instances = []
    for instance_id, instance in zip(template.instance_ids, config.instances):
      if not isinstance(instance_id, str) or not instance_id:
        raise ValidationError(
            '$.instances', 'template IDs must be nonempty text'
        )
      if instance.prefab not in config.prefabs:
        raise ValidationError('$.instances', 'unregistered prefab in template')
      params = dict(instance.params)
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
    return dict(
        schema_version=1,
        template=key,
        instances=instances,
        premise=config.default_premise,
        max_steps=config.default_max_steps,
    )

  @staticmethod
  def _scalar(value: Any, path: str) -> None:
    # Version 1 is intentionally finite. Do not stringify unknown values.
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
    template order; stable IDs select the corresponding trusted definition.
    """
    self._keys(
        document,
        {'schema_version', 'template', 'instances', 'premise', 'max_steps'},
        '$',
    )
    if (
        not _is_integer(document['schema_version'])
        or document['schema_version'] != 1
    ):
      raise ValidationError('$.schema_version', 'only version 1 is supported')
    if not isinstance(document['template'], str):
      raise ValidationError('$.template', 'expected text')
    template = self._template(document['template'])
    baseline = self.default_document(document['template'])
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
    expected = {item['id']: item for item in baseline['instances']}
    by_id = {}
    names = set()
    for index, item in enumerate(document['instances']):
      path = f'$.instances[{index}]'
      self._keys(item, {'id', 'role', 'prefab', 'params'}, path)
      instance_id = item['id']
      if not isinstance(instance_id, str) or instance_id not in expected:
        raise ValidationError(path + '.id', 'unknown template instance ID')
      if instance_id in by_id:
        raise ValidationError(path + '.id', 'duplicate instance ID')
      for field in ('prefab', 'role'):
        if item[field] != expected[instance_id][field]:
          raise ValidationError(
              path + '.' + field, 'must match trusted template'
          )
      defaults = expected[instance_id]['params']
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
    if set(by_id) != set(expected):
      raise ValidationError(
          '$.instances', 'must contain every template instance'
      )
    for (instance_id, field), role in template.references.items():
      path = f'$.instances[{instance_id}].params.{field}'
      if instance_id not in by_id or field not in by_id[instance_id]['params']:
        raise ValidationError(path, 'invalid trusted reference definition')
      target = by_id[instance_id]['params'][field]
      if target not in by_id or by_id[target]['role'] != role.value:
        raise ValidationError(
            path, 'expected target instance ID with role ' + role.value
        )
    result = copy.deepcopy(document)
    result['instances'] = [copy.deepcopy(by_id[key]) for key in expected]
    template.validate(result)
    return result

  def to_config(self, document: Any) -> prefab_lib.Config:
    """Build the same fresh Config for an editor Run or headless execution."""
    normalized = self.normalize(document)
    template = self._template(normalized['template'])
    config = template.factory()
    names = {
        item['id']: item['params']['name'] for item in normalized['instances']
    }
    instances = []
    for item in normalized['instances']:
      params = copy.deepcopy(item['params'])
      for field in params:
        if (item['id'], field) in template.references:
          params[field] = names[params[field]]
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

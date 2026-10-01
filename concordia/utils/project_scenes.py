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

"""Validate editor scene records and build standard Concordia scene specs.

A group is a named, reusable set of actor IDs, such as the residents of a house.
Scene types use a group to define possible participants; each scene selects
participants from that set. Groups let an author reuse a cast across scenes
and validate membership in one place. They are authoring records rather than
actors or simulated institutions.

Registry calls validate before to_scenes resolves IDs to runtime names and
builds SceneTypeSpec and SceneSpec objects for standard SceneTracker scheduling.
"""

import re
from typing import Any

from concordia.typing import scene as scene_lib
from concordia.utils import project_config

FIELDS = {'groups', 'scene_types', 'scenes'}


def _error(path, message):
  raise project_config.ValidationError(path, message)


def _text(value, path, *, nonempty=False):
  project_config.Registry._scalar(value, path)
  if not isinstance(value, str) or (nonempty and not value.strip()):
    _error(path, 'expected ' + ('nonempty ' if nonempty else '') + 'text')


def _records(value, fields, path):
  if not isinstance(value, list) or not 1 <= len(value) <= 100:
    _error(path, 'expected 1–100 records')
  result = {}
  for index, item in enumerate(value):
    location = f'{path}[{index}]'
    project_config.Registry._keys(item, fields, location)
    identifier = item['id']
    if not isinstance(identifier, str) or not re.fullmatch(
        r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', identifier
    ):
      _error(
          location + '.id',
          'expected a stable ID (1–128 letters, digits, _ or -)',
      )
    if identifier in result:
      _error(location + '.id', 'duplicate ID')
    _text(item['name'], location + '.name', nonempty=True)
    result[identifier] = item
  return result


def _reference(value, targets, path):
  if not isinstance(value, str) or value not in targets:
    _error(path, 'expected an existing reference with the required role')


def _participants(value, actors, path):
  if not isinstance(value, list) or not 1 <= len(value) <= 100:
    _error(path, 'select 1–100 participants')
  seen = set()
  for target in value:
    _reference(target, actors, path)
    if target in seen:
      _error(path, 'duplicate participant')
    seen.add(target)


def validate(document: dict[str, Any], prototypes: tuple[str, ...]) -> None:
  """Validate every reference before building any standard scene object."""
  actors = {x['id'] for x in document['instances'] if x['role'] == 'entity'}
  masters = {
      x['id']
      for x in document['instances']
      if x['role'] == 'game_master' and x['prototype'] in prototypes
  }
  groups = _records(
      document['groups'], {'id', 'name', 'participants'}, '$.groups'
  )
  types = _records(
      document['scene_types'],
      {'id', 'name', 'game_master', 'group', 'premise'},
      '$.scene_types',
  )
  scenes = _records(
      document['scenes'],
      {'id', 'name', 'scene_type', 'participants', 'num_rounds', 'premise'},
      '$.scenes',
  )
  for key, group in groups.items():
    _participants(
        group['participants'], actors, f'$.groups[{key}].participants'
    )
  for key, kind in types.items():
    path = f'$.scene_types[{key}]'
    _reference(kind['game_master'], masters, path + '.game_master')
    _reference(kind['group'], groups, path + '.group')
    _text(kind['premise'], path + '.premise')
  total_rounds = 0
  for key, scene in scenes.items():
    path = f'$.scenes[{key}]'
    _reference(scene['scene_type'], types, path + '.scene_type')
    _participants(scene['participants'], actors, path + '.participants')
    allowed = groups[types[scene['scene_type']]['group']]['participants']
    if not set(scene['participants']) <= set(allowed):
      _error(
          path + '.participants',
          'participants must belong to the scene type group',
      )
    rounds = scene['num_rounds']
    if type(rounds) is not int or not 1 <= rounds <= 1000:
      _error(path + '.num_rounds', 'expected integer from 1 to 1000')
    total_rounds += rounds
    if scene['premise'] is not None:
      _text(scene['premise'], path + '.premise')
  if total_rounds > 1000:
    _error('$.scenes', 'total scene rounds must not exceed 1000')


def to_scenes(document: dict[str, Any]) -> list[scene_lib.SceneSpec]:
  """Convert a validated document; preserve literal text and participant order."""
  names = {x['id']: x['params']['name'] for x in document['instances']}
  groups = {x['id']: x for x in document['groups']}
  types = {}
  for kind in document['scene_types']:
    participants = [names[x] for x in groups[kind['group']]['participants']]
    types[kind['id']] = scene_lib.SceneTypeSpec(
        name=kind['name'],
        game_master_name=names[kind['game_master']],
        possible_participants=participants,
        default_premise={name: [kind['premise']] for name in participants},
    )
  return [
      scene_lib.SceneSpec(
          scene_type=types[scene['scene_type']],
          participants=[names[x] for x in scene['participants']],
          num_rounds=scene['num_rounds'],
          premise=None
          if scene['premise'] is None
          else {names[x]: [scene['premise']] for x in scene['participants']},
      )
      for scene in document['scenes']
  ]

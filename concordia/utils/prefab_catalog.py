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

"""Discover entity prefab classes from Concordia's installed Python packages.

Only the standard and contrib prefab packages are searched. JSON never supplies
an import path. Qualified catalog keys distinguish modules and classes; an
application can keep its own short aliases and configured prefab instances.
Missing optional packages are reported separately from available definitions.
"""

import dataclasses
import importlib
import inspect
import pkgutil

from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass(frozen=True)
class Entry:
  key: str
  prefab: prefab_lib.Prefab
  role: prefab_lib.Role


@dataclasses.dataclass(frozen=True)
class Unavailable:
  module: str
  dependency: str


def discover() -> tuple[dict[str, Entry], list[Unavailable]]:
  """Return fresh prefab defaults in stable order, with optional import failures.

  A missing external dependency excludes the importing module (or package). Missing Concordia
  imports and other import/constructor errors propagate rather than concealing
  installation or programming errors. Discovery does not build any entities.
  """
  entries: dict[str, Entry] = {}
  unavailable = []
  for namespace in ('concordia.prefabs', 'concordia.contrib.prefabs'):
    for location, role in (
        ('entity', prefab_lib.Role.ENTITY),
        ('game_master', prefab_lib.Role.GAME_MASTER),
    ):
      package_name = namespace + '.' + location
      try:
        package = importlib.import_module(package_name)
      except ModuleNotFoundError as error:
        if not error.name or error.name.startswith('concordia'):
          raise
        unavailable.append(Unavailable(package_name, error.name))
        continue
      pending = sorted(
          pkgutil.iter_modules(package.__path__, package.__name__ + '.'),
          key=lambda item: item.name,
      )
      while pending:
        info = pending.pop(0)
        if info.name.endswith('_test'):
          continue
        try:
          module = importlib.import_module(info.name)
        except ModuleNotFoundError as error:
          if not error.name or error.name.startswith('concordia'):
            raise
          unavailable.append(Unavailable(info.name, error.name))
          continue
        if info.ispkg:
          pending.extend(
              sorted(
                  pkgutil.iter_modules(module.__path__, module.__name__ + '.'),
                  key=lambda item: item.name,
              )
          )
          continue
        classes = [
            value
            for _, value in inspect.getmembers(module, inspect.isclass)
            if value.__module__ == module.__name__
            and issubclass(value, prefab_lib.Prefab)
            and not inspect.isabstract(value)
        ]
        for cls in classes:
          key = (
              info.name.removeprefix('concordia.').replace('prefabs.', '')
              + '.'
              + cls.__name__
          )
          if key in entries:
            raise ValueError('Duplicate installed prefab key: ' + key)
          instance = cls()
          entries[key] = Entry(key, instance, instance.default_role or role)
  return entries, unavailable

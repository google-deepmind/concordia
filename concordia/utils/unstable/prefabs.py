# Copyright 2025 DeepMind Technologies Limited.
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

"""Utility functions for loading and working with prefabs.
"""

import inspect
import types


def get_package_classes(module: types.ModuleType):
  """Load all classes defined in any file within a package."""
  package_name = module.__package__
  prefabs = {}
  submodule_names = [
      value for value in dir(module) if not value.startswith('__')]
  for submodule_name in submodule_names:
    submodule = getattr(module, submodule_name)
    all_var_names = dir(submodule)
    for var_name in all_var_names:
      var = getattr(submodule, var_name)
      if inspect.isclass(var) and var.__module__.startswith(package_name):
        key = f'{submodule_name}__{var_name}'
        prefabs[key] = var()
  return prefabs

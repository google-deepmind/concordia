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

"""Loads Deep Time's YAML data files from `examples/deep_time/data/`."""

import copy
import functools
import os
from typing import Any

import yaml

_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_ROOT, 'data')
if not os.path.isdir(DATA_DIR):
  DATA_DIR = os.path.join(
      os.path.dirname(os.path.dirname(_ROOT)),
      'opensource',
      'examples',
      'deep_time',
      'data',
  )


@functools.cache
def _load_cached(name: str) -> Any:
  with open(os.path.join(DATA_DIR, name), encoding='utf-8') as f:
    return yaml.safe_load(f)


def load(name: str) -> Any:
  """Returns a fresh copy of a parsed YAML file, e.g. `load('topics.yaml')`."""
  return copy.deepcopy(_load_cached(name))


def scenario(name: str) -> dict[str, Any]:
  """Returns one scenario definition from `scenarios.yaml`."""
  scenarios = load('scenarios.yaml')
  if name not in scenarios:
    raise KeyError(f'Unknown scenario {name!r}; known: {sorted(scenarios)}')
  return scenarios[name]

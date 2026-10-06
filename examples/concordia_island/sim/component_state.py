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

"""Typed readers for Concordia `ComponentState` dictionaries.

`entity_component.ComponentState` is a mapping whose values are a broad union
(`Collection | Mapping | float | int | str | None`), because it has to survive a
JSON round trip. Every component's `set_state` therefore has to narrow each
value before assigning it to a concretely-typed attribute, and a type checker
is right to reject the unguarded version.

These helpers exist so that narrowing is written once. The alternative -- open
coding `isinstance` ladders inside each `set_state` -- is what the old
`sim/python/` copy did, and it added roughly eighty lines to a single component
while remaining invisible to anyone reading the other copy.

Every helper is total: it returns the supplied default rather than raising when
a key is missing or holds an unusable type. That is deliberate. `set_state` is
called when restoring a checkpoint, and a checkpoint written by an older build
will legitimately be missing keys added since. Losing one restored field is
recoverable; refusing to restore the run is not.
"""

from collections.abc import Iterable, Mapping
from typing import Any


def as_int(state: Mapping[str, Any], key: str, default: int = 0) -> int:
  """Reads an int, falling back to `default` if absent or uncoercible."""
  raw = state.get(key, default)
  if isinstance(raw, bool):
    return int(raw)
  if isinstance(raw, (int, float)):
    return int(raw)
  if isinstance(raw, str):
    try:
      return int(raw)
    except ValueError:
      return default
  return default


def as_float(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
  """Reads a float, falling back to `default` if absent or uncoercible."""
  raw = state.get(key, default)
  if isinstance(raw, (int, float)) and not isinstance(raw, bool):
    return float(raw)
  if isinstance(raw, str):
    try:
      return float(raw)
    except ValueError:
      return default
  return default


def as_bool(state: Mapping[str, Any], key: str, default: bool = False) -> bool:
  """Reads a bool, falling back to `default` if absent."""
  raw = state.get(key, default)
  if raw is None:
    return default
  return bool(raw)


def as_str(state: Mapping[str, Any], key: str, default: str = '') -> str:
  """Reads a str, falling back to `default` if absent or None."""
  raw = state.get(key, default)
  return default if raw is None else str(raw)


def as_optional_int(state: Mapping[str, Any], key: str) -> int | None:
  """Reads an int, preserving None.

  Distinct from `as_int` because several components treat "never set" as
  meaningfully different from zero -- an hour of 0 is midnight, not "no hour
  recorded".

  Args:
    state: Component state mapping.
    key: State key to read.

  Returns:
    The integer value, or None if the key is missing or uncoercible.
  """
  raw = state.get(key)
  if raw is None:
    return None
  if isinstance(raw, bool):
    return int(raw)
  if isinstance(raw, (int, float)):
    return int(raw)
  if isinstance(raw, str):
    try:
      return int(raw)
    except ValueError:
      return None
  return None


def as_optional_str(state: Mapping[str, Any], key: str) -> str | None:
  """Reads a str, preserving None (as distinct from the empty string)."""
  raw = state.get(key)
  return None if raw is None else str(raw)


def as_str_list(state: Mapping[str, Any], key: str) -> list[str]:
  """Reads a list of strings; returns empty if absent or not a sequence."""
  raw = state.get(key)
  if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, Mapping)):
    return [str(item) for item in raw]
  return []


def as_str_set(state: Mapping[str, Any], key: str) -> set[str]:
  """Reads a set of strings; returns empty if absent or not a sequence."""
  return set(as_str_list(state, key))


def as_dict_list(state: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
  """Reads a list of record dicts, skipping entries that are not mappings."""
  raw = state.get(key)
  if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, Mapping)):
    return []
  out: list[dict[str, Any]] = []
  for item in raw:
    if isinstance(item, Mapping):
      out.append({str(k): v for k, v in item.items()})
  return out


def as_str_int_map(state: Mapping[str, Any], key: str) -> dict[str, int]:
  """Reads a `{str: int}` mapping, skipping uncoercible values."""
  raw = state.get(key)
  if not isinstance(raw, Mapping):
    return {}
  out: dict[str, int] = {}
  for name, value in raw.items():
    if isinstance(value, bool) or isinstance(value, (int, float)):
      out[str(name)] = int(value)
    elif isinstance(value, str):
      try:
        out[str(name)] = int(value)
      except ValueError:
        continue
  return out


def as_str_str_map(state: Mapping[str, Any], key: str) -> dict[str, str]:
  """Reads a `{str: str}` mapping."""
  raw = state.get(key)
  if not isinstance(raw, Mapping):
    return {}
  return {str(name): str(value) for name, value in raw.items()}

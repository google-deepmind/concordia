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

"""Single source of truth for nighttime marketplace GM shard assignment.

The nighttime marketplace can be split across `num_shards` cluster workers. Two
*separate processes* must independently arrive at the same agent -> shard
assignment:

  * The island coordinator GM, whose `TimeBasedNextGM` decides which shard name
    to return for each entity at the day/night boundary.
  * Each marketplace GM worker, which decides which slice of the roster it is
    responsible for.

They cannot share memory, so they must agree by construction. This module is
the only place the mapping is computed, and it imposes a **canonical ordering**
(sorted, de-duplicated) on the roster before slicing. That removes the
assumption -- which silently failed in an earlier run -- that two independently
derived name lists happen to be in the same order.

Everything here is pure and deterministic: given the same *set* of names and
the same `num_shards`, every caller gets the same answer regardless of the
order in which the names were handed to it.
"""

from collections.abc import Iterable, Mapping, Sequence

# Name of the single, unsharded nighttime marketplace GM. This is also the
# prefix used for shard names.
MARKETPLACE_GM_PREFIX = 'marketplace_rules'


def canonical_roster(player_names: Iterable[str]) -> list[str]:
  """Returns the roster in the canonical order used for shard assignment.

  Sorting (rather than trusting the caller's ordering) is what makes the
  coordinator and the workers agree without having to exchange the mapping.

  Args:
    player_names: Any iterable of agent names, in any order, possibly with
      duplicates.

  Returns:
    The de-duplicated names in ascending lexicographic order.
  """
  return sorted(set(player_names))


def shard_gm_name(shard_idx: int, num_shards: int) -> str:
  """Returns the GM name for a shard.

  Args:
    shard_idx: Zero-based shard index.
    num_shards: Total number of marketplace shards.

  Returns:
    `'marketplace_rules'` when unsharded, else `'marketplace_rules N'`. The
    unsharded name is preserved byte-for-byte so that `num_shards == 1`
    behaves exactly as it did before sharding existed.

  Raises:
    ValueError: If `num_shards` is not positive or `shard_idx` is out of range.
  """
  if num_shards < 1:
    raise ValueError(f'num_shards must be >= 1, got {num_shards}')
  if not 0 <= shard_idx < num_shards:
    raise ValueError(
        f'shard_idx {shard_idx} out of range for num_shards {num_shards}'
    )
  if num_shards == 1:
    return MARKETPLACE_GM_PREFIX
  return f'{MARKETPLACE_GM_PREFIX} {shard_idx}'


def parse_shard_index(gm_name: str) -> int:
  """Extracts the shard index from a marketplace GM name.

  Args:
    gm_name: Either `'marketplace_rules'` or `'marketplace_rules N'`.

  Returns:
    The shard index; 0 for the unsharded name.

  Raises:
    ValueError: If `gm_name` is not a marketplace GM name, or carries a
      suffix that is not a non-negative integer.
  """
  parts = gm_name.split()
  if not parts or parts[0] != MARKETPLACE_GM_PREFIX:
    raise ValueError(
        f'{gm_name!r} is not a marketplace GM name (expected prefix '
        f'{MARKETPLACE_GM_PREFIX!r})'
    )
  if len(parts) == 1:
    return 0
  if len(parts) > 2:
    raise ValueError(f'Malformed marketplace GM name {gm_name!r}')
  try:
    shard_idx = int(parts[1])
  except ValueError as e:
    raise ValueError(
        f'Malformed marketplace GM name {gm_name!r}: shard suffix '
        f'{parts[1]!r} is not an integer'
    ) from e
  if shard_idx < 0:
    raise ValueError(f'Negative shard index in {gm_name!r}')
  return shard_idx


def agents_for_shard(
    player_names: Iterable[str],
    num_shards: int,
    shard_idx: int,
) -> list[str]:
  """Returns the slice of the roster owned by one shard.

  Args:
    player_names: The full roster, in any order.
    num_shards: Total number of marketplace shards.
    shard_idx: Zero-based index of the shard whose slice is wanted.

  Returns:
    This shard's agents, in canonical order. The union over all shards is the
    whole roster and the slices are pairwise disjoint.

  Raises:
    ValueError: If `num_shards`/`shard_idx` are inconsistent.
  """
  if num_shards < 1:
    raise ValueError(f'num_shards must be >= 1, got {num_shards}')
  if not 0 <= shard_idx < num_shards:
    raise ValueError(
        f'shard_idx {shard_idx} out of range for num_shards {num_shards}'
    )
  roster = canonical_roster(player_names)
  if num_shards == 1:
    return roster
  return [name for i, name in enumerate(roster) if i % num_shards == shard_idx]


def assign_shard_map(
    player_names: Iterable[str],
    num_shards: int,
) -> Mapping[str, str]:
  """Returns the agent -> marketplace GM name mapping.

  This is the coordinator-side counterpart of `agents_for_shard`, computed
  from the same canonical ordering so the two are guaranteed consistent.

  Args:
    player_names: The full roster, in any order.
    num_shards: Total number of marketplace shards.

  Returns:
    An empty mapping when `num_shards == 1` (there is nothing to route: the
    single GM is addressed by its plain name). Otherwise agent name -> shard
    GM name for every agent in the roster.

  Raises:
    ValueError: If `num_shards` is not positive.
  """
  if num_shards < 1:
    raise ValueError(f'num_shards must be >= 1, got {num_shards}')
  if num_shards == 1:
    return {}
  roster = canonical_roster(player_names)
  return {
      name: shard_gm_name(i % num_shards, num_shards)
      for i, name in enumerate(roster)
  }


def check_partition(
    player_names: Sequence[str],
    num_shards: int,
) -> None:
  """Asserts that the shards partition the roster. Raises if they do not.

  Intended for use in tests and as a startup self-check; it is cheap enough to
  run on every worker launch.

  Args:
    player_names: The full roster.
    num_shards: Total number of marketplace shards.

  Raises:
    ValueError: If the per-shard slices are not a partition of the roster, or
      disagree with `assign_shard_map`.
  """
  roster = canonical_roster(player_names)
  shard_map = assign_shard_map(roster, num_shards)
  seen: set[str] = set()
  for shard_idx in range(num_shards):
    slice_names = agents_for_shard(roster, num_shards, shard_idx)
    overlap = seen & set(slice_names)
    if overlap:
      raise ValueError(
          f'Shard {shard_idx} overlaps earlier shards on {sorted(overlap)}'
      )
    seen |= set(slice_names)
    if num_shards > 1:
      expected = shard_gm_name(shard_idx, num_shards)
      for name in slice_names:
        if shard_map[name] != expected:
          raise ValueError(
              f'Assignment disagreement for {name!r}: slice says '
              f'{expected!r}, map says {shard_map[name]!r}'
          )
  if seen != set(roster):
    raise ValueError(
        'Shards do not cover the roster; missing '
        f'{sorted(set(roster) - seen)}'
    )

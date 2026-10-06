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

"""Tests for marketplace shard assignment."""

import random

from absl.testing import absltest
from absl.testing import parameterized

from examples.concordia_island.sim import marketplace_sharding


def _roster(n: int) -> list[str]:
  return [f'Agent {i:03d}' for i in range(n)]


class ShardGmNameTest(parameterized.TestCase):

  def test_unsharded_name_is_unchanged(self):
    # The N==1 name must stay byte-for-byte what it was before sharding
    # existed, because the Go router special-cases the literal string.
    self.assertEqual(
        marketplace_sharding.shard_gm_name(0, 1), 'marketplace_rules'
    )

  @parameterized.parameters(0, 1, 7, 9)
  def test_sharded_names(self, shard_idx):
    self.assertEqual(
        marketplace_sharding.shard_gm_name(shard_idx, 10),
        f'marketplace_rules {shard_idx}',
    )

  def test_round_trips_through_parse(self):
    for num_shards in (1, 2, 5, 10):
      for shard_idx in range(num_shards):
        name = marketplace_sharding.shard_gm_name(shard_idx, num_shards)
        self.assertEqual(
            marketplace_sharding.parse_shard_index(name), shard_idx
        )

  @parameterized.parameters(-1, 10, 11)
  def test_rejects_out_of_range_index(self, shard_idx):
    with self.assertRaises(ValueError):
      marketplace_sharding.shard_gm_name(shard_idx, 10)

  @parameterized.parameters(
      'island rules',
      'conversation_rules 0',
      'marketplace_rules x',
      'marketplace_rules 1 2',
  )
  def test_parse_rejects_bad_names(self, gm_name):
    with self.assertRaises(ValueError):
      marketplace_sharding.parse_shard_index(gm_name)


class AssignmentTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 7, 10)
  def test_shards_partition_the_roster(self, num_shards):
    names = _roster(100)
    marketplace_sharding.check_partition(names, num_shards)

  def test_unsharded_shard_owns_everyone(self):
    names = _roster(100)
    self.assertCountEqual(
        marketplace_sharding.agents_for_shard(names, 1, 0), names
    )

  def test_assignment_is_independent_of_input_order(self):
    """The regression test for a sharding failure in an earlier run.

    The coordinator and the workers derive the roster from different sources
    and cannot be assumed to produce it in the same order. Assignment must
    therefore depend only on the *set* of names.
    """
    names = _roster(100)
    shuffled = list(names)
    random.Random(0).shuffle(shuffled)
    self.assertNotEqual(names, shuffled)

    self.assertEqual(
        marketplace_sharding.assign_shard_map(names, 10),
        marketplace_sharding.assign_shard_map(shuffled, 10),
    )
    for shard_idx in range(10):
      self.assertEqual(
          marketplace_sharding.agents_for_shard(names, 10, shard_idx),
          marketplace_sharding.agents_for_shard(shuffled, 10, shard_idx),
      )

  def test_coordinator_map_agrees_with_worker_slice(self):
    """Whatever the coordinator routes to shard k is what shard k claims."""
    names = _roster(100)
    num_shards = 10
    shard_map = marketplace_sharding.assign_shard_map(names, num_shards)
    for shard_idx in range(num_shards):
      claimed = set(
          marketplace_sharding.agents_for_shard(names, num_shards, shard_idx)
      )
      routed = {
          name
          for name, gm in shard_map.items()
          if gm == marketplace_sharding.shard_gm_name(shard_idx, num_shards)
      }
      self.assertEqual(claimed, routed)

  def test_no_shard_is_empty_when_roster_exceeds_shard_count(self):
    names = _roster(100)
    for shard_idx in range(10):
      self.assertNotEmpty(
          marketplace_sharding.agents_for_shard(names, 10, shard_idx)
      )

  def test_unsharded_map_is_empty(self):
    # At N==1 there is nothing to route: the single GM is addressed by name.
    self.assertEmpty(marketplace_sharding.assign_shard_map(_roster(10), 1))

  def test_duplicates_are_collapsed(self):
    names = ['Ann', 'Bob', 'Ann', 'Cal']
    self.assertEqual(
        marketplace_sharding.canonical_roster(names), ['Ann', 'Bob', 'Cal']
    )

  @parameterized.parameters(0, -1)
  def test_rejects_non_positive_shard_count(self, num_shards):
    with self.assertRaises(ValueError):
      marketplace_sharding.assign_shard_map(_roster(10), num_shards)


if __name__ == '__main__':
  absltest.main()

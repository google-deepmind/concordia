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

"""Tests for balance sweeps."""

from absl.testing import absltest
from examples.deep_time import sweep


class SweepTest(absltest.TestCase):

  def test_sweep_and_summary(self):
    rows = sweep.sweep('daisyworld', seeds=2, variants=['passive'], ticks=6)
    self.assertLen(rows, 2)
    self.assertEqual({r['seed'] for r in rows}, {0, 1})
    self.assertTrue(all(r['ticks'] == 6 for r in rows))
    (summary,) = sweep.summarize(rows)
    self.assertEqual(summary['variant'], 'passive')
    self.assertEqual(summary['runs'], 2)
    self.assertEqual(summary['win_rate'], 0.0)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the page's lab, and the sweep it plays."""

import time

from absl.testing import absltest
from examples.lifeways import sweep
from examples.lifeways.web import lab as lab_lib
from examples.lifeways.web import session as session_lib


def _until_ended(lab: lab_lib.Lab, timeout: float = 120.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    job = lab.status()
    if job and job['state'] != 'playing':
      return job
    time.sleep(0.05)
  raise AssertionError('The lab did not end in time.')


class SweepTest(absltest.TestCase):

  def test_a_spread_gives_each_measure_over_seeds(self):
    def row(living):
      return {
          'population': living,
          'held': {'distinct': 3, 'diversity': 0.5, 'top': [['c1', 0.9]]},
          'peoples': {'peoples': 1},
      }

    spread = sweep.spread({0: [row(10)] * 2, 1: [row(20)] * 2})

    self.assertEqual(spread['seeds'], [0, 1])
    self.assertEqual(set(spread['measures']), set(sweep.MEASURES))
    self.assertEqual(
        spread['measures']['living'],
        {
            'mean': 15.0,
            'sd': 7.071,
            'least': 10.0,
            'most': 20.0,
            'by_seed': {'0': 10.0, '1': 20.0},
        },
    )
    self.assertEqual(spread['measures']['peoples']['sd'], 0.0)

  def test_a_run_plays_a_worlds_own_settings(self):
    rows = sweep.play(
        'sandbox', None, sweep.Condition(3, False), 2, {'founders': 4}
    )

    self.assertLen(rows, 2)
    self.assertLessEqual(rows[0]['population'], 5)


class LabTest(absltest.TestCase):

  def test_a_comparison_and_a_replication_play_in_the_background(self):
    lab = lab_lib.Lab(workers=0)
    try:
      started = lab.start(
          lab_lib.COMPARE, 'sandbox', {'free': 0.1}, 2, 3, ('richness', 0.5)
      )
      self.assertEqual(
          (started['runs'], started['seeds'], started['state']),
          (4, [0, 1], 'playing'),
      )
      with self.assertRaisesRegex(ValueError, 'playing'):
        lab.start(lab_lib.REPLICATE, 'sandbox', {}, 1, 3)
      compared = _until_ended(lab)
      self.assertEqual((compared['state'], compared['done']), ('done', 4))
      self.assertIn('difference', compared['result']['measures']['living'])

      lab.start(lab_lib.REPLICATE, 'sandbox', {}, 2, 3, first_seed=7)
      replicated = _until_ended(lab)
      self.assertEqual(replicated['result']['seeds'], [7, 8])
      self.assertIn('least', replicated['result']['measures']['living'])
    finally:
      lab.close()

  def test_it_refuses_what_it_cannot_play(self):
    lab = lab_lib.Lab(workers=0)
    for kind, seeds, years, lever, said in (
        ('guess', 1, 3, None, 'compare or replicate'),
        (lab_lib.REPLICATE, 0, 3, None, '1 to 10 seeds'),
        (lab_lib.REPLICATE, 1, 301, None, '1 to 300 years'),
        (lab_lib.COMPARE, 1, 3, None, 'pulls a lever'),
        (lab_lib.COMPARE, 1, 3, ('richness', 9.0), 'from'),
    ):
      with self.assertRaisesRegex(ValueError, said):
        lab.start(kind, 'sandbox', {}, seeds, years, lever)
    self.assertIsNone(lab.status())

  def test_runs_play_in_processes_of_their_own(self):
    lab = lab_lib.Lab(workers=1)
    try:
      lab.start(lab_lib.REPLICATE, 'sandbox', {}, 1, 2)
      job = _until_ended(lab)
    finally:
      lab.close()

    self.assertEqual(job['state'], 'done', job['error'])
    self.assertEqual(job['result']['seeds'], [0])


class SessionLabTest(absltest.TestCase):

  def test_a_replication_is_read_beside_this_world(self):
    session = session_lib.Session(years_per_second=200, lab_workers=0)
    try:
      session.start()
      deadline = time.monotonic() + 60
      while session.state()['year'] < 5 and time.monotonic() < deadline:
        time.sleep(0.05)
      started = session.lab_start(lab_lib.REPLICATE, 2, 4)
      self.assertEqual(started['seeds'], [1, 2])  # The seeds after its own.
      assert session._lab is not None
      _until_ended(session._lab)

      job = session.lab_status()
      assert job is not None
      self.assertEqual(set(job['here']), set(sweep.MEASURES))
    finally:
      session.close()


if __name__ == '__main__':
  absltest.main()

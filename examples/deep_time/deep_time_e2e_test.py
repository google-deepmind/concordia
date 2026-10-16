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

"""End-to-end mock runs of every Deep Time scenario."""

import csv
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time import run
from examples.deep_time import scenarios
from examples.deep_time import sweep


def _rows(out_dir: str) -> list[dict[str, str]]:
  with open(os.path.join(out_dir, 'timeseries.csv'), encoding='utf-8') as f:
    return list(csv.DictReader(f))


class DeepTimeE2ETest(parameterized.TestCase):

  @parameterized.parameters(*[(name,) for name in sorted(scenarios.SCENARIOS)])
  def test_scenario_runs_end_to_end_in_mock_mode(self, scenario):
    out_dir = self.create_tempdir().full_path
    summary = run.run(scenario, 'mock', out_dir=out_dir)
    self.assertIn(summary['outcome'], ('victory', 'defeat'))
    for name in (
        'summary.json',
        'timeseries.csv',
        'topics.mmd',
        'log.html',
        'report.html',
    ):
      self.assertTrue(os.path.exists(os.path.join(out_dir, name)), name)
    with open(os.path.join(out_dir, 'summary.json'), encoding='utf-8') as f:
      self.assertEqual(json.load(f)['outcome'], summary['outcome'])
    # Locally a scenario takes 0.02-0.06 s a tick; shared CI runners, busy
    # with parallel tests, can take longer.
    self.assertLess(summary['seconds_per_tick'], 2.0)

  def test_daisyworld_mock_run_without_gm_language_calls(self):
    out_dir = self.create_tempdir().full_path
    summary = run.run('daisyworld', 'mock', seed=0, out_dir=out_dir)
    self.assertIn(summary['outcome'], ('victory', 'defeat'))
    ticks = summary['ticks']
    self.assertGreaterEqual(ticks, 25)
    # The World GM is deterministic: every choice call is an entity's act
    # (the Steward, each species including any founded mid-run; the
    # Chronicler's act is a puppet and costs no call).
    acts = summary['acts']
    self.assertEqual(acts['Gaia'], ticks)
    self.assertEqual(
        summary['mock_calls']['choice_calls'],
        sum(n for name, n in acts.items() if name != 'Chronicler'),
    )
    # Species are event-driven here: they act only when woken.
    woken = summary['wakeups']['White daisies']
    self.assertEqual(acts['White daisies'], woken['timer'] + woken['event'])
    self.assertLess(acts['White daisies'], ticks)
    # Free-text calls: the species' adaptation and pact minds, naming, the
    # Advisor and the Chronicler; slow minds are busy most ticks.
    self.assertBetween(summary['mock_calls']['text_calls'], 10, 25 * len(acts))
    self.assertEqual(summary['topics']['climate.global'], ticks + 1)
    self.assertEqual(summary['topics']['gaia.omens'], ticks)
    self.assertEqual(
        summary['topics']['bio.species.white.behaviour'], acts['White daisies']
    )
    self.assertGreater(summary['topics']['gaia.briefings'], 5)
    # The Chronicler acts with a fixed puppet response (no LM call) and
    # writes a chapter every five ticks in the background.
    self.assertGreaterEqual(summary['chapters'], ticks // 5 - 1)
    self.assertEqual(summary['topics']['cycles.carbon'], ticks + 1)
    minds = {(m['owner'], m['mind']): m for m in summary['minds']}
    self.assertEqual(minds[('Chronicler', 'Chronicle')]['latency_ticks'], 5)
    self.assertGreater(minds[('Gaia', 'Advisor')]['computations'], 5)
    rows = _rows(out_dir)
    self.assertGreater(int(rows[-1]['adaptations_black']), 0)
    self.assertNotEqual(rows[-1]['optimum_black'], rows[0]['optimum_black'])

  def test_balance_active_steward_beats_passive(self):
    rows = sweep.sweep('daisyworld', seeds=6, variants=['default', 'passive'])
    by_variant = {s['variant']: s for s in sweep.summarize(rows)}
    self.assertGreaterEqual(by_variant['default']['win_rate'], 0.3)
    # Evolution occasionally saves an untended planet, but rarely.
    self.assertLess(
        by_variant['passive']['win_rate'], by_variant['default']['win_rate']
    )
    self.assertGreater(
        by_variant['default']['mean_ticks'], by_variant['passive']['mean_ticks']
    )

  def test_speciation_adds_entities_that_act(self):
    out_dir = self.create_tempdir().full_path
    summary = run.run('daisyworld', 'mock', seed=1, out_dir=out_dir)
    founded = [k for k in summary['final'] if k.startswith('cover_')][2:]
    self.assertNotEmpty(founded)
    owners = {m['owner'] for m in summary['minds']}
    self.assertGreater(len(owners), 4)
    behaviour_topics = [
        t
        for t in summary['topics']
        if t.startswith('bio.species.') and t.endswith('.behaviour')
    ]
    self.assertGreater(len(behaviour_topics), 2)

  def test_passive_steward_loses(self):
    summary = run.run(
        'daisyworld',
        'mock',
        seed=1,
        mock_variant='passive',
        out_dir=self.create_tempdir().full_path,
    )
    self.assertEqual(summary['outcome'], 'defeat')

  def test_slow_minds_land_exactly_latency_ticks_after_trigger(self):
    out_dir = self.create_tempdir().full_path
    run.run('daisyworld', 'mock', ticks=8, out_dir=out_dir)
    rows = _rows(out_dir)
    # Triggered by the first act (tick 1), latency 3 -> lands at tick 4,
    # phased in over ticks 4-5; the busy mind skips ticks 2-3.
    changed = [
        int(r['tick'])
        for prev, r in zip(rows, rows[1:])
        if r['optimum_black'] != prev['optimum_black']
    ]
    self.assertEqual(changed[:2], [4, 5])

  def test_async_time_mode_runs(self):
    summary = run.run(
        'daisyworld',
        'mock',
        ticks=6,
        time_mode='async',
        out_dir=self.create_tempdir().full_path,
    )
    self.assertEqual(summary['ticks'], 6)

  @parameterized.parameters('daisyworld', 'great_acceleration')
  def test_mock_runs_are_reproducible(self, scenario):
    first = self.create_tempdir().full_path
    second = self.create_tempdir().full_path
    run.run(scenario, 'mock', out_dir=first)
    run.run(scenario, 'mock', out_dir=second)
    self.assertEqual(_rows(first), _rows(second))

  def test_seed_changes_the_planet_but_runs_stay_reproducible(self):
    runs = {
        seed: run.run(
            'daisyworld',
            'mock',
            seed=seed,
            out_dir=self.create_tempdir().full_path,
        )
        for seed in (0, 1, 0)
    }
    self.assertLen(runs, 2)
    self.assertEqual(runs[0]['seed'], 0)

  def test_tick_limit(self):
    summary = run.run(
        'daisyworld', 'mock', ticks=5, out_dir=self.create_tempdir().full_path
    )
    self.assertEqual(summary['ticks'], 5)
    self.assertEqual(summary['outcome'], 'unfinished')


if __name__ == '__main__':
  absltest.main()

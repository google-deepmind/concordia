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

"""Tests for fingerprint_run."""

import io
import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from examples.games import fingerprint_run


def _run(
    game='haggling',
    scenario='fruitville',
    model_name='gpt-4o',
    api_type='openai',
    timestamp='2026-01-01T00:00:00+00:00',
    command='run.py --game=haggling',
    focal_scores=None,
    background_scores=None,
):
  return {
      'game': game,
      'scenario': scenario,
      'model_name': model_name,
      'api_type': api_type,
      'timestamp': timestamp,
      'command': command,
      'focal_scores': focal_scores or {'alice': 1.5},
      'background_scores': background_scores or {'bob': -0.5},
  }


class ContentHashTest(absltest.TestCase):

  def test_deterministic_for_same_input(self):
    runs = [_run()]
    self.assertEqual(
        fingerprint_run.build_content_hash(runs),
        fingerprint_run.build_content_hash(runs),
    )

  def test_independent_of_run_list_order(self):
    run_a = _run(scenario='fruitville')
    run_b = _run(scenario='fruitville_multi', game='haggling_multi_item')
    self.assertEqual(
        fingerprint_run.build_content_hash([run_a, run_b]),
        fingerprint_run.build_content_hash([run_b, run_a]),
    )

  def test_independent_of_dict_key_order(self):
    run_a = _run(focal_scores={'alice': 1.5, 'carol': 2.0})
    run_b = _run(focal_scores={'carol': 2.0, 'alice': 1.5})
    self.assertEqual(
        fingerprint_run.build_content_hash([run_a]),
        fingerprint_run.build_content_hash([run_b]),
    )

  def test_differs_when_a_single_agent_score_changes(self):
    baseline = fingerprint_run.build_content_hash(
        [_run(focal_scores={'alice': 1.5})]
    )
    drifted = fingerprint_run.build_content_hash(
        [_run(focal_scores={'alice': 1.50001})]
    )
    self.assertNotEqual(baseline, drifted)

  def test_matching_aggregate_but_different_per_agent_split_differs(self):
    # Two runs with the same *total* score across two agents, but a
    # different per-agent split -- this is exactly the "aggregate Elo
    # matches, but per-scenario outcomes differ" case #271 is about.
    run_a = _run(focal_scores={'alice': 3.0, 'bob': 1.0})
    run_b = _run(focal_scores={'alice': 2.0, 'bob': 2.0})
    self.assertNotEqual(
        fingerprint_run.build_content_hash([run_a]),
        fingerprint_run.build_content_hash([run_b]),
    )

  def test_differs_when_scenario_changes_but_scores_identical(self):
    run_a = _run(scenario='fruitville')
    run_b = _run(scenario='london', game='pub_coordination')
    self.assertNotEqual(
        fingerprint_run.build_content_hash([run_a]),
        fingerprint_run.build_content_hash([run_b]),
    )

  def test_focal_and_background_scores_are_distinguished(self):
    # Same agent name, same score, but as a focal player vs a background
    # player -- these must not collide.
    focal = _run(focal_scores={'alice': 1.0}, background_scores={})
    background = _run(focal_scores={}, background_scores={'alice': 1.0})
    self.assertNotEqual(
        fingerprint_run.build_content_hash([focal]),
        fingerprint_run.build_content_hash([background]),
    )

  def test_empty_runs_list_is_well_defined(self):
    self.assertEqual(
        fingerprint_run.build_content_hash([]),
        fingerprint_run.build_content_hash([]),
    )

  def test_missing_score_dicts_do_not_raise(self):
    minimal_run = {'game': 'haggling', 'scenario': 'fruitville'}
    self.assertEqual(
        fingerprint_run.build_content_hash([minimal_run]),
        fingerprint_run.build_content_hash([]),
    )


class BundleHashTest(absltest.TestCase):

  def test_same_scores_different_provenance_share_content_but_not_bundle(self):
    run_a = _run(model_name='gpt-4o', timestamp='2026-01-01T00:00:00+00:00')
    run_b = _run(
        model_name='gpt-4o-2026-06-01', timestamp='2026-06-01T00:00:00+00:00'
    )

    content_a = fingerprint_run.build_content_hash([run_a])
    content_b = fingerprint_run.build_content_hash([run_b])
    self.assertEqual(content_a, content_b)

    bundle_a = fingerprint_run.build_bundle_hash(content_a, [run_a])
    bundle_b = fingerprint_run.build_bundle_hash(content_b, [run_b])
    self.assertNotEqual(bundle_a, bundle_b)

  def test_independent_of_run_list_order(self):
    run_a = _run(scenario='fruitville')
    run_b = _run(scenario='london', game='pub_coordination')
    content_hash = fingerprint_run.build_content_hash([run_a, run_b])
    self.assertEqual(
        fingerprint_run.build_bundle_hash(content_hash, [run_a, run_b]),
        fingerprint_run.build_bundle_hash(content_hash, [run_b, run_a]),
    )

  def test_deterministic_for_same_input(self):
    runs = [_run()]
    content_hash = fingerprint_run.build_content_hash(runs)
    self.assertEqual(
        fingerprint_run.build_bundle_hash(content_hash, runs),
        fingerprint_run.build_bundle_hash(content_hash, runs),
    )


class BuildFingerprintTest(absltest.TestCase):

  def test_reports_scenarios_and_run_count(self):
    runs = [
        _run(scenario='fruitville'),
        _run(scenario='london', game='pub_coordination'),
    ]
    report = fingerprint_run.build_fingerprint(runs)
    self.assertEqual(report['num_runs'], 2)
    self.assertEqual(report['scenarios'], ['fruitville', 'london'])
    self.assertIn('content_hash', report)
    self.assertIn('bundle_hash', report)


class LoadAndResolvePathsTest(absltest.TestCase):

  def test_load_score_files_round_trips_json(self):
    run = _run()
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'scores.json')
      with open(path, 'w', encoding='utf-8') as f:
        json.dump(run, f)
      loaded = fingerprint_run.load_score_files([path])
    self.assertEqual(loaded, [run])

  def test_resolve_paths_expands_glob(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path_a = os.path.join(
          tmpdir, 'simulation_scores_haggling_fruitville.json'
      )
      path_b = os.path.join(
          tmpdir, 'simulation_scores_pub_coordination_london.json'
      )
      for path in (path_a, path_b):
        with open(path, 'w', encoding='utf-8') as f:
          json.dump(_run(), f)
      resolved = fingerprint_run.resolve_paths(
          [os.path.join(tmpdir, 'simulation_scores_*.json')]
      )
    self.assertCountEqual(resolved, [path_a, path_b])

  def test_resolve_paths_accepts_literal_path(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'scores.json')
      with open(path, 'w', encoding='utf-8') as f:
        json.dump(_run(), f)
      resolved = fingerprint_run.resolve_paths([path])
    self.assertEqual(resolved, [path])

  def test_resolve_paths_warns_on_no_match(self):
    with mock.patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
      resolved = fingerprint_run.resolve_paths(['/nonexistent/path/*.json'])
    self.assertEmpty(resolved)
    self.assertIn('Warning', mock_stderr.getvalue())


class MainTest(absltest.TestCase):

  def test_main_writes_report_and_exits_cleanly(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      scores_path = os.path.join(tmpdir, 'scores.json')
      output_path = os.path.join(tmpdir, 'fingerprint.json')
      with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(_run(), f)

      fingerprint_run.main(['--scores', scores_path, '--output', output_path])

      with open(output_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
      self.assertEqual(report['num_runs'], 1)
      self.assertIn('content_hash', report)
      self.assertIn('bundle_hash', report)

  def test_main_exits_nonzero_when_no_files_found(self):
    with self.assertRaises(SystemExit) as cm:
      fingerprint_run.main(['--scores', '/nonexistent/path/*.json'])
    self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
  absltest.main()

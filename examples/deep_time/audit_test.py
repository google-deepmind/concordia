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

"""Tests for the complexity audit registry."""

from absl.testing import absltest
from examples.deep_time import audit
from examples.deep_time import scenarios
from examples.deep_time.web import server as server_lib
from examples.deep_time.world import data


class AuditTest(absltest.TestCase):

  def test_registry_is_consistent(self):
    self.assertEqual(audit.validate(audit.registry()), [])

  def test_counts(self):
    counts = audit.counts()
    self.assertGreaterEqual(counts['systems'], 4)
    self.assertGreaterEqual(counts['matrix_cells'], 10)
    self.assertGreaterEqual(counts['scenarios'], 1)
    self.assertLen(data.load('topics.yaml')['topics'], counts['topics'])

  def test_validate_reports_split_yaml_values(self):
    reg = {
        'systems': [{'key': 'a', 'name': 'A', 'module': 'x', 'y': None}],
        'cells': [],
        'loops': [],
    }
    self.assertEqual(
        audit.validate(reg), ["systems entry has unexpected keys ['y']"]
    )

  def test_validate_reports_unknown_systems(self):
    reg = audit.registry()
    reg['cells'].append({'from': 'climate', 'to': 'mars', 'by': 'x'})
    reg['loops'].append({'name': 'bad', 'sign': '?', 'path': ['climate']})
    problems = audit.validate(reg)
    self.assertIn("cell refers to unknown system 'mars'", problems)
    self.assertIn("loop 'bad' has bad sign", problems)


class MapRegistryTest(absltest.TestCase):

  def test_map_counts_and_overlays_match_the_page(self):
    counts = audit.counts()
    self.assertEqual(counts['tiles'], 1200)
    page = open(server_lib.PAGE, encoding='utf-8').read()
    start = page.index('const LAYER_NAMES')
    names = page[start : page.index('};', start)]
    for overlay in audit.registry()['map']['overlays']:
      self.assertIn(f"{overlay['key']}:", names, overlay['key'])

  def test_every_scenario_runs_within_the_mock_budget(self):
    per_tick = audit.speed(ticks=2)
    self.assertCountEqual(per_tick, scenarios.SCENARIOS)
    self.assertLess(max(per_tick.values()), 1.5)  # Stage 2 goal: 0.3 s/tick.


if __name__ == '__main__':
  absltest.main()

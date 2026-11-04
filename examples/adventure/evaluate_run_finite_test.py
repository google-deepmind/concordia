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

"""Finite score validation through the public artifact evaluation writer."""

import json
from pathlib import Path
import tempfile
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from concordia.language_model import no_language_model
from concordia.utils import structured_logging
from examples.adventure import evaluate_run


def _response(score: Any, **overrides: Any) -> str:
  return json.dumps({
      'dimensions': {
          name: {
              'score': overrides.get(name, score),
              'evidence_for': 'Synthetic positive evidence',
              'evidence_against': 'Synthetic negative evidence',
          }
          for name in evaluate_run.DIMENSION_WEIGHTS
      }
  })


class _ScriptedGrader(no_language_model.NoLanguageModel):

  def __init__(self, responses):
    super().__init__()
    self._responses = iter(responses)
    self.prompts = []

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    self.prompts.append(prompt)
    return next(self._responses)


class FiniteAutoraterTest(parameterized.TestCase):

  def _artifacts(self) -> Path:
    temporary = tempfile.TemporaryDirectory(prefix='concordia-evaluation-')
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    (root / 'simulation_structured.json').write_text(
        structured_logging.SimulationLog.from_raw_log([]).to_json()
    )
    (root / 'world_ledger.json').write_text(
        json.dumps({
            'player_name': 'Traveler',
            'npcs': [],
            'story': {},
            'telemetry': {'tick': 0},
            'narrative_log': [],
        })
    )
    (root / 'latency.json').write_text('{}')
    return root

  @parameterized.named_parameters(
      ('nan', float('nan')),
      ('positive_infinity', float('inf')),
      ('negative_infinity', float('-inf')),
      ('nan_string', 'NaN'),
      ('positive_infinity_string', 'Infinity'),
      ('negative_infinity_string', '-Infinity'),
      ('overflow_string', '1e1000'),
  )
  def test_nonfinite_responses_do_not_write_a_report(self, score):
    root = self._artifacts()
    model = _ScriptedGrader([_response(score)] * 3)
    with self.assertRaisesRegex(evaluate_run.AutoraterParseError, 'not finite'):
      evaluate_run.evaluate_artifacts_directory(str(root), model=model)
    self.assertLen(model.prompts, 3)
    self.assertFalse((root / 'evaluation_report.json').exists())

  def test_existing_retry_accepts_a_corrected_finite_response(self):
    root = self._artifacts()
    model = _ScriptedGrader([_response('NaN'), _response(42.5)])
    report = evaluate_run.evaluate_artifacts_directory(str(root), model=model)
    persisted = json.loads((root / 'evaluation_report.json').read_text())
    self.assertLen(model.prompts, 2)
    self.assertIn('not finite', model.prompts[1])
    self.assertEqual(report['autorater']['weighted_total'], 42.5)
    self.assertEqual(persisted['autorater']['weighted_total'], 42.5)

  @parameterized.parameters(
      (0, 0.0), (100, 100.0), ('55.5', 55.5), (-10, 0.0), (110, 100.0)
  )
  def test_finite_scores_keep_existing_numeric_and_clamping_semantics(
      self, score, expected
  ):
    root = self._artifacts()
    model = _ScriptedGrader([_response(score)])
    report = evaluate_run.evaluate_artifacts_directory(str(root), model=model)
    self.assertLen(model.prompts, 1)
    self.assertEqual(report['autorater']['weighted_total'], expected)
    self.assertEqual(
        json.loads((root / 'evaluation_report.json').read_text())['autorater'],
        report['autorater'],
    )

  @parameterized.parameters(None, 'NaN')
  def test_unexercised_dimension_remains_excluded(self, excluded_score):
    root = self._artifacts()
    model = _ScriptedGrader([_response(50, player_parliament=excluded_score)])
    report = evaluate_run.evaluate_artifacts_directory(str(root), model=model)
    dimension = report['autorater']['dimensions']['player_parliament']
    self.assertFalse(dimension['applicable'])
    self.assertIsNone(dimension['score'])
    self.assertEqual(report['autorater']['weighted_total'], 50.0)
    self.assertLen(model.prompts, 1)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the HTML report."""

import contextlib
import io

from absl.testing import absltest
from absl.testing import parameterized
from examples.deep_time import model_setup
from examples.deep_time import run
from examples.deep_time import simulation
from examples.deep_time.report import html_report


class NiceAxisTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.01, 0.68, [0.0, 0.2, 0.4, 0.6, 0.8]),
      (6.5, 36.7, [0.0, 10.0, 20.0, 30.0, 40.0]),
      (22.5, 22.5, [22.5, 22.75, 23.0, 23.25, 23.5]),
  )
  def test_ticks(self, low, high, expected):
    ticks = html_report.nice_axis(low, high)
    self.assertSequenceAlmostEqual(ticks, expected)
    self.assertLessEqual(ticks[0], low)
    self.assertGreaterEqual(ticks[-1], high)


class ReportTest(absltest.TestCase):

  def test_report_has_every_section(self):
    models = model_setup.make_models('mock')
    sim, context = simulation.build_simulation(
        'daisyworld', models, max_ticks=12
    )
    with contextlib.redirect_stdout(io.StringIO()):
      sim.play(max_steps=12)
    summary = run.summarize(context, models, wall_seconds=1.0)
    page = html_report.build_report(context, summary)
    self.assertTrue(page.startswith('<!doctype html>'))
    for heading in (
        'Global temperature',
        'Daisies cover',
        'Evolved growth optimum',
        'Cognition meters',
        'Chronicle',
        'Advisor briefings',
        'Event log',
        'Topics',
        'Data table',
    ):
      self.assertIn(heading, page)
    self.assertEqual(page.count('<figure data-series='), 6)
    self.assertIn('What the Steward read vs the true temperature', page)
    self.assertIn('bio.species.white.adaptation', page)
    self.assertNotIn('<script src=', page)

  def test_missing_values_leave_gaps(self):
    self.assertEqual(
        html_report._segments([None, 1.0, 2.0, None, 3.0]),
        [[(1, 1.0), (2, 2.0)], [(4, 3.0)]],
    )
    rows = [{'tick': 0}, {'tick': 1, 'b': 2.0}, {'tick': 2, 'b': 3.0}]
    svg = html_report.line_chart('t', rows, [html_report.Series('b', 'b', 1)])
    self.assertEqual(svg.count('<polyline'), 1)
    self.assertIn('\\u2013', svg)  # The en dash placeholder, JSON-escaped.

  def test_line_chart_escapes_and_labels(self):
    rows = [{'tick': 0, 'a': 1.0, 'b': 2.0}, {'tick': 1, 'a': 2.0, 'b': 1.0}]
    svg = html_report.line_chart(
        'A <b> & B',
        rows,
        [
            html_report.Series('a', 'alpha', 1, 'α'),
            html_report.Series('b', 'beta', 2),
        ],
    )
    self.assertIn('A &lt;b&gt; &amp; B', svg)
    self.assertIn('>α</text>', svg)
    self.assertIn('class="legend"', svg)


if __name__ == '__main__':
  absltest.main()

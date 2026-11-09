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

"""Tests for the world on one page."""

from absl.testing import absltest
from examples.lifeways.web import report


class ReportTest(absltest.TestCase):

  def test_a_world_is_written_on_one_page(self):
    rows = [
        {'year': 1, 'population': 10, 'lean': True, 'held': {'top': []}},
        {
            'year': 2,
            'population': 12,
            'fields': 1,
            'peoples': {'peoples': 1},
            'held': {'top': [['c1', 0.5]], 'distinct': 4},
        },
    ]
    notebook = [
        {
            'id': 1,
            'year': 1,
            'kind': 'hypothesis',
            'text': 'Fields <grow>.',
            'check': {'due': 9},
        },
        {'id': 2, 'year': 2, 'kind': 'event', 'text': 'Ada built a school.'},
        {'id': 3, 'year': 2, 'kind': 'move', 'text': 'You queue a move.'},
    ]
    releases = [{
        'year': 1,
        'kind': 'habit',
        'domain': 'food',
        'text': 'Eat fish.',
        'heard': 3,
        'peak': 2,
        'died': None,
        'holders': [[1, 2], [2, 1]],
    }]
    surveys = [{
        'year': 2,
        'question': 'Why?',
        'asked': ['Ada', 'Bo'],
        'answers': {'Ada': ['c1']},
        'tally': [['c1', 'Eat fish.', 1]],
    }]

    page = report.page(
        'The Sandbox, seed 0, year 2',
        rows,
        notebook,
        releases,
        surveys,
        None,
        '29 October 2026',
        top_text='Eat fish.',
    )

    self.assertStartsWith(page, '<!doctype html>')
    self.assertNotIn('<script', page)
    self.assertIn('12 alive in year 2; 1 peoples', page)
    self.assertIn('"Eat fish.", is held by 50% of the living', page)
    self.assertEqual(page.count('<svg'), 4)  # No tongues, no inequality.
    self.assertIn('fill="var(--lean)"', page)
    self.assertIn(
        'Fields &lt;grow&gt;. (The census checks it in year 9.)', page
    )
    self.assertIn('Year 2: Ada built a school.', page)
    self.assertNotIn('You queue a move.', page)
    self.assertIn('Heard by 3; at most 2 held it; 1 hold it now.', page)
    self.assertIn(
        '1 of 2 answered. Most named: &quot;Eat fish.&quot; (1)', page
    )

  def test_a_chart_needs_a_value(self):
    self.assertEqual(report.chart([{'year': 1}], lambda row: None), '')


if __name__ == '__main__':
  absltest.main()

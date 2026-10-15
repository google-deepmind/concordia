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

"""Tests for the campaign: gating by stars, best stars, the daily seed."""

import datetime
import os
import tempfile

from absl.testing import absltest
from examples.crucible import campaign
from examples.crucible import toys


class CampaignTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.data = toys.load_data('orders')

  def test_chapters_open_with_stars(self):
    rows = campaign.chapters(self.data, {})
    self.assertEqual([r['open'] for r in rows], [True, False, False, False])
    rows = campaign.chapters(self.data, {'one_hand': 2})
    self.assertEqual([r['open'] for r in rows], [True, True, False, False])
    self.assertFalse(campaign.is_open(self.data, {}, 'voyage'))
    self.assertTrue(campaign.is_open(self.data, {}, 'free'))

  def test_progress_keeps_the_best_and_persists(self):
    path = os.path.join(tempfile.mkdtemp(), 'campaign.json')
    progress = campaign.Progress(path)
    self.assertTrue(progress.record('one_hand', 2))
    self.assertFalse(progress.record('one_hand', 1))
    self.assertEqual(campaign.Progress(path).snapshot(), {'one_hand': 2})

  def test_daily_is_the_same_all_day_and_changes_by_day(self):
    day = datetime.date(2026, 10, 16)
    self.assertEqual(
        campaign.daily(self.data, day), campaign.daily(self.data, day)
    )
    self.assertNotEqual(
        campaign.daily(self.data, day)['seed'],
        campaign.daily(self.data, day + datetime.timedelta(days=1))['seed'],
    )


if __name__ == '__main__':
  absltest.main()

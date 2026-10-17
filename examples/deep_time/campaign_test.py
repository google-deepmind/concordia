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

import datetime
import os
from unittest import mock

from absl.testing import absltest
from examples.deep_time import campaign as campaign_lib
from examples.deep_time import scenarios
from examples.deep_time.world import data

DAY = datetime.date(2026, 10, 15)


class CampaignTest(absltest.TestCase):

  def test_every_scenario_is_a_chapter_once_with_stars(self):
    chapters = campaign_lib.chapters()
    self.assertGreaterEqual(len(chapters), 9)
    keys = [c['scenario'] for c in chapters]
    self.assertCountEqual(keys, scenarios.SCENARIOS)
    for key in keys:
      self.assertNotEmpty(data.scenario(key).get('stars'), key)
    for chapter in chapters:  # Words, not numbers cut at a comma.
      self.assertRegex(chapter['when'], r'[a-z]', chapter['scenario'])

  def test_the_daily_challenge_is_one_world_a_day(self):
    today = campaign_lib.daily(DAY)
    self.assertEqual(today, campaign_lib.daily(DAY))
    self.assertIn(today['scenario'], scenarios.SCENARIOS)
    self.assertBetween(today['seed'], 1, 997)
    week = {
        (
            campaign_lib.daily(DAY + datetime.timedelta(days=d))['scenario'],
            campaign_lib.daily(DAY + datetime.timedelta(days=d))['seed'],
        )
        for d in range(7)
    }
    self.assertGreater(len(week), 1)  # It changes from day to day.

  def test_a_star_opens_the_next_chapter_and_bests_are_kept(self):
    path = os.path.join(self.create_tempdir().full_path, 'progress.json')
    progress = campaign_lib.Progress(path)
    opened = [c['open'] for c in progress.campaign(DAY)['chapters']]
    self.assertEqual(opened[:2], [True, False])  # Only the first is open.
    first = campaign_lib.chapters()[0]['scenario']
    progress.record(first, 0, 2, DAY)
    progress.record(first, 0, 1, DAY)  # A worse game keeps the best.
    chapters = progress.campaign(DAY)['chapters']
    self.assertEqual((chapters[0]['stars'], chapters[1]['open']), (2, True))
    self.assertFalse(chapters[2]['open'])
    today = campaign_lib.daily(DAY)
    progress.record(today['scenario'], today['seed'], 3, DAY)
    self.assertIsNone(progress.campaign(DAY)['daily'])  # Not offered now.
    with mock.patch.object(campaign_lib, 'offered', return_value=True):
      self.assertEqual(progress.campaign(DAY)['daily']['stars'], 3)
      tomorrow = progress.campaign(DAY + datetime.timedelta(days=1))
      self.assertEqual(tomorrow['daily']['stars'], 0)  # A new day.

  def test_a_decided_game_knows_its_chapter_and_the_next(self):
    path = os.path.join(self.create_tempdir().full_path, 'progress.json')
    progress = campaign_lib.Progress(path)

    def chapter(scenario, seed, stars, before):
      found = progress.chapter(scenario, seed, stars, before, DAY)
      assert found is not None
      return found

    first, second = [c['scenario'] for c in campaign_lib.chapters()[:2]]
    before = progress.record(first, 5, 2, DAY)
    found = chapter(first, 5, 2, before)
    self.assertEqual((found['number'], found['best'], before), (1, 2, 0))
    self.assertTrue(found['new_best'])
    self.assertEqual(found['next']['scenario'], second)
    self.assertTrue(found['next']['opened'])  # Its first star opened it.
    self.assertEqual(found['total'], 2)
    before = progress.record(first, 5, 1, DAY)  # A worse game.
    again = chapter(first, 5, 1, before)
    self.assertEqual((again['best'], again['new_best']), (2, False))
    self.assertTrue(again['next']['open'])
    self.assertFalse(again['next']['opened'])  # Open already.
    lost = chapter(second, 5, 0, progress.record(second, 5, 0, DAY))
    self.assertFalse(lost['next']['open'])  # No star: the next stays shut.
    last = campaign_lib.chapters()[-1]['scenario']
    self.assertIsNone(chapter(last, 5, 3, 0)['next'])
    self.assertIsNone(progress.chapter('no_such_world', 5, 3, 0, DAY))
    today = campaign_lib.daily(DAY)
    self.assertFalse(chapter(today['scenario'], today['seed'], 1, 0)['daily'])
    with mock.patch.object(campaign_lib, 'offered', return_value=True):
      self.assertTrue(chapter(today['scenario'], today['seed'], 1, 0)['daily'])

  def test_no_file_no_progress(self):
    progress = campaign_lib.Progress(None)
    progress.record('daisyworld', 0, 3, DAY)
    self.assertEqual(progress.campaign(DAY)['chapters'][0]['stars'], 0)


if __name__ == '__main__':
  absltest.main()

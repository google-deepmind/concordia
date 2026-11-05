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

"""Tests for feuds."""

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import feuds as feuds_lib
from concordia.language_model import no_language_model

_HILL = 'Hill people'
_RIVER = 'River people'


class FeudsTest(absltest.TestCase):

  def test_grievance_past_the_threshold_is_a_feud_and_fades(self):
    feuds = feuds_lib.Feuds(decay=0.5, threshold=1.0, floor=0.3)

    feuds.grieve(_HILL, _RIVER, 1.5, step=0, why='a field taken')

    self.assertTrue(feuds.hostile(_HILL, _RIVER))
    self.assertFalse(feuds.hostile(_RIVER, _HILL))  # Grievance is directed.
    feuds.step(1)
    self.assertAlmostEqual(feuds.grievance(_HILL, _RIVER), 0.75)
    self.assertFalse(feuds.hostile(_HILL, _RIVER))
    feuds.step(2)
    self.assertAlmostEqual(feuds.grievance(_HILL, _RIVER), 0.375)
    feuds.step(3)
    self.assertEqual(feuds.feuds(), [])  # 0.1875: forgotten below 0.3.

  def test_a_raid_is_a_contest_and_feeds_the_feud(self):
    feuds = feuds_lib.Feuds(seed=1, injury=0.5)

    certain = feuds.raid(_HILL, _RIVER, attack=3.0, defence=0.0, step=1)
    hopeless = feuds.raid(_HILL, _RIVER, attack=0.0, defence=3.0, step=2)

    self.assertTrue(certain.won)
    self.assertFalse(hopeless.won)
    self.assertEqual(certain.chance, 1.0)
    self.assertAlmostEqual(feuds.grievance(_RIVER, _HILL), 1.0 + 0.5)

  def test_raids_replay_and_strength_shifts_the_odds(self):
    def wins(seed, attack):
      feuds = feuds_lib.Feuds(seed=seed)
      return [
          feuds.raid(_HILL, _RIVER, attack, 1.0, step=s).won for s in range(60)
      ]

    self.assertEqual(wins(2, 1.0), wins(2, 1.0))
    self.assertGreater(sum(wins(2, 3.0)), sum(wins(2, 1.0 / 3.0)))

  def test_peace_clears_grievance_holds_a_while_and_a_raid_ends_it(self):
    feuds = feuds_lib.Feuds(truce=5)
    feuds.grieve(_HILL, _RIVER, 2.0, step=0)
    feuds.grieve(_RIVER, _HILL, 2.0, step=0)

    feuds.make_peace(_RIVER, _HILL, step=1)

    self.assertEqual(feuds.feuds(), [])
    self.assertTrue(feuds.at_peace(_HILL, _RIVER, step=5))
    feuds.grieve(_HILL, _RIVER, 2.0, step=5)  # Under the truce: nothing.
    self.assertEqual(feuds.grievance(_HILL, _RIVER), 0.0)
    feuds.raid(_HILL, _RIVER, 1.0, 1.0, step=5)
    self.assertFalse(feuds.at_peace(_HILL, _RIVER, step=5))
    feuds.make_peace(_RIVER, _HILL, step=6)
    feuds.step(11)
    self.assertEqual(feuds.truces(), [])

  def test_a_group_that_is_gone_is_forgotten(self):
    feuds = feuds_lib.Feuds()
    feuds.grieve(_HILL, _RIVER, 2.0, step=0)
    feuds.make_peace('Lake people', _RIVER, step=0)

    feuds.forget(_RIVER)

    self.assertEqual((feuds.feuds(), feuds.truces()), ([], []))

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'decay'):
      feuds_lib.Feuds(decay=1.0)
    with self.assertRaisesRegex(ValueError, 'threshold'):
      feuds_lib.Feuds(threshold=0.0)


class RoundTripTest(absltest.TestCase):

  def test_the_feuds_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Valley',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              feuds_lib.DEFAULT_FEUDS_COMPONENT_KEY: feuds_lib.Feuds(seed=3)
          },
      )

    def feuds_of(gm):
      return gm.get_component(
          feuds_lib.DEFAULT_FEUDS_COMPONENT_KEY, type_=feuds_lib.Feuds
      )

    gm = game_master()
    feuds_of(gm).grieve(_HILL, _RIVER, 1.5, step=0)
    feuds_of(gm).raid(_HILL, _RIVER, 2.0, 1.0, step=1)
    feuds_of(gm).make_peace('Lake people', _HILL, step=1)
    again = game_master()
    again.set_state(gm.get_state())

    self.assertEqual(feuds_of(again).get_state(), feuds_of(gm).get_state())
    self.assertEqual(
        feuds_of(again).raid(_HILL, _RIVER, 1.0, 1.0, step=2),
        feuds_of(gm).raid(_HILL, _RIVER, 1.0, 1.0, step=2),
    )
    feuds_of(again).set_state({})  # A state from before it kept any.
    self.assertEqual(feuds_of(again).feuds(), [])


if __name__ == '__main__':
  absltest.main()

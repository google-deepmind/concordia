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

"""Tests for social ties."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import social_ties
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class SocialTiesTest(absltest.TestCase):

  def test_a_meeting_brings_two_closer_and_more_meetings_add_less(self):
    ties = social_ties.SocialTies()

    first = ties.strengthen('Ada', 'Bo', 0.5, kind='work')
    second = ties.strengthen('Bo', 'Ada', 0.5, kind='kin')

    self.assertAlmostEqual(first, 0.5)
    self.assertAlmostEqual(second, 0.75)  # Half of the remaining half.
    self.assertAlmostEqual(ties.strength('Ada', 'Bo'), 0.75)
    self.assertEqual(ties.kinds('Bo', 'Ada'), ['kin', 'work'])
    self.assertEqual(ties.strengthen('Ada', 'Ada', 1.0), 0.0)
    self.assertEqual(ties.size(), 1)

  def test_ties_fade_each_step_and_the_faintest_are_forgotten(self):
    ties = social_ties.SocialTies(decay=0.5, floor=0.1)
    ties.strengthen('Ada', 'Bo', 0.8)
    ties.strengthen('Ada', 'Cy', 0.15)

    forgotten = ties.step()

    self.assertAlmostEqual(ties.strength('Ada', 'Bo'), 0.4)
    self.assertEqual(forgotten, [('Ada', 'Cy')])
    self.assertEqual(ties.ties_of('Cy'), [])

  def test_each_keeps_only_its_strongest_ties(self):
    ties = social_ties.SocialTies(max_ties=2)
    ties.strengthen('Ada', 'Bo', 0.9)
    ties.strengthen('Ada', 'Cy', 0.5)
    ties.strengthen('Ada', 'Di', 0.7)

    self.assertEqual([o for o, _ in ties.ties_of('Ada')], ['Bo', 'Di'])
    self.assertEqual(ties.strength('Ada', 'Cy'), 0.0)
    self.assertEqual(ties.ties_of('Cy'), [])

  def test_ties_of_orders_by_strength_then_name_and_filters_by_kind(self):
    ties = social_ties.SocialTies()
    ties.strengthen('Ada', 'Cy', 0.5, kind='work')
    ties.strengthen('Ada', 'Bo', 0.5, kind='kin')
    ties.strengthen('Ada', 'Di', 0.9, kind='work')

    self.assertEqual(
        ties.ties_of('Ada'), [('Di', 0.9), ('Bo', 0.5), ('Cy', 0.5)]
    )
    self.assertEqual(ties.ties_of('Ada', limit=1), [('Di', 0.9)])
    self.assertEqual(
        [o for o, _ in ties.ties_of('Ada', kind='work')], ['Di', 'Cy']
    )

  def test_contacts_are_who_each_is_tied_to(self):
    ties = social_ties.SocialTies()
    ties.strengthen('Ada', 'Bo', 0.6)
    ties.strengthen('Bo', 'Cy', 0.2)

    self.assertEqual(
        ties.contacts(), {'Ada': ['Bo'], 'Bo': ['Ada', 'Cy'], 'Cy': ['Bo']}
    )
    self.assertEqual(
        ties.contacts(['Bo', 'Zed'], min_strength=0.5),
        {'Bo': ['Ada'], 'Zed': []},
    )

  def test_one_who_leaves_takes_no_ties_along(self):
    ties = social_ties.SocialTies()
    ties.strengthen('Ada', 'Bo', 0.6)
    ties.strengthen('Bo', 'Cy', 0.6)

    ties.remove('Bo')

    self.assertEqual(ties.size(), 0)
    self.assertEqual(ties.contacts(), {})

  def test_everyone_in_a_group_is_brought_closer(self):
    ties = social_ties.SocialTies()

    social_ties.strengthen_all(
        ties, {'rite': ['Ada', 'Bo', 'Cy'], 'field': ['Di']}, 0.3, 'rite'
    )

    self.assertEqual(ties.size(), 3)  # Ada-Bo, Ada-Cy, Bo-Cy; Di alone.
    self.assertAlmostEqual(ties.strength('Bo', 'Cy'), 0.3)

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'decay'):
      social_ties.SocialTies(decay=1.0)
    with self.assertRaisesRegex(ValueError, 'max_ties'):
      social_ties.SocialTies(max_ties=0)


class RoundTripTest(absltest.TestCase):

  def test_the_ties_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              social_ties.DEFAULT_SOCIAL_TIES_COMPONENT_KEY: (
                  social_ties.SocialTies()
              )
          },
      )

    def ties_of(gm):
      return gm.get_component(
          social_ties.DEFAULT_SOCIAL_TIES_COMPONENT_KEY,
          type_=social_ties.SocialTies,
      )

    gm = game_master()
    ties_of(gm).strengthen('Ada', 'Bo', 0.6, kind='kin')
    ties_of(gm).strengthen('Bo', 'Cy', 0.3, kind='work')

    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    restored = ties_of(again)
    self.assertEqual(restored.get_state(), ties_of(gm).get_state())
    self.assertEqual(restored.contacts()['Bo'], ['Ada', 'Cy'])
    self.assertEqual(
        restored.pre_act(entity_lib.free_action_spec(call_to_action='?')), ''
    )
    restored.set_state({})  # A state from before it had ties.
    self.assertEqual(restored.size(), 0)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for reputation."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import reputation as reputation_lib
from concordia.language_model import no_language_model


class ReputationTest(absltest.TestCase):

  def test_deeds_seen_and_heard_make_a_standing(self):
    reputation = reputation_lib.Reputation()

    reputation.witness('Ada', 'generous', 1.0, witnesses=['Bo', 'Cy', 'Ada'])
    reputation.hear(
        'Di', 'Ada', 'generous', reputation_lib.hearsay_weight(1.0, 0.8)
    )

    self.assertEqual(reputation.view('Bo', 'Ada', 'generous'), 1.0)
    self.assertEqual(reputation.view('Ada', 'Ada', 'generous'), 0.0)
    self.assertAlmostEqual(reputation.view('Di', 'Ada', 'generous'), 0.4)
    self.assertAlmostEqual(reputation.standing('Ada', 'generous'), 2.4 / 3)
    self.assertEqual(reputation.known_by('Ada', 'generous'), 3)
    self.assertEqual(reputation.standing('Bo', 'generous'), 0.0)

  def test_bad_report_weighs_against_good(self):
    reputation = reputation_lib.Reputation()
    reputation.witness('Ada', 'generous', 1.0, witnesses=['Bo'])

    reputation.witness('Ada', 'generous', -1.5, witnesses=['Bo', 'Cy'])

    self.assertAlmostEqual(reputation.view('Bo', 'Ada', 'generous'), -0.5)
    self.assertAlmostEqual(reputation.standing('Ada', 'generous'), -1.0)
    self.assertEqual(reputation.known_for('Ada'), [('generous', -1.0, 2)])

  def test_views_fade_and_the_faintest_are_forgotten(self):
    reputation = reputation_lib.Reputation(decay=0.5, floor=0.2)
    reputation.witness('Ada', 'generous', 1.0, witnesses=['Bo'])
    reputation.witness('Ada', 'honest', 0.3, witnesses=['Bo'])

    held = reputation.step()

    self.assertEqual(held, 1)
    self.assertAlmostEqual(reputation.view('Bo', 'Ada', 'generous'), 0.5)
    self.assertEqual(reputation.view('Bo', 'Ada', 'honest'), 0.0)
    reputation.step()
    reputation.step()
    self.assertEqual(reputation.known_for('Ada'), [])

  def test_an_observer_keeps_views_of_its_strongest_only(self):
    reputation = reputation_lib.Reputation(max_known=2)
    reputation.witness('Ada', 'generous', 1.0, witnesses=['Zed'])
    reputation.witness('Bo', 'generous', 0.2, witnesses=['Zed'])
    reputation.witness('Cy', 'generous', -0.9, witnesses=['Zed'])

    self.assertEqual(reputation.view('Zed', 'Bo', 'generous'), 0.0)
    self.assertEqual(reputation.view('Zed', 'Cy', 'generous'), -0.9)

  def test_actors_are_ranked_publicly_or_by_one_observer(self):
    reputation = reputation_lib.Reputation()
    reputation.witness('Ada', 'generous', 1.0, witnesses=['Bo', 'Cy'])
    reputation.witness('Di', 'generous', 2.0, witnesses=['Bo'])
    reputation.witness('Ada', 'generous', 2.0, witnesses=['Zed'])

    self.assertEqual(
        reputation.ranked(['Ada', 'Di', 'Eve'], 'generous'),
        ['Di', 'Ada', 'Eve'],
    )
    self.assertEqual(
        reputation.ranked(['Ada', 'Di'], 'generous', observer='Zed'),
        ['Ada', 'Di'],
    )

  def test_one_who_leaves_is_forgotten_both_ways(self):
    reputation = reputation_lib.Reputation()
    reputation.witness('Ada', 'generous', 1.0, witnesses=['Bo'])
    reputation.witness('Bo', 'generous', 1.0, witnesses=['Ada'])

    reputation.forget('Ada')

    self.assertEqual(reputation.known_for('Ada'), [])
    self.assertEqual(reputation.known_for('Bo'), [])
    self.assertEqual(reputation.get_state(), {'views': []})

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'decay'):
      reputation_lib.Reputation(decay=1.0)
    with self.assertRaisesRegex(ValueError, 'max_known'):
      reputation_lib.Reputation(max_known=0)


class RoundTripTest(absltest.TestCase):

  def test_the_views_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              reputation_lib.DEFAULT_REPUTATION_COMPONENT_KEY: (
                  reputation_lib.Reputation()
              )
          },
      )

    def reputation_of(gm):
      return gm.get_component(
          reputation_lib.DEFAULT_REPUTATION_COMPONENT_KEY,
          type_=reputation_lib.Reputation,
      )

    gm = game_master()
    reputation_of(gm).witness('Ada', 'generous', 1.0, witnesses=['Bo', 'Cy'])
    reputation_of(gm).hear('Di', 'Ada', 'honest', -0.4)
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    restored = reputation_of(again)
    self.assertEqual(restored.get_state(), reputation_of(gm).get_state())
    self.assertEqual(
        restored.known_for('Ada'), reputation_of(gm).known_for('Ada')
    )
    restored.set_state({})  # A state from before it kept reputations.
    self.assertEqual(restored.known_for('Ada'), [])


if __name__ == '__main__':
  absltest.main()

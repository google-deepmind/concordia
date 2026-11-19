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

"""Tests for gatherings."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import gatherings
from concordia.language_model import no_language_model


class GatheringsTest(absltest.TestCase):

  def test_a_gathering_falls_every_so_many_steps(self):
    calendar = gatherings.Gatherings()
    calendar.add('midsummer', place='the ford', every=5, phase=2)

    self.assertEqual([s for s in range(15) if calendar.due(s)], [2, 7, 12])
    self.assertEqual(calendar.next_held('midsummer', 8), 12)
    self.assertEqual(calendar.next_held('midsummer', 12), 12)

  def test_those_who_come_are_handed_to_the_game_when_it_is_held(self):
    calendar = gatherings.Gatherings()
    calendar.add('fair', place='the market', every=2, host='Ada')

    self.assertTrue(calendar.attend('Bo', 'fair', step=4))
    self.assertTrue(calendar.attend('Ada', 'fair', step=4))
    self.assertFalse(calendar.attend('Cy', 'fair', step=5))  # Not held.
    held = calendar.close(step=4)

    self.assertEqual(
        held,
        [
            gatherings.Gathered(
                id='fair',
                place='the market',
                step=4,
                attendees=['Ada', 'Bo'],
                host='Ada',
            )
        ],
    )
    record = calendar.gatherings()[0]
    self.assertEqual(
        (record['held'], record['last'], record['most']), (1, 4, 2)
    )
    self.assertEqual(calendar.close(step=6)[0].attendees, [])  # Cleared.

  def test_a_gathering_by_chance_replays(self):
    def held(seed):
      calendar = gatherings.Gatherings(seed=seed)
      calendar.add('storm dance', place='the hill', every=1, chance=0.5)
      return [s for s in range(40) if calendar.due(s)]

    self.assertEqual(held(1), held(1))
    self.assertNotEqual(held(1), held(2))
    self.assertBetween(len(held(1)), 10, 30)

  def test_only_so_many_are_kept_and_leavers_are_forgotten(self):
    calendar = gatherings.Gatherings(max_gatherings=2)
    calendar.add('a', place='x', every=1, step=0)
    calendar.add('b', place='y', every=1, step=1)
    calendar.attend('Ada', 'b', step=2)
    calendar.add('c', place='z', every=1, step=2)

    self.assertEqual([g['id'] for g in calendar.gatherings()], ['b', 'c'])
    calendar.forget('Ada')
    self.assertEqual(calendar.close(step=2)[0].attendees, [])
    calendar.remove('b')
    self.assertEqual(calendar.due(step=3), ['c'])

  def test_settings_are_checked(self):
    calendar = gatherings.Gatherings()
    with self.assertRaisesRegex(ValueError, 'every'):
      calendar.add('a', place='x', every=0)
    with self.assertRaisesRegex(ValueError, 'chance'):
      calendar.add('a', place='x', every=1, chance=2.0)


class RoundTripTest(absltest.TestCase):

  def test_the_calendar_is_saved_and_restored_with_its_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              gatherings.DEFAULT_GATHERINGS_COMPONENT_KEY: (
                  gatherings.Gatherings(seed=4)
              )
          },
      )

    def calendar_of(gm):
      return gm.get_component(
          gatherings.DEFAULT_GATHERINGS_COMPONENT_KEY,
          type_=gatherings.Gatherings,
      )

    gm = game_master()
    calendar_of(gm).add('fair', place='the market', every=3, chance=0.7)
    calendar_of(gm).close(step=3)
    due = calendar_of(gm).due(step=6)
    for gathering_id in due:
      calendar_of(gm).attend('Ada', gathering_id, step=6)
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    self.assertEqual(
        calendar_of(again).get_state(), calendar_of(gm).get_state()
    )
    self.assertEqual(
        calendar_of(again).close(step=6), calendar_of(gm).close(step=6)
    )
    calendar_of(again).set_state({})  # A state from before it kept any.
    self.assertEqual(calendar_of(again).gatherings(), [])


if __name__ == '__main__':
  absltest.main()

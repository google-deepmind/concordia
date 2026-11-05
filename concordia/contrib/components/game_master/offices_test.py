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

"""Tests for offices."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import offices as offices_lib
from concordia.language_model import no_language_model


class OfficesTest(absltest.TestCase):

  def test_an_office_is_vacant_until_filled_and_held_for_its_term(self):
    offices = offices_lib.Offices()
    offices.create('chief', succession='acclaim', term=10)

    self.assertEqual(offices.vacant(), [('chief', 'acclaim')])
    offices.install('chief', 'Ada', step=3, how='acclaimed')

    self.assertEqual(offices.holder('chief'), 'Ada')
    self.assertEqual(offices.holds('Ada'), ['chief'])
    self.assertEqual(offices.vacant(), [])
    self.assertEqual(offices.offices()[0]['ends'], 13)
    self.assertEqual(offices.expire(step=12), [])
    ended = offices.expire(step=13)
    self.assertEqual(
        ended,
        [offices_lib.Tenure('chief', 'Ada', 3, 13, 'acclaimed', 'term')],
    )
    self.assertEqual(offices.holder('chief'), '')

  def test_one_who_leaves_vacates_and_a_successor_ends_the_tenure(self):
    offices = offices_lib.Offices()
    offices.create('chief')
    offices.create('priest')
    offices.install('chief', 'Ada', step=0)
    offices.install('priest', 'Ada', step=1)

    left = offices.forget('Ada', step=5)

    self.assertEqual([t.office for t in left], ['chief', 'priest'])
    self.assertEqual([t.why_ended for t in left], ['left', 'left'])
    offices.install('chief', 'Bo', step=6)
    offices.install('chief', 'Cy', step=9, how='elected')
    self.assertEqual(
        [(t.holder, t.why_ended) for t in offices.history('chief')],
        [('Ada', 'left'), ('Bo', 'succeeded')],
    )
    tenure = offices.tenure('chief')
    assert tenure is not None
    self.assertEqual(tenure.how, 'elected')

  def test_a_holder_may_step_down_and_history_is_bounded(self):
    offices = offices_lib.Offices(max_history=2)
    for step, name in enumerate(['Ada', 'Bo', 'Cy']):
      offices.install('chief', name, step=step)
    tenure = offices.vacate('chief', step=5, why='stepped down')

    assert tenure is not None
    self.assertEqual(tenure.holder, 'Cy')
    self.assertEqual([t.holder for t in offices.history('chief')], ['Bo', 'Cy'])
    self.assertIsNone(offices.vacate('chief', step=6))

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'term'):
      offices_lib.Offices().create('chief', term=-1)
    with self.assertRaisesRegex(ValueError, 'max_history'):
      offices_lib.Offices(max_history=0)


class RoundTripTest(absltest.TestCase):

  def test_the_offices_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              offices_lib.DEFAULT_OFFICES_COMPONENT_KEY: offices_lib.Offices()
          },
      )

    def offices_of(gm):
      return gm.get_component(
          offices_lib.DEFAULT_OFFICES_COMPONENT_KEY, type_=offices_lib.Offices
      )

    gm = game_master()
    offices_of(gm).create('chief', succession='acclaim', term=10)
    offices_of(gm).install('chief', 'Ada', step=0)
    offices_of(gm).install('chief', 'Bo', step=4, how='acclaimed')
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    self.assertEqual(offices_of(again).get_state(), offices_of(gm).get_state())
    self.assertEqual(offices_of(again).expire(14), offices_of(gm).expire(14))
    offices_of(again).set_state({})  # A state from before it kept any.
    self.assertEqual(offices_of(again).offices(), [])


if __name__ == '__main__':
  absltest.main()

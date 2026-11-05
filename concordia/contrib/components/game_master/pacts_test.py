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

"""Tests for pacts."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import pacts as pacts_lib
from concordia.language_model import no_language_model

_AID = 'We help each other in lean years.'


def _offer(pacts: pacts_lib.Pacts, *args, **kwargs) -> str:
  offer = pacts.offer(*args, **kwargs)
  assert offer is not None
  return offer


def _accept(pacts: pacts_lib.Pacts, *args, **kwargs) -> str:
  pact = pacts.accept(*args, **kwargs)
  assert pact is not None
  return pact


class PactsTest(absltest.TestCase):

  def test_an_offer_accepted_the_next_step_is_a_pact_for_its_term(self):
    pacts = pacts_lib.Pacts()
    offer = _offer(pacts, 'Ada', 'Bo', 'aid', _AID, step=3, lasts=10)

    self.assertEqual([o.id for o in pacts.offers_to('Bo', 4)], [offer])
    self.assertIsNone(pacts.accept('Bo', offer, step=3))  # Not yet.
    self.assertIsNone(pacts.accept('Cy', offer, step=4))  # Not Cy's.
    pact = _accept(pacts, 'Bo', offer, step=4)

    found = pacts.between('Bo', 'Ada', 'aid')[0]
    self.assertEqual(
        (found.id, found.parties, found.made, found.until),
        (pact, ('Ada', 'Bo'), 4, 14),
    )
    self.assertEqual(pacts.expire(13), [])
    ended = pacts.expire(14)
    self.assertEqual([(p.id, p.why_ended) for p in ended], [(pact, 'term')])
    self.assertEqual(pacts.size(), 0)

  def test_an_offer_unanswered_lapses_and_one_may_decline(self):
    pacts = pacts_lib.Pacts(answer_within=1)
    first = _offer(pacts, 'Ada', 'Bo', 'aid', _AID, step=0)
    second = _offer(pacts, 'Cy', 'Bo', 'aid', _AID, step=0)

    self.assertTrue(pacts.decline('Bo', second))
    pacts.expire(2)
    self.assertIsNone(pacts.accept('Bo', first, step=2))
    self.assertEqual(pacts.offers_to('Bo', 1), [])

  def test_a_breach_past_tolerance_breaks_the_pact(self):
    pacts = pacts_lib.Pacts(tolerance=1)
    pact = _accept(
        pacts, 'Bo', _offer(pacts, 'Ada', 'Bo', 'aid', _AID, step=0), step=1
    )

    first = pacts.breach(pact, by='Bo', step=5, why='kept full stores')
    assert first is not None
    self.assertIsNone(first.ended)
    second = pacts.breach(pact, by='Bo', step=7, why='again')
    assert second is not None
    self.assertEqual((second.ended, second.why_ended), (7, 'broken'))
    self.assertEqual(
        second.breaches, ((5, 'Bo', 'kept full stores'), (7, 'Bo', 'again'))
    )
    self.assertEqual(pacts.history()[-1].id, pact)
    self.assertIsNone(pacts.breach(pact, by='Bo', step=8))

  def test_one_pact_of_a_kind_between_two_and_one_who_leaves(self):
    pacts = pacts_lib.Pacts()
    offer = _offer(pacts, 'Ada', 'Bo', 'aid', _AID, step=0)
    self.assertEqual(pacts.offer('Bo', 'Ada', 'aid', _AID, step=0), offer)
    _accept(pacts, 'Bo', offer, step=1)

    self.assertIsNone(pacts.offer('Bo', 'Ada', 'aid', _AID, step=1))
    self.assertIsNotNone(pacts.offer('Bo', 'Ada', 'peace', 'No raids.', step=1))
    self.assertIsNone(pacts.offer('Ada', 'Ada', 'aid', _AID, step=1))
    ended = pacts.forget('Ada', step=2)
    self.assertEqual([p.why_ended for p in ended], ['left'])
    self.assertEqual(pacts.offers_to('Ada', 2), [])

  def test_only_so_many_hold_and_wait(self):
    pacts = pacts_lib.Pacts(max_pacts=1, max_offers=1, max_history=1)
    _offer(pacts, 'Ada', 'Bo', 'aid', _AID, step=0)
    latest = _offer(pacts, 'Ada', 'Cy', 'aid', _AID, step=0)

    self.assertEqual([o.id for o in pacts.offers_to('Cy', 1)], [latest])
    self.assertEqual(pacts.offers_to('Bo', 1), [])
    _accept(pacts, 'Cy', latest, step=1)
    _accept(
        pacts, 'Di', _offer(pacts, 'Ada', 'Di', 'aid', _AID, step=1), step=2
    )
    self.assertEqual(pacts.size(), 1)
    self.assertEqual(pacts.history()[0].why_ended, 'crowded out')

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'answer_within'):
      pacts_lib.Pacts(answer_within=0)
    with self.assertRaisesRegex(ValueError, 'tolerance'):
      pacts_lib.Pacts(tolerance=-1)


class RoundTripTest(absltest.TestCase):

  def test_the_pacts_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              pacts_lib.DEFAULT_PACTS_COMPONENT_KEY: pacts_lib.Pacts()
          },
      )

    def pacts_of(gm):
      return gm.get_component(
          pacts_lib.DEFAULT_PACTS_COMPONENT_KEY, type_=pacts_lib.Pacts
      )

    gm = game_master()
    made = _offer(
        pacts_of(gm), 'Ada', 'Bo', 'aid', _AID, step=0, lasts=9, data={'n': 1}
    )
    _accept(pacts_of(gm), 'Bo', made, step=1)
    pacts_of(gm).breach(made, by='Bo', step=2, why='hoarded')
    _offer(pacts_of(gm), 'Cy', 'Di', 'peace', 'No raids.', step=2)
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    self.assertEqual(pacts_of(again).get_state(), pacts_of(gm).get_state())
    self.assertEqual(
        pacts_of(again).offers_to('Di', 3), pacts_of(gm).offers_to('Di', 3)
    )
    saved = pacts_of(gm).get_state()
    pacts_of(again).set_state(
        {'offers': [*saved['offers'], {'id': 'p9'}]}  # No 'next'.
    )
    waiting = pacts_of(again).offers_to('Di', 3)
    self.assertLen(waiting, 1)  # The bad offer skipped.
    later = pacts_of(again).offer('Ada', 'Cy', 'aid', _AID, step=3)
    self.assertNotIn(later, {made, waiting[0].id})  # No id reused.
    with self.assertRaises(TypeError):  # Data handed out is read-only.
      waiting[0].data['n'] = 2
    pacts_of(again).set_state({})  # A state from before it kept any.
    self.assertEqual(pacts_of(again).size(), 0)


if __name__ == '__main__':
  absltest.main()

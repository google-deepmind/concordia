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

"""Tests for gifts and debts."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import reciprocity
from concordia.language_model import no_language_model


class ReciprocityTest(absltest.TestCase):

  def test_a_gift_is_owed_back_within_the_term(self):
    ledger = reciprocity.Reciprocity(term=5, return_share=0.5)

    debt = ledger.give('Ada', 'Bo', 2.0, step=10, good='food')

    assert debt is not None
    self.assertEqual(
        (debt.debtor, debt.creditor, debt.amount, debt.due),
        ('Bo', 'Ada', 1.0, 15),
    )
    self.assertEqual(ledger.owes('Bo', 'Ada'), 1.0)
    self.assertEqual(ledger.owed_to('Ada'), [debt])
    self.assertIsNone(ledger.give('Ada', 'Ada', 2.0, step=10))
    self.assertIsNone(ledger.give('Ada', 'Bo', 0.0, step=10))

  def test_a_return_settles_the_oldest_debts_first(self):
    ledger = reciprocity.Reciprocity()
    ledger.give('Ada', 'Bo', 1.0, step=1)
    ledger.give('Cy', 'Bo', 1.0, step=2)
    ledger.give('Ada', 'Bo', 2.0, step=3)

    paid = ledger.repay('Bo', 'Ada', 2.5, step=4)

    self.assertEqual(paid, 2.5)
    self.assertEqual(
        [(d.creditor, d.amount) for d in ledger.owed_by('Bo')],
        [('Cy', 1.0), ('Ada', 0.5)],
    )
    self.assertEqual(ledger.repay('Bo', 'Ada', 3.0, step=5), 0.5)
    self.assertEqual(ledger.owes('Bo', 'Ada'), 0.0)

  def test_a_debt_past_due_is_reported_once_and_may_still_be_repaid(self):
    ledger = reciprocity.Reciprocity(term=2)
    ledger.give('Ada', 'Bo', 1.0, step=0)

    self.assertEqual(ledger.overdue(step=2), [])  # Due by step 2.
    defaulted = ledger.overdue(step=3)
    self.assertEqual([d.debtor for d in defaulted], ['Bo'])
    self.assertEqual(ledger.overdue(step=4), [])  # Once only.
    self.assertEqual(ledger.repay('Bo', 'Ada', 1.0, step=4), 1.0)
    self.assertEqual(ledger.size(), 0)

  def test_a_creditor_may_forgive(self):
    ledger = reciprocity.Reciprocity()
    ledger.give('Ada', 'Bo', 1.0, step=0)
    ledger.give('Ada', 'Bo', 2.0, step=1)
    ledger.give('Cy', 'Bo', 1.0, step=1)

    self.assertEqual(ledger.forgive('Ada', 'Bo'), 3.0)
    self.assertEqual([d.creditor for d in ledger.owed_by('Bo')], ['Cy'])

  def test_an_estate_passes_its_claims_and_cancels_its_debts(self):
    ledger = reciprocity.Reciprocity()
    ledger.give('Ada', 'Bo', 1.0, step=0)  # Bo owes Ada.
    ledger.give('Cy', 'Ada', 2.0, step=0)  # Ada owes Cy.

    ledger.settle_estate('Ada', heir='Di')

    self.assertEqual(ledger.owes('Bo', 'Di'), 1.0)
    self.assertEqual(ledger.owed_by('Ada'), [])
    self.assertEqual(ledger.owed_to('Cy'), [])
    passing = reciprocity.Reciprocity()
    passing.give('Cy', 'Ada', 2.0, step=0)
    passing.settle_estate('Ada', heir='Di', debts_pass=True)
    self.assertEqual(passing.owes('Di', 'Cy'), 2.0)
    none = reciprocity.Reciprocity()
    none.give('Ada', 'Bo', 1.0, step=0)
    none.settle_estate('Ada')
    self.assertEqual(none.size(), 0)

  def test_a_debtor_keeps_only_so_many_open_debts(self):
    ledger = reciprocity.Reciprocity(max_open=2)
    for step in range(4):
      ledger.give('Ada', 'Bo', 1.0, step=step)

    self.assertEqual([d.given for d in ledger.owed_by('Bo')], [2, 3])

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'term'):
      reciprocity.Reciprocity(term=0)
    with self.assertRaisesRegex(ValueError, 'return_share'):
      reciprocity.Reciprocity(return_share=-1.0)


class RoundTripTest(absltest.TestCase):

  def test_the_ledger_is_saved_and_restored_with_its_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              reciprocity.DEFAULT_RECIPROCITY_COMPONENT_KEY: (
                  reciprocity.Reciprocity(term=2)
              )
          },
      )

    def ledger_of(gm):
      return gm.get_component(
          reciprocity.DEFAULT_RECIPROCITY_COMPONENT_KEY,
          type_=reciprocity.Reciprocity,
      )

    gm = game_master()
    ledger_of(gm).give('Ada', 'Bo', 1.0, step=0, good='food')
    ledger_of(gm).give('Cy', 'Bo', 2.0, step=1, good='food')
    ledger_of(gm).overdue(step=3)  # Ada's is reported.
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    restored = ledger_of(again)
    self.assertEqual(restored.get_state(), ledger_of(gm).get_state())
    self.assertEqual(restored.overdue(step=4), ledger_of(gm).overdue(step=4))
    later = restored.give('Ada', 'Cy', 1.0, step=4)
    assert later is not None
    self.assertEqual(later.id, 'd3')
    restored.set_state({})  # A state from before it kept a ledger.
    self.assertEqual(restored.size(), 0)


if __name__ == '__main__':
  absltest.main()

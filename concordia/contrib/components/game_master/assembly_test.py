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

"""Tests for an assembly."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import assembly
from concordia.language_model import no_language_model

_LAW = 'We spare the fields of others.'
_ALL = ['Ada', 'Bo', 'Cy', 'Di']


def _propose(council: assembly.Assembly, *args, **kwargs) -> str:
  motion = council.propose(*args, **kwargs)
  assert motion is not None
  return motion


class AssemblyTest(parameterized.TestCase):

  def test_a_motion_is_open_after_it_is_proposed_and_passes_into_law(self):
    council = assembly.Assembly()
    motion = _propose(council, 'Ada', _LAW, step=3, kind='spare', data={'n': 1})

    self.assertFalse(council.vote('Bo', motion, True, step=3))  # Not yet.
    self.assertTrue(council.vote('Bo', motion, True, step=4))
    self.assertTrue(council.vote('Cy', motion, False, step=4))
    self.assertTrue(council.vote('Cy', motion, True, step=4))  # Changed.
    self.assertEqual(council.open_motions(4)[0]['for'], 2)
    decided = council.close(step=4, eligible=_ALL)

    self.assertEqual(
        decided,
        [
            assembly.Decision(
                motion_id=motion,
                text=_LAW,
                kind='spare',
                proposer='Ada',
                passed=True,
                votes_for=['Bo', 'Cy'],
                votes_against=[],
                eligible=4,
            )
        ],
    )
    law = council.in_force()[0]
    self.assertEqual(
        (law['text'], law['kind'], law['data'], law['passed'], law['expires']),
        (_LAW, 'spare', {'n': 1}, 4, None),
    )
    self.assertFalse(council.vote('Di', motion, True, step=5))  # Closed.

  @parameterized.named_parameters(
      ('majority_wins', assembly.MAJORITY, 2, 1, True),
      ('majority_tie_fails', assembly.MAJORITY, 1, 1, False),
      ('two_thirds_wins', assembly.TWO_THIRDS, 2, 1, True),
      ('two_thirds_fails', assembly.TWO_THIRDS, 1, 1, False),
      ('consensus_wins', assembly.CONSENSUS, 3, 0, True),
      ('consensus_fails', assembly.CONSENSUS, 3, 1, False),
  )
  def test_the_rule_decides(self, rule, votes_for, against, passed):
    council = assembly.Assembly(rule=rule)
    motion = _propose(council, 'Ada', _LAW, step=0)
    for voter in _ALL[:votes_for]:
      council.vote(voter, motion, True, step=1)
    for voter in _ALL[votes_for : votes_for + against]:
      council.vote(voter, motion, False, step=1)

    decision = council.close(step=1, eligible=_ALL)[0]

    self.assertEqual(decision.passed, passed)
    self.assertEqual(decision.reason, '' if passed else rule)

  def test_too_few_votes_fail_the_quorum_and_only_the_eligible_count(self):
    council = assembly.Assembly(quorum=0.5)
    motion = _propose(council, 'Ada', _LAW, step=0)
    council.vote('Bo', motion, True, step=1)
    council.vote('Zed', motion, True, step=1)  # Not eligible at the close.

    decision = council.close(step=1, eligible=_ALL)[0]

    self.assertEqual((decision.passed, decision.reason), (False, 'quorum'))
    self.assertEqual(decision.votes_for, ['Bo'])

  def test_a_motion_stays_open_for_its_time(self):
    council = assembly.Assembly(open_for=2)
    council.propose('Ada', _LAW, step=0)

    self.assertEqual(council.close(step=1, eligible=_ALL), [])
    self.assertLen(council.close(step=2, eligible=_ALL), 1)

  def test_laws_lapse_after_their_term_or_by_repeal_and_are_bounded(self):
    council = assembly.Assembly(term=5, max_laws=2)
    for step, text in enumerate(['One.', 'Two.', 'Three.']):
      motion = _propose(council, 'Ada', text, step=step * 2)
      council.vote('Bo', motion, True, step=step * 2 + 1)
      council.close(step=step * 2 + 1, eligible=_ALL)

    self.assertEqual(
        [law['text'] for law in council.in_force()], ['Two.', 'Three.']
    )
    council.close(step=8, eligible=_ALL)  # 'Two.' passed at 3: lapses at 8.
    self.assertEqual([law['text'] for law in council.in_force()], ['Three.'])
    self.assertTrue(council.repeal(council.in_force()[0]['id']))
    self.assertEqual(council.in_force(), [])

  def test_the_same_text_is_one_motion_and_so_many_may_be_open(self):
    council = assembly.Assembly(max_motions=1)
    first = council.propose('Ada', _LAW, step=0)

    self.assertEqual(council.propose('Bo', _LAW, step=0), first)
    self.assertIsNone(council.propose('Bo', 'Another.', step=0))

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'rule'):
      assembly.Assembly(rule='lottery')
    with self.assertRaisesRegex(ValueError, 'quorum'):
      assembly.Assembly(quorum=1.5)


class RoundTripTest(absltest.TestCase):

  def test_motions_and_laws_are_saved_and_restored_with_their_gm(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              assembly.DEFAULT_ASSEMBLY_COMPONENT_KEY: assembly.Assembly(
                  term=10
              )
          },
      )

    def council_of(gm):
      return gm.get_component(
          assembly.DEFAULT_ASSEMBLY_COMPONENT_KEY, type_=assembly.Assembly
      )

    gm = game_master()
    passed = _propose(council_of(gm), 'Ada', _LAW, step=0, kind='spare')
    council_of(gm).vote('Bo', passed, True, step=1)
    council_of(gm).close(step=1, eligible=_ALL)
    pending = _propose(council_of(gm), 'Bo', 'Another.', step=1)
    council_of(gm).vote('Cy', pending, False, step=2)
    again = game_master()
    again.set_state(gm.get_state())

    self.assertEqual(council_of(again).get_state(), council_of(gm).get_state())
    self.assertEqual(
        council_of(again).close(step=2, eligible=_ALL),
        council_of(gm).close(step=2, eligible=_ALL),
    )
    council_of(again).set_state({})  # A state from before it kept any.
    self.assertEqual(council_of(again).in_force(), [])


if __name__ == '__main__':
  absltest.main()

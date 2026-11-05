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

"""Tests for rumours."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import rumors as rumors_lib
from concordia.language_model import no_language_model

_CHAIN = {'Ada': ['Bo'], 'Bo': ['Ada', 'Cy'], 'Cy': ['Bo', 'Di'], 'Di': ['Cy']}
_TEXT = 'The river flooded the low fields by the ford.'


class RumorsTest(absltest.TestCase):

  def test_a_rumour_travels_a_chain_a_step_a_link(self):
    rumors = rumors_lib.Rumors(chance=1.0, garble=0.0, keep=0.5)
    rumor_id = rumors.start(_TEXT, ['Ada'], step=0, about='', kind='flood')

    first = rumors.spread(_CHAIN, step=1)
    second = rumors.spread(_CHAIN, step=2)

    self.assertEqual(
        [(h.hearer, h.teller, h.fidelity) for h in first], [('Bo', 'Ada', 0.5)]
    )
    self.assertEqual([(h.hearer, h.teller) for h in second], [('Cy', 'Bo')])
    self.assertEqual(second[0].fidelity, 0.25)
    self.assertEqual(second[0].kind, 'flood')
    self.assertEqual(rumors.rumors()[0]['id'], rumor_id)
    self.assertEqual(rumors.rumors()[0]['reach'], 3)

  def test_one_tells_only_for_a_while_after_hearing(self):
    rumors = rumors_lib.Rumors(chance=1.0, tell_for=1)
    rumors.start(_TEXT, ['Ada'], step=0)

    rumors.spread({'Ada': ['Bo']}, step=1)
    late = rumors.spread({'Ada': ['Cy']}, step=2)  # Ada told it at step 1.

    self.assertEqual(late, [])

  def test_retellings_drop_words_but_keep_the_first_and_a_few(self):
    rumors = rumors_lib.Rumors(chance=1.0, garble=1.0)
    rumors.start(_TEXT, ['Ada'], step=0)

    texts = [rumors.spread(_CHAIN, step=s)[0].text for s in (1, 2, 3)]
    self.assertEqual(rumors.rumors()[0]['furthest'], texts[-1])

    lengths = [len(t.split()) for t in texts]
    self.assertEqual(lengths, [8, 7, 6])  # A word fewer each retelling.
    self.assertTrue(all(t.startswith('The ') for t in texts))
    short = rumors_lib.Rumors(chance=1.0, garble=1.0)
    short.start('Fire up north.', ['Ada'], step=0)
    self.assertEqual(short.spread(_CHAIN, step=1)[0].text, 'Fire up north.')

  def test_the_draws_replay_and_a_chance_holds_some_back(self):
    def heard(seed):
      rumors = rumors_lib.Rumors(seed=seed, chance=0.5)
      rumors.start(_TEXT, ['Ada'], step=0)
      many = {'Ada': [f'p{i}' for i in range(40)]}
      return [h.hearer for h in rumors.spread(many, step=1)]

    self.assertEqual(heard(3), heard(3))
    self.assertNotEqual(heard(3), heard(4))
    self.assertBetween(len(heard(3)), 10, 30)

  def test_untold_rumours_are_forgotten_and_only_so_many_are_kept(self):
    rumors = rumors_lib.Rumors(chance=1.0, forget_after=2, max_rumors=2)
    rumors.start('First news of the hills.', ['Ada'], step=0)
    rumors.start('Second news of the hills.', ['Ada'], step=0)
    rumors.start('Third news of the hills.', ['Ada'], step=1)

    self.assertEqual([r['id'] for r in rumors.rumors()], ['r3', 'r2'])
    rumors.spread({}, step=2)  # r2 untold since step 0, r3 since step 1.
    self.assertEqual([r['id'] for r in rumors.rumors()], ['r3'])

  def test_what_someone_heard_and_one_who_leaves(self):
    rumors = rumors_lib.Rumors(chance=1.0, garble=0.0)
    rumors.start(_TEXT, ['Ada'], step=0, about='Cy', kind='deed')
    rumors.spread(_CHAIN, step=1)

    known = rumors.known_by('Bo')
    self.assertEqual(
        {k: known[0][k] for k in ('teller', 'about', 'kind', 'heard')},
        {'teller': 'Ada', 'about': 'Cy', 'kind': 'deed', 'heard': 1},
    )
    rumors.forget('Bo')
    self.assertEqual(rumors.known_by('Bo'), [])
    self.assertEqual(rumors.rumors()[0]['reach'], 1)  # Ada alone.

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'chance'):
      rumors_lib.Rumors(chance=1.5)
    with self.assertRaisesRegex(ValueError, 'tell_for'):
      rumors_lib.Rumors(tell_for=0)


class RoundTripTest(absltest.TestCase):

  def test_the_rumours_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              rumors_lib.DEFAULT_RUMORS_COMPONENT_KEY: rumors_lib.Rumors(
                  seed=5, chance=0.7
              )
          },
      )

    def rumors_of(gm):
      return gm.get_component(
          rumors_lib.DEFAULT_RUMORS_COMPONENT_KEY, type_=rumors_lib.Rumors
      )

    gm = game_master()
    rumors_of(gm).start(_TEXT, ['Ada', 'Bo'], step=0, kind='flood')
    rumors_of(gm).spread(_CHAIN, step=1)
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    self.assertEqual(rumors_of(again).get_state(), rumors_of(gm).get_state())
    # Both play on alike.
    self.assertEqual(
        rumors_of(again).spread(_CHAIN, step=2),
        rumors_of(gm).spread(_CHAIN, step=2),
    )
    self.assertEqual(rumors_of(again).start('More.', ['Ada'], step=2), 'r2')
    saved = rumors_of(gm).get_state()
    rumors_of(again).set_state(
        {'rumors': [*saved['rumors'], {'text': 'no id'}]}  # No 'next'.
    )
    self.assertLen(rumors_of(again).rumors(), 1)  # The bad record skipped.
    self.assertEqual(rumors_of(again).start('Later.', ['Ada'], step=3), 'r2')
    rumors_of(again).set_state({})  # A state from before it had rumours.
    self.assertEqual(rumors_of(again).rumors(), [])


if __name__ == '__main__':
  absltest.main()

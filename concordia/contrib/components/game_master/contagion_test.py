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

"""Tests for contagion."""

import json

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.contrib.components.game_master import contagion
from concordia.language_model import no_language_model

_CHAIN = {'Ada': ['Bo'], 'Bo': ['Ada', 'Cy'], 'Cy': ['Bo']}


class ContagionTest(absltest.TestCase):

  def test_it_passes_a_link_a_step_and_illness_ends(self):
    sickness = contagion.Contagion(transmission=1.0, mortality=0.0)
    self.assertEqual(sickness.infect(['Ada'], step=0), ['Ada'])

    first = sickness.step(_CHAIN, step=1)
    second = sickness.step(_CHAIN, step=2)

    self.assertEqual(dict(first.exposed), {'Bo': 'Ada'})
    self.assertEqual(list(first.recovered), ['Ada'])
    self.assertEqual(dict(second.exposed), {'Cy': 'Bo'})
    self.assertEqual(sickness.status('Ada'), (contagion.IMMUNE, 9))
    self.assertEqual(
        sickness.counts(),
        {contagion.EXPOSED: 0, contagion.INFECTIOUS: 1, contagion.IMMUNE: 2},
    )

  def test_a_case_caught_in_a_step_begins_with_the_next(self):
    sickness = contagion.Contagion(transmission=1.0, mortality=0.0)
    sickness.infect(['Ada'], step=5)

    same = sickness.step(_CHAIN, step=5)
    after = sickness.step(_CHAIN, step=6)

    self.assertEqual((dict(same.exposed), list(same.recovered)), ({}, []))
    self.assertEqual(dict(after.exposed), {'Bo': 'Ada'})
    self.assertEqual(list(after.recovered), ['Ada'])

  def test_incubation_delays_passing_it_on(self):
    sickness = contagion.Contagion(transmission=1.0, incubation=1)
    sickness.infect(['Ada'], step=0)

    first = sickness.step(_CHAIN, step=1)
    second = sickness.step(_CHAIN, step=2)

    self.assertEqual(dict(first.exposed), {})  # Ada was incubating.
    self.assertEqual(list(first.sick), ['Ada'])
    self.assertEqual(dict(second.exposed), {'Bo': 'Ada'})
    self.assertEqual(sickness.status('Bo'), (contagion.EXPOSED, 1))

  def test_the_protected_and_the_immune_do_not_catch_it(self):
    sickness = contagion.Contagion(transmission=1.0, mortality=0.0)
    sickness.infect(['Bo'], step=0)

    report = sickness.step(_CHAIN, step=1, protected=['Ada'])

    self.assertEqual(dict(report.exposed), {'Cy': 'Bo'})
    sickness.infect(['Bo'], step=1)  # Bo is immune now.
    self.assertEqual(sickness.status('Bo')[0], contagion.IMMUNE)

  def test_immunity_wanes(self):
    sickness = contagion.Contagion(immunity_for=2, mortality=0.0)
    sickness.infect(['Ada'], step=0)

    for step in (1, 2, 3):
      sickness.step({}, step=step)

    self.assertEqual(sickness.status('Ada'), (contagion.SUSCEPTIBLE, 0))
    never = contagion.Contagion(immunity_for=0, mortality=0.0)
    never.infect(['Ada'], step=0)
    never.step({}, step=1)
    self.assertEqual(never.status('Ada')[0], contagion.SUSCEPTIBLE)

  def test_the_vulnerable_die_more_often_and_draws_replay(self):
    def deaths(seed, vulnerability):
      sickness = contagion.Contagion(seed=seed, mortality=0.2)
      names = [f'p{i}' for i in range(200)]
      sickness.infect(names, step=0)
      return sickness.step({}, step=1, vulnerability=vulnerability).died

    few = deaths(1, {})
    many = deaths(1, {f'p{i}': 3.0 for i in range(200)})
    self.assertEqual(deaths(1, {}), few)
    self.assertBetween(len(few), 20, 60)
    self.assertGreater(len(many), 2 * len(few))

  def test_one_who_leaves_is_forgotten(self):
    sickness = contagion.Contagion()
    sickness.infect(['Ada'], step=0)

    sickness.forget('Ada')

    self.assertEqual(sickness.infectious(), [])

  def test_settings_are_checked(self):
    with self.assertRaisesRegex(ValueError, 'transmission'):
      contagion.Contagion(transmission=2.0)
    with self.assertRaisesRegex(ValueError, 'infectious_for'):
      contagion.Contagion(infectious_for=0)


class RoundTripTest(absltest.TestCase):

  def test_the_cases_are_saved_and_restored_with_their_game_master(self):
    def game_master():
      return entity_agent_with_logging.EntityAgentWithLogging(
          agent_name='Village',
          act_component=concat_act_component.ConcatActComponent(
              model=no_language_model.NoLanguageModel()
          ),
          context_components={
              contagion.DEFAULT_CONTAGION_COMPONENT_KEY: contagion.Contagion(
                  seed=3, transmission=0.6, incubation=1
              )
          },
      )

    def sickness_of(gm):
      return gm.get_component(
          contagion.DEFAULT_CONTAGION_COMPONENT_KEY,
          type_=contagion.Contagion,
      )

    gm = game_master()
    sickness_of(gm).infect(['Ada', 'Bo'], step=0)
    sickness_of(gm).step(_CHAIN, step=1)
    again = game_master()
    # Through JSON, as a checkpoint is saved.
    again.set_state(json.loads(json.dumps(gm.get_state())))

    self.assertEqual(
        sickness_of(again).get_state(), sickness_of(gm).get_state()
    )
    self.assertEqual(
        sickness_of(again).step(_CHAIN, step=2),
        sickness_of(gm).step(_CHAIN, step=2),
    )
    sickness_of(again).set_state({})  # A state from before it kept cases.
    self.assertEqual(sickness_of(again).infectious(), [])


if __name__ == '__main__':
  absltest.main()

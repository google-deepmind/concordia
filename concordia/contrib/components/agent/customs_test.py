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

"""Tests of customs as components and the Learner that keeps them."""

from collections.abc import Mapping, Sequence
import json
from typing import Any, override

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.contrib.components.agent import customs
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class _JoinContexts(entity_component.ActingComponent):
  """Acts by joining its contexts, sorted by key."""

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    return '\n'.join(contexts[k] for k in sorted(contexts) if contexts[k])

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _spec(
    custom_id: str,
    kind: str = 'belief',
    domain: str = 'sacred',
    level: int | None = None,
) -> dict[str, Any]:
  return {
      'id': custom_id,
      'kind': kind,
      'domain': domain,
      'text': f'The way of {custom_id}.',
      'params': {} if level is None else {'level': level},
  }


def _agent(
    learner: customs.Learner, name: str = 'Ada'
) -> entity_agent.EntityAgent:
  return entity_agent.EntityAgent(
      name,
      act_component=_JoinContexts(),
      context_components={'learner': learner},
      component_factory=customs.custom_factory,
  )


def _held(agent: entity_agent.EntityAgent) -> list[str]:
  return sorted(
      c.id
      for c in agent.get_all_context_components().values()
      if isinstance(c, customs.Custom)
  )


class OfferTest(absltest.TestCase):

  def test_an_offer_line_reads_back_whole(self):
    spec = _spec('c7', 'skill', 'food', level=2)

    line = customs.offer_line(spec, 'observation', 'Bo')
    (offer,) = customs.parse_offers(f'Ada sees Bo fish: {line}\nThe sun sets.')

    self.assertEqual(offer.spec, spec)
    self.assertEqual((offer.via, offer.source), ('observation', 'Bo'))

  def test_an_offer_whose_params_do_not_read_is_passed_over(self):
    good = customs.offer_line(_spec('c2'), 'story')
    bad = customs.offer_line(_spec('c1'), 'story') + ' |params={no}'

    self.assertEqual(
        [o.id for o in customs.parse_offers(f'{bad}\n{good}')], ['c2']
    )


class LearnerTest(absltest.TestCase):

  def test_an_offer_taken_up_becomes_a_component_with_its_context(self):
    agent = _agent(customs.Learner(uptake={'observation': 1.0}))

    agent.observe(customs.offer_line(_spec('c1'), 'observation', 'Bo'))

    self.assertEqual(_held(agent), ['c1'])
    self.assertIn('A belief of ours (sacred) [c1]: The way of c1.', agent.act())
    learner = agent.get_component('learner', type_=customs.Learner)
    self.assertEqual(learner.came('c1'), {'via': 'observation', 'from': 'Bo'})

  def test_a_channel_with_no_uptake_moves_nothing(self):
    agent = _agent(customs.Learner(uptake={'observation': 1.0}))

    agent.observe(customs.offer_line(_spec('c1'), 'preaching'))

    self.assertEqual(_held(agent), [])

  def test_a_new_custom_replaces_the_one_at_its_locus(self):
    agent = _agent(customs.Learner(uptake={'story': 1.0}))
    agent.observe(customs.offer_line(_spec('c1'), 'story'))
    agent.observe(customs.offer_line(_spec('c2', domain='kin'), 'story'))

    agent.observe(customs.offer_line(_spec('c3'), 'story'))

    self.assertEqual(_held(agent), ['c2', 'c3'])

  def test_a_level_is_taken_only_if_better(self):
    agent = _agent(customs.Learner(uptake={'apprenticeship': 1.0}))
    agent.observe(
        customs.offer_line(_spec('c1', 'skill', 'food', 3), 'apprenticeship')
    )

    agent.observe(
        customs.offer_line(_spec('c2', 'skill', 'food', 2), 'apprenticeship')
    )
    self.assertEqual(_held(agent), ['c1'])

    agent.observe(
        customs.offer_line(_spec('c3', 'skill', 'food', 4), 'apprenticeship')
    )
    self.assertEqual(_held(agent), ['c3'])

  def test_past_capacity_the_oldest_goes(self):
    learner = customs.Learner(uptake={'story': 1.0}, capacity=2, per_domain=1)
    agent = _agent(learner)
    for i, domain in enumerate(('a', 'b', 'c')):
      agent.observe(
          customs.offer_line(_spec(f'c{i}', 'belief', domain), 'story')
      )
    self.assertEqual(learner.held(), ['c1', 'c2'])

    agent.observe(customs.offer_line(_spec('c9', 'value', 'c'), 'story'))

    self.assertEqual(learner.held(), ['c1', 'c9'])  # One of domain c.

  def test_inherited_customs_are_held_without_a_draw(self):
    learner = customs.Learner(uptake={})
    agent = _agent(learner)

    learner.inherit(_spec('c1'))

    self.assertEqual(_held(agent), ['c1'])
    self.assertEqual(learner.came('c1'), {'via': customs.BIRTH, 'from': ''})

  def test_uptake_is_drawn_from_the_seed_not_from_timing(self):
    def taken(seed: int) -> list[str]:
      agent = _agent(
          customs.Learner(uptake={'story': 0.5}, seed=seed, capacity=30)
      )
      for i in range(20):
        agent.observe(
            customs.offer_line(_spec(f'c{i}', domain=f'd{i}'), 'story')
        )
      return _held(agent)

    self.assertEqual(taken(1), taken(1))
    self.assertNotEqual(taken(1), taken(2))
    self.assertTrue(0 < len(taken(1)) < 20)  # Some are drawn, not all.

  def test_many_doing_the_same_spread_it_faster(self):
    one = customs.offer_line(_spec('c1'), 'observation', 'Bo')
    many = '\n'.join(
        customs.offer_line(_spec('c1'), 'observation', who)
        for who in ('Bo', 'Cy', 'Di', 'Ed', 'Fa', 'Gu')
    )

    def took(observation: str) -> int:
      return sum(
          bool(_held(self._observe(observation, seed))) for seed in range(40)
      )

    self.assertGreater(took(many), took(one))

  def _observe(self, observation: str, seed: int) -> entity_agent.EntityAgent:
    agent = _agent(customs.Learner(uptake={'observation': 0.2}, seed=seed))
    agent.observe(observation)
    return agent

  def test_a_subclass_may_change_chances_and_refuse_customs(self):
    class Traditional(customs.Learner):

      def chance(self, offer: customs.Offer) -> float:
        return 0.0 if offer.via == 'preaching' else 1.0

      def accepts(
          self, spec: Mapping[str, Any], locus: Sequence[Mapping[str, Any]]
      ) -> bool:
        return spec['kind'] != 'value'

    agent = _agent(Traditional(uptake={}))

    agent.observe(
        '\n'.join([
            customs.offer_line(_spec('c1'), 'preaching'),
            customs.offer_line(_spec('c2', 'value', 'x'), 'upbringing'),
            customs.offer_line(_spec('c3', 'norm', 'y'), 'upbringing'),
        ])
    )

    self.assertEqual(_held(agent), ['c3'])

  def test_a_restored_entity_holds_the_customs_it_had_taken_up(self):
    agent = _agent(customs.Learner(uptake={'story': 1.0}))
    agent.observe(customs.offer_line(_spec('c1', 'skill', 'food', 2), 'story'))
    saved = json.loads(json.dumps(agent.get_state()))

    restored = _agent(customs.Learner(uptake={'story': 1.0}))
    restored.set_state(saved)

    self.assertEqual(_held(restored), ['c1'])
    self.assertEqual(restored.act(), agent.act())
    learner = restored.get_component('learner', type_=customs.Learner)
    self.assertEqual(learner.came('c1'), {'via': 'story', 'from': ''})


if __name__ == '__main__':
  absltest.main()

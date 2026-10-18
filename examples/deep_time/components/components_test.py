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

"""Tests for the World GM components."""

from absl.testing import absltest
from concordia.components.game_master import next_acting
from concordia.environment import engine as engine_lib
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time.components import action_specs
from examples.deep_time.components import resolution
from concordia.typing import entity as entity_lib


class _EchoActor(actors_lib.Actor):

  def __init__(self, name: str, active: bool = True):
    super().__init__(name, outcome_topic='test.outcomes')
    self._active = active
    self.actions: list[str] = []

  def is_active(self) -> bool:
    return self._active

  def action_spec(self) -> entity_lib.ActionSpec:
    return entity_lib.choice_action_spec(
        call_to_action='Pick', options=('a', 'b')
    )

  def resolve(self, action: str) -> str:
    self.actions.append(action)
    return f'{self.name} did {action}'


def _spec_call(name: str) -> entity_lib.ActionSpec:
  return entity_lib.ActionSpec(
      call_to_action=next_acting.DEFAULT_CALL_TO_NEXT_ACTION_SPEC.format(
          name=name
      ),
      output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
  )


class SplitActionsTest(absltest.TestCase):

  def test_splits_multiline_actions_by_name(self):
    text = 'Gaia: wait\nRose: grow\nand spread\nUnknown: noise'
    self.assertEqual(
        resolution.split_actions(text, ['Gaia', 'Rose']),
        {'Gaia': 'wait', 'Rose': 'grow\nand spread\nUnknown: noise'},
    )

  def test_ignores_text_before_first_name(self):
    self.assertEqual(
        resolution.split_actions('preamble\nGaia: wait', ['Gaia']),
        {'Gaia': 'wait'},
    )


class ActorComponentsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.context = game_lib.new_game('daisyworld')
    self.active = _EchoActor('Rose')
    self.dormant = _EchoActor('Moss', active=False)
    self.context.add_actor(self.active)
    self.context.add_actor(self.dormant)

  def test_action_specs_per_entity_and_skip(self):
    component = action_specs.ActorActionSpecs(self.context)
    spec = engine_lib.action_spec_parser(component.pre_act(_spec_call('Rose')))
    self.assertEqual(spec.output_type, entity_lib.OutputType.CHOICE)
    self.assertEqual(tuple(spec.options), ('a', 'b'))
    skip = engine_lib.action_spec_parser(component.pre_act(_spec_call('Moss')))
    self.assertEqual(skip.output_type, entity_lib.OutputType.SKIP_THIS_STEP)
    other = entity_lib.ActionSpec(
        call_to_action='x', output_type=entity_lib.OutputType.RESOLVE
    )
    self.assertEqual(component.pre_act(other), '')

  def test_resolution_routes_actions_and_publishes_outcomes(self):
    component = resolution.ActorResolution(self.context)
    component.pre_observe(f'{resolution.PUTATIVE_EVENT_TAG} Rose: b')
    result = component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='?', output_type=entity_lib.OutputType.RESOLVE
        )
    )
    self.assertEqual(result, 'Rose did b')
    self.assertEqual(self.active.actions, ['b'])
    latest = self.context.bus.latest('test.outcomes')
    assert latest is not None
    self.assertEqual(latest.source, 'Rose')
    self.assertEqual(
        self.context.observation_queue.get_and_clear('Rose'), ['Rose did b']
    )

  def test_a_persons_question_is_free_when_a_move_may_be_aimed(self):
    class _Aiming(_EchoActor):

      def aimable(self, option: str) -> bool:
        return option == 'a'

    self.context.add_actor(_Aiming('Sower'))
    component = action_specs.ActorActionSpecs(self.context)
    model_spec = engine_lib.action_spec_parser(
        component.pre_act(_spec_call('Sower'))
    )
    self.assertEqual(model_spec.output_type, entity_lib.OutputType.CHOICE)
    self.context.humans['Sower'] = lambda request: 'a'
    person_spec = engine_lib.action_spec_parser(
        component.pre_act(_spec_call('Sower'))
    )
    self.assertEqual(person_spec.output_type, entity_lib.OutputType.FREE)
    self.assertIn('(a) a\n(b) b', person_spec.call_to_action)
    self.assertIn(actors_lib.AIM_HINT, person_spec.call_to_action)
    # Not aimable: a person still gets the choice.
    self.context.humans['Rose'] = lambda request: 'a'
    rose = engine_lib.action_spec_parser(component.pre_act(_spec_call('Rose')))
    self.assertEqual(rose.output_type, entity_lib.OutputType.CHOICE)

  def test_an_aimed_move_is_aimed_as_it_is_resolved(self):
    self.assertEqual(
        actors_lib.aimed('seed white daisies @ 9, 20'),
        ('seed white daisies', (9, 20)),
    )
    self.assertEqual(actors_lib.aimed('wait'), ('wait', None))
    planet = self.context.planet
    steward = actors_lib.StewardActor('Gaia', planet)
    seed = next(o for o in steward.options() if steward.aimable(o))
    self.assertEqual(steward.take_aim(f'{seed} @ 9,20', planet), seed)
    self.assertEqual(planet.targets['Gaia'], (planet.tiles.rows - 1 - 9, 20))
    # A move that cannot be aimed keeps its words, and aims nothing.
    planet.targets.clear()
    self.assertEqual(steward.take_aim('wait @ 9,20', planet), 'wait @ 9,20')
    self.assertEqual(planet.targets, {})

  def test_active_entities(self):
    self.assertEqual(self.context.active_entities(), ['Rose'])

  def test_adaptation_messages_reach_the_planet(self):
    self.context.bus.publish(
        'bio.species.white.adaptation', 'paler petals', source='White daisies'
    )
    self.context.bus.publish('bio.species.white.behaviour', 'spread')
    events = self.context.deliver_adaptations()
    self.assertLen(events, 1)
    self.assertIn('paler petals', events[0].text)
    self.assertEqual(self.context.planet.pending_adaptations[0].option, 'paler')
    self.assertEqual(self.context.deliver_adaptations(), [])


if __name__ == '__main__':
  absltest.main()

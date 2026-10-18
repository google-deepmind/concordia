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

  def test_the_question_is_free_when_a_move_may_be_aimed(self):
    class _Aiming(_EchoActor):

      def aimable(self, option: str) -> bool:
        return option == 'a'

    self.context.add_actor(_Aiming('Sower'))
    component = action_specs.ActorActionSpecs(self.context)
    # The same question whoever plays it (a model, or a person).
    spec = engine_lib.action_spec_parser(component.pre_act(_spec_call('Sower')))
    self.assertEqual(spec.output_type, entity_lib.OutputType.FREE)
    self.assertIn('(a) a\n(b) b', spec.call_to_action)
    self.assertIn(actors_lib.AIM_HINT, spec.call_to_action)
    # A free answer is read as the move it names, its aim kept.
    sower = self.context.actors['Sower']
    self.assertEqual(sower.understood('Sower (b) b, for the rain'), 'b')
    self.assertEqual(sower.understood('a @ 3,4'), 'a @ 3,4')
    self.assertEqual(sower.understood('nothing at all'), 'nothing at all')
    # Not aimable: the choice.
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

  def test_the_registry_keeps_each_actors_state_with_the_game_master(self):
    registry = self.context.registry
    self.assertIs(registry.actors, self.context.actors)
    component = resolution.ActorResolution(self.context)
    component.pre_observe(f'{resolution.PUTATIVE_EVENT_TAG} Rose: a')
    component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='?', output_type=entity_lib.OutputType.RESOLVE
        )
    )
    self.active.outbox.append(('test.later', 'not yet sent'))
    saved = registry.get_state()
    rose_saved = self.active.get_state()
    self.assertEqual(saved['Rose'], rose_saved)
    self.assertEqual(rose_saved['acts'], 1)
    self.assertEqual(rose_saved['log'][0][1:], ['a', 'Rose did a'])
    fresh = game_lib.new_game('daisyworld')
    rose = _EchoActor('Rose')
    fresh.add_actor(rose)
    fresh.registry.set_state(saved)
    self.assertEqual(rose.acts, 1)
    self.assertEqual(rose.log[0][1:], ('a', 'Rose did a'))
    self.assertEqual(rose.outbox, [('test.later', 'not yet sent')])

  def test_active_entities(self):
    self.assertEqual(self.context.active_entities(), ['Rose'])

  def test_a_chapter_travels_in_the_chroniclers_action(self):
    self.context.add_actor(
        actors_lib.ChroniclerActor('Chronicler', 'chronicle')
    )
    component = resolution.ActorResolution(self.context)
    resolve = entity_lib.ActionSpec(
        call_to_action='?', output_type=entity_lib.OutputType.RESOLVE
    )
    watch = actors_lib.CHRONICLER_RESPONSE
    component.pre_observe(
        f'{resolution.PUTATIVE_EVENT_TAG} Chronicler: {watch}'
    )
    component.pre_act(resolve)
    self.assertEqual(self.context.bus.history('chronicle'), [])
    component.pre_observe(
        f'{resolution.PUTATIVE_EVENT_TAG} Chronicler:'
        f' {actors_lib.HAND_IN_CHAPTER}: The Quiet Years.'
    )
    # Published on the chronicle, though it is no news of the world.
    self.assertEqual(component.pre_act(resolve), '')
    (chapter,) = self.context.bus.history('chronicle')
    self.assertEqual(
        (chapter.payload, chapter.source), ('The Quiet Years.', 'Chronicler')
    )

  def test_an_adaptation_is_enacted_by_the_species_own_action(self):
    white = actors_lib.SpeciesActor(
        'White daisies', 'white', self.context.planet
    )
    # Its slow mind's decision is counsel; its action names the shift.
    self.assertIn(actors_lib.EVOLVE, white.options())
    self.assertIn(
        'named none', white.resolve(actors_lib.EVOLVE)
    )  # No words: nothing changes, and the time is not spent.
    self.assertEqual(self.context.planet.pending_adaptations, [])
    news = white.resolve(f'{actors_lib.EVOLVE}: paler petals')
    self.assertIn('begin evolving', news)
    self.assertEqual(self.context.planet.pending_adaptations[0].option, 'paler')
    # A shift under way is not begun again (its mind paces the next).
    self.assertIn(
        'already evolving', white.resolve(f'{actors_lib.EVOLVE}: paler petals')
    )
    self.assertLen(self.context.planet.pending_adaptations, 1)
    self.assertEqual(
        self.context.planet.get_state()['enacted']['White daisies:adaptation'],
        self.context.planet.tick,
    )


if __name__ == '__main__':
  absltest.main()

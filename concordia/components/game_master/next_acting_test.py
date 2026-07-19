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

"""Tests for NextActingInFixedOrder."""

import threading

from absl.testing import absltest
from concordia.components.game_master import next_acting
from concordia.typing import entity as entity_lib


def _next_acting_spec(options=('alice', 'bob', 'carol')):
  return entity_lib.ActionSpec(
      call_to_action='Who is next?',
      output_type=entity_lib.OutputType.NEXT_ACTING,
      options=options,
  )


_NON_NEXT_ACTING_SPEC = entity_lib.ActionSpec(
    call_to_action='Do something.',
    output_type=entity_lib.OutputType.FREE,
)


class NextActingInFixedOrderTest(absltest.TestCase):

  def test_get_currently_active_player_starts_as_none(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    self.assertIsNone(component.get_currently_active_player())

  def test_pre_act_cycles_through_sequence_in_order(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    turns = [component.pre_act(_next_acting_spec()) for _ in range(5)]
    self.assertEqual(turns, ['alice', 'bob', 'carol', 'alice', 'bob'])

  def test_pre_act_ignores_non_next_acting_action_specs(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    result = component.pre_act(_NON_NEXT_ACTING_SPEC)
    self.assertEqual(result, '')
    self.assertIsNone(component.get_currently_active_player())

  def test_get_currently_active_player_reflects_last_pre_act(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    component.pre_act(_next_acting_spec())
    self.assertEqual(component.get_currently_active_player(), 'alice')
    component.pre_act(_next_acting_spec())
    self.assertEqual(component.get_currently_active_player(), 'bob')

  def test_remove_actor_from_sequence(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    component.remove_actor_from_sequence('bob')
    turns = [component.pre_act(_next_acting_spec()) for _ in range(3)]
    self.assertEqual(turns, ['alice', 'carol', 'alice'])

  def test_remove_unknown_actor_raises(self):
    component = next_acting.NextActingInFixedOrder(['alice'])
    with self.assertRaises(ValueError):
      component.remove_actor_from_sequence('nobody')

  def test_add_actor_to_sequence(self):
    component = next_acting.NextActingInFixedOrder(['alice'])
    component.add_actor_to_sequence('bob')
    turns = [component.pre_act(_next_acting_spec()) for _ in range(3)]
    self.assertEqual(turns, ['alice', 'bob', 'alice'])

  def test_get_state_and_set_state_round_trip(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    component.pre_act(_next_acting_spec())  # advances to 'alice', idx 0

    state = component.get_state()
    restored = next_acting.NextActingInFixedOrder(['someone_else'])
    restored.set_state(state)

    self.assertEqual(restored.get_currently_active_player(), 'alice')
    # The next turn after restoring should continue the original sequence.
    self.assertEqual(restored.pre_act(_next_acting_spec()), 'bob')

  def test_concurrent_pre_act_and_sequence_mutation_do_not_raise(self):
    # NextActingInFixedOrder.pre_act() reads len(self._sequence) and then
    # indexes into it, while remove_actor_from_sequence()/
    # add_actor_to_sequence() mutate that same list. Both are guarded by
    # the same lock so this should never raise, even when driven
    # concurrently from multiple threads.
    names = [f'player_{i}' for i in range(200)]
    component = next_acting.NextActingInFixedOrder(list(names))
    errors = []

    def cycle():
      for _ in range(500):
        try:
          component.pre_act(_next_acting_spec())
          component.get_currently_active_player()
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)

    def mutate():
      for name in names[:150]:
        try:
          component.remove_actor_from_sequence(name)
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)
      for name in names[:150]:
        try:
          component.add_actor_to_sequence(name)
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)

    threads = [
        threading.Thread(target=cycle),
        threading.Thread(target=cycle),
        threading.Thread(target=mutate),
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEmpty(errors)


if __name__ == '__main__':
  absltest.main()

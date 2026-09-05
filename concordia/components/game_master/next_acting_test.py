# Copyright 2023 DeepMind Technologies Limited.
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


# NextActingInFixedOrder picks the next player from its own sequence and
# ignores the options on the action spec, but NEXT_ACTING is a choice-type
# output so the spec must still carry some.
_NEXT_ACTING_SPEC = entity_lib.ActionSpec(
    call_to_action='Who is next?',
    output_type=entity_lib.OutputType.NEXT_ACTING,
    options=('alice', 'bob', 'carol'),
)

_FREE_SPEC = entity_lib.ActionSpec(
    call_to_action='Do something.',
    output_type=entity_lib.OutputType.FREE,
)


class NextActingInFixedOrderTest(absltest.TestCase):

  def test_removing_an_earlier_actor_keeps_the_active_player(self):
    # Regression test: removing an actor positioned before the currently
    # active player used to leave `_currently_active_player_idx` dangling,
    # so `get_currently_active_player` raised IndexError and the following
    # turn repeated the active player instead of advancing to the next one.
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    for _ in range(3):
      component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(component.get_currently_active_player(), 'carol')

    component.remove_actor_from_sequence('alice')

    self.assertEqual(component.get_currently_active_player(), 'carol')
    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'bob')

  def test_removing_the_active_actor_resumes_with_their_successor(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    component.pre_act(_NEXT_ACTING_SPEC)
    component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(component.get_currently_active_player(), 'bob')

    component.remove_actor_from_sequence('bob')

    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'carol')
    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'alice')

  def test_removing_the_active_actor_at_the_front_wraps_around(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'alice')

    component.remove_actor_from_sequence('alice')

    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'bob')
    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'carol')

  def test_removing_a_later_actor_keeps_the_active_player(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    component.pre_act(_NEXT_ACTING_SPEC)
    self.assertEqual(component.get_currently_active_player(), 'alice')

    component.remove_actor_from_sequence('carol')

    self.assertEqual(component.get_currently_active_player(), 'alice')
    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'bob')

  def test_removing_the_last_remaining_actor_clears_the_active_player(self):
    component = next_acting.NextActingInFixedOrder(['alice'])
    component.pre_act(_NEXT_ACTING_SPEC)

    component.remove_actor_from_sequence('alice')

    self.assertIsNone(component.get_currently_active_player())

  def test_removal_before_any_turn_leaves_the_active_player_unset(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    component.remove_actor_from_sequence('alice')
    self.assertIsNone(component.get_currently_active_player())
    self.assertEqual(component.pre_act(_NEXT_ACTING_SPEC), 'bob')

  def test_get_currently_active_player_starts_as_none(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    self.assertIsNone(component.get_currently_active_player())

  def test_pre_act_cycles_through_the_sequence_in_order(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob', 'carol'])
    turns = [component.pre_act(_NEXT_ACTING_SPEC) for _ in range(5)]
    self.assertEqual(turns, ['alice', 'bob', 'carol', 'alice', 'bob'])

  def test_pre_act_ignores_other_action_specs(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    self.assertEqual(component.pre_act(_FREE_SPEC), '')
    self.assertIsNone(component.get_currently_active_player())

  def test_removing_an_unknown_actor_raises(self):
    component = next_acting.NextActingInFixedOrder(['alice'])
    with self.assertRaises(ValueError):
      component.remove_actor_from_sequence('nobody')

  def test_add_actor_to_sequence(self):
    component = next_acting.NextActingInFixedOrder(['alice'])
    component.add_actor_to_sequence('bob')
    turns = [component.pre_act(_NEXT_ACTING_SPEC) for _ in range(3)]
    self.assertEqual(turns, ['alice', 'bob', 'alice'])

  def test_get_state_and_set_state_round_trip(self):
    component = next_acting.NextActingInFixedOrder(['alice', 'bob'])
    component.pre_act(_NEXT_ACTING_SPEC)

    restored = next_acting.NextActingInFixedOrder(['someone_else'])
    restored.set_state(component.get_state())

    self.assertEqual(restored.get_currently_active_player(), 'alice')
    self.assertEqual(restored.pre_act(_NEXT_ACTING_SPEC), 'bob')

  def test_concurrent_turn_taking_and_removal_is_consistent(self):
    # `pre_act` reads `len(self._sequence)` and then indexes into it, while
    # `remove_actor_from_sequence` mutates that same list under `self._lock`.
    # Reading without the lock lets the sequence shrink in between, so the
    # index can fall off the end.
    names = [f'player_{i}' for i in range(100)]
    component = next_acting.NextActingInFixedOrder(list(names))
    errors = []
    start = threading.Barrier(3)

    def take_turns():
      start.wait()
      for _ in range(200):
        try:
          component.pre_act(_NEXT_ACTING_SPEC)
          component.get_currently_active_player()
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)

    def churn_sequence():
      start.wait()
      for name in names[:50]:
        try:
          component.remove_actor_from_sequence(name)
          component.add_actor_to_sequence(name)
        except Exception as e:  # pylint: disable=broad-exception-caught
          errors.append(e)

    threads = [
        threading.Thread(target=take_turns),
        threading.Thread(target=take_turns),
        threading.Thread(target=churn_sequence),
    ]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    self.assertEmpty(errors)


if __name__ == '__main__':
  absltest.main()

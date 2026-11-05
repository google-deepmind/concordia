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

"""Tests for ThreadSafeGenerativeClock following a PlayerRoster."""

from collections.abc import Sequence
from unittest import mock

from absl.testing import absltest
from concordia.components.game_master import player_roster
from concordia.contrib.components.game_master import thread_safe_generative_clock
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal as minimal_entity
from concordia.typing import entity as entity_lib
import numpy as np


def _embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


class _TickingModel(no_language_model.NoLanguageModel):
  """Returns deterministic incremented time strings on open questions."""

  def __init__(self):
    super().__init__()
    self._call_count = 0

  def sample_text(self, prompt: str, **kwargs) -> str:
    del prompt, kwargs
    self._call_count += 1
    return f'Hour {self._call_count - 1}'


def _resolve_spec() -> entity_lib.ActionSpec:
  return entity_lib.ActionSpec(
      call_to_action='Resolve',
      output_type=entity_lib.OutputType.RESOLVE,
  )


class ThreadSafeGenerativeClockTest(absltest.TestCase):

  def _make_roster_and_clock(
      self,
      first_players: Sequence[str],
      aliases: dict[str, str] | None = None,
  ) -> tuple[
      player_roster.PlayerRoster,
      thread_safe_generative_clock.ThreadSafeGenerativeClock,
  ]:
    roster = player_roster.PlayerRoster(
        players=[],
        player_prefabs={'person': minimal_entity.Entity()},
        model=no_language_model.NoLanguageModel(),
        embedder=_embedder,
    )
    for name in first_players:
      roster.join('person', {'name': name})
    mock_gm = mock.MagicMock()
    mock_gm.get_component.return_value = roster
    clock = thread_safe_generative_clock.ThreadSafeGenerativeClock(
        model=_TickingModel(),
        prompt='Advance by 1 hour each round.',
        start_time='Hour 0',
        player_names=[],
        aliases=aliases,
        roster_component_key=player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY,
    )
    clock.set_entity(mock_gm)
    return roster, clock

  def test_player_leaving_before_acting_does_not_block_the_round(self):
    roster, clock = self._make_roster_and_clock(['Alice', 'Bob'])
    self.assertEqual(clock.get_player_names(), frozenset({'Alice', 'Bob'}))

    clock.pre_act(_resolve_spec())
    clock.post_act('Event: Alice posted.')
    self.assertEqual(clock.get_step_count(), 0)

    # Bob leaves through the roster; Alice's next action completes the round.
    roster.leave('Bob')
    clock.pre_act(_resolve_spec())
    clock.post_act('Event: Alice posted again.')
    self.assertEqual(clock.get_step_count(), 1)
    self.assertEqual(clock.get_pre_act_value().strip(), 'Hour 1')

  def test_player_joining_must_act_before_the_next_tick(self):
    roster, clock = self._make_roster_and_clock(
        ['Alice'], aliases={'CyberCy': 'Cy'}
    )
    roster.join('person', {'name': 'Cy'})
    self.assertEqual(clock.get_player_names(), frozenset({'Alice', 'Cy'}))

    clock.pre_act(_resolve_spec())
    clock.post_act('Event: Alice posted.')
    self.assertEqual(clock.get_step_count(), 0)

    clock.pre_act(_resolve_spec())
    clock.post_act('Event: CyberCy posted.')
    self.assertEqual(clock.get_step_count(), 1)

  def test_empty_roster_does_not_tick_the_clock(self):
    roster, clock = self._make_roster_and_clock(['Alice'])
    roster.leave('Alice')
    self.assertEmpty(clock.get_player_names())

    clock.pre_act(_resolve_spec())
    clock.post_act('Event: Nothing.')
    self.assertEqual(clock.get_step_count(), 0)


if __name__ == '__main__':
  absltest.main()

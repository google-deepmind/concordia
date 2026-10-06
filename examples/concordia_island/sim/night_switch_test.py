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

"""Deterministic tests for the island -> nighttime GM switch and island gate."""

from absl.testing import absltest
from concordia.components.game_master import make_observation as make_observation_lib
from concordia.typing import entity as entity_lib

from examples.concordia_island.prefabs import island_gm
from examples.concordia_island.sim import fixed_clock
from examples.concordia_island.sim import time_based_next_gm

_PLAYERS = ('Ann Lee', 'Bo Diaz', 'Cy Park')


class _FakeGameMaster:
  """Minimal stand-in for the island GM entity: a named component registry."""

  def __init__(self, components):
    self.name = 'island rules'
    self._components = components

  def get_component(self, key, type_=None):
    del type_
    if key not in self._components:
      raise KeyError(key)
    return self._components[key]


def _next_gm_spec(agent: str) -> entity_lib.ActionSpec:
  return entity_lib.ActionSpec(
      call_to_action='Which game master should run next?',
      output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
      options=('island rules', 'marketplace_rules'),
      tag=f'next_game_master:{agent}',
  )


class NightSwitchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.clock = fixed_clock.FixedIntervalClock(
        start_time='Thursday, January 1st, 7:00 AM',
        tick_interval_minutes=120,
        waking_hour_start=7,
        waking_hour_end=23,
        player_names=_PLAYERS,
    )
    # The island GM prefab turns this on whenever a nighttime GM exists.
    self.clock.enable_night_gate()
    self.next_gm = time_based_next_gm.TimeBasedNextGM(
        nighttime_gm_name='marketplace_rules',
        social_scheduler_key='',
        player_names=_PLAYERS,
    )
    self.gate = island_gm.TickGatedNextActing(
        player_names=_PLAYERS,
        social_scheduler_key='',
        has_nighttime_gm=True,
    )
    self.make_obs = island_gm.LocationAwareMakeObservation(
        model=None,
        player_names=_PLAYERS,
        social_scheduler_key='',
        allow_llm_fallback=False,
    )
    gm = _FakeGameMaster({
        'clock': self.clock,
        'next_gm': self.next_gm,
        'gate': self.gate,
        'make_obs': self.make_obs,
    })
    for component in (self.clock, self.next_gm, self.gate, self.make_obs):
      component.set_entity(gm)

  def _advance_to_next_morning(self):
    for _ in range(self.clock.ticks_per_day):
      self.clock._advance_to_next_tick()  # pylint: disable=protected-access

  def _route(self, agent: str) -> str:
    return self.next_gm.pre_act(_next_gm_spec(agent))

  def _observe(self, agent: str) -> str:
    return self.make_obs.pre_act(
        entity_lib.ActionSpec(
            call_to_action=(
                make_observation_lib.DEFAULT_CALL_TO_MAKE_OBSERVATION.format(
                    name=agent
                )
            ),
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )

  def test_first_day_has_no_night(self):
    for agent in _PLAYERS:
      self.assertEqual(self._route(agent), 'island rules')
      self.assertTrue(self.gate._is_eligible(agent))  # pylint: disable=protected-access
      self.assertTrue(self.clock.is_agent_past_night(agent))

  def test_agent_is_held_off_island_until_it_finishes_the_night(self):
    for agent in _PLAYERS:
      self._route(agent)
    self._advance_to_next_morning()
    # Simulate an earlier night having completed globally; the old gate let
    # every agent through on this flag, so a racing agent took its 7 AM
    # island action before the marketplace.
    self.clock.mark_nighttime_completed(1)
    self.clock.mark_nighttime_completed(2)
    for agent in _PLAYERS:
      self.assertFalse(self.gate._is_eligible(agent), agent)  # pylint: disable=protected-access

    self.assertEqual(self._route('Ann Lee'), 'marketplace_rules')
    # Ann has *entered* the night but is not back yet: still held, so her
    # island action cannot interleave with the nighttime GM.
    self.assertFalse(self.gate._is_eligible('Ann Lee'))  # pylint: disable=protected-access
    self.assertFalse(self.gate._is_eligible('Bo Diaz'))  # pylint: disable=protected-access
    # Each agent enters each night exactly once; the second query is the
    # nighttime GM handing Ann back, which releases her.
    self.assertEqual(self._route('Ann Lee'), 'island rules')
    self.assertTrue(self.gate._is_eligible('Ann Lee'))  # pylint: disable=protected-access
    self.assertFalse(self.gate._is_eligible('Bo Diaz'))  # pylint: disable=protected-access
    self.assertEqual(self._route('Bo Diaz'), 'marketplace_rules')
    self.assertEqual(self._route('Cy Park'), 'marketplace_rules')

  def test_morning_observations_are_queued_until_after_the_night(self):
    for agent in _PLAYERS:
      self._route(agent)
    self._advance_to_next_morning()
    # The engine flushes the island GM's observations at the switch, while
    # the clock already reads 7:00 AM of the new day.
    self.make_obs.add_to_queue('Ann Lee', 'prepares a meal')
    self.assertEqual(self._route('Ann Lee'), 'marketplace_rules')
    self.assertEqual(self._observe('Ann Lee'), '')
    # Back from the night: the queued morning now lands after the night.
    self.assertEqual(self._route('Ann Lee'), 'island rules')
    released = self._observe('Ann Lee')
    self.assertIn('prepares a meal', released)
    self.assertIn('7:00 AM', released)
    # Nothing was lost or duplicated.
    self.assertEqual(self._observe('Ann Lee'), '')

  def test_gate_is_inert_without_a_nighttime_gm(self):
    clock = fixed_clock.FixedIntervalClock(
        tick_interval_minutes=120, player_names=_PLAYERS
    )
    for _ in range(clock.ticks_per_day):
      clock._advance_to_next_tick()  # pylint: disable=protected-access
    for agent in _PLAYERS:
      self.assertTrue(clock.is_agent_past_night(agent))

  def test_every_night_switches_again_across_month_boundary(self):
    for agent in _PLAYERS:
      self._route(agent)
    # 40 nights: crosses into February, where day-of-month values repeat.
    for _ in range(40):
      self._advance_to_next_morning()
      for agent in _PLAYERS:
        self.assertFalse(self.gate._is_eligible(agent))  # pylint: disable=protected-access
        self.assertEqual(self._route(agent), 'marketplace_rules')
        self.assertFalse(self.gate._is_eligible(agent))  # pylint: disable=protected-access
        self.assertEqual(self._route(agent), 'island rules')
        self.assertTrue(self.gate._is_eligible(agent))  # pylint: disable=protected-access

  def test_night_entries_survive_checkpoint(self):
    self._advance_to_next_morning()
    self._route('Ann Lee')
    self._route('Bo Diaz')
    self._route('Bo Diaz')
    restored = fixed_clock.FixedIntervalClock(
        tick_interval_minutes=120, player_names=_PLAYERS
    )
    restored.set_state(self.clock.get_state())
    ordinal = restored.current_date_ordinal()
    self.assertTrue(restored.has_agent_entered_night('Ann Lee', ordinal))
    self.assertFalse(restored.has_agent_finished_night('Ann Lee', ordinal))
    self.assertFalse(restored.is_agent_past_night('Ann Lee'))
    self.assertTrue(restored.has_agent_finished_night('Bo Diaz', ordinal))
    self.assertTrue(restored.is_agent_past_night('Bo Diaz'))
    self.assertFalse(restored.has_agent_entered_night('Cy Park', ordinal))

  def test_legacy_checkpoint_does_not_hold_agents(self):
    self._advance_to_next_morning()
    state = dict(self.clock.get_state())
    del state['night_entered']
    del state['night_finished']
    restored = fixed_clock.FixedIntervalClock(
        tick_interval_minutes=120, player_names=_PLAYERS
    )
    restored.set_state(state)
    ordinal = restored.current_date_ordinal()
    for agent in _PLAYERS:
      self.assertTrue(restored.has_agent_entered_night(agent, ordinal))
      self.assertTrue(restored.has_agent_finished_night(agent, ordinal))
      self.assertTrue(restored.is_agent_past_night(agent))

  def test_night_gm_resolution_does_not_count_as_island_turn(self):
    proxy = fixed_clock.ClockProxy(self.clock)
    for call in ('What will Ann Lee do next?', 'Resolve the round.'):
      proxy.pre_act(
          entity_lib.ActionSpec(
              call_to_action=call, output_type=entity_lib.OutputType.RESOLVE
          )
      )
    # Neither an attributed nor an unattributed ('__sequential__') nighttime
    # resolution may leave the island clock expecting a daytime turn.
    self.assertEmpty(self.clock._in_resolve_phase)  # pylint: disable=protected-access

  def test_no_night_once_tick_limit_is_reached(self):
    clock = fixed_clock.FixedIntervalClock(
        tick_interval_minutes=120, player_names=_PLAYERS, max_ticks=8
    )
    next_gm = time_based_next_gm.TimeBasedNextGM(
        social_scheduler_key='', player_names=_PLAYERS
    )
    gm = _FakeGameMaster({'clock': clock, 'next_gm': next_gm})
    clock.set_entity(gm)
    next_gm.set_entity(gm)
    next_gm.pre_act(_next_gm_spec('Ann Lee'))
    for _ in range(clock.ticks_per_day):
      clock._advance_to_next_tick()  # pylint: disable=protected-access
    self.assertTrue(clock.reached_max_ticks())
    self.assertEqual(next_gm.pre_act(_next_gm_spec('Ann Lee')), 'island rules')

  def test_global_switching_releases_everyone_at_once(self):
    # Sequential/simultaneous engines: the island GM's own NEXT_ACTING leaks a
    # per-thread capture key, so an attributed query must NOT be treated as a
    # per-entity switch there; the whole population moves together.
    next_gm = time_based_next_gm.TimeBasedNextGM(
        nighttime_gm_name='marketplace_rules',
        social_scheduler_key='',
        player_names=_PLAYERS,
        per_agent_switching=False,
    )
    gate = island_gm.TickGatedNextActing(
        player_names=_PLAYERS,
        social_scheduler_key='',
        has_nighttime_gm=True,
    )
    gm = _FakeGameMaster({'clock': self.clock, 'next_gm': next_gm, 'g': gate})
    next_gm.set_entity(gm)
    gate.set_entity(gm)
    self.assertEqual(next_gm.pre_act(_next_gm_spec('Ann Lee')), 'island rules')
    self._advance_to_next_morning()
    self.assertEqual(
        next_gm.pre_act(_next_gm_spec('Ann Lee')), 'marketplace_rules'
    )
    for agent in _PLAYERS:
      self.assertTrue(self.clock.is_agent_past_night(agent))
      self.assertTrue(gate._is_eligible(agent))  # pylint: disable=protected-access
    # In this mode the nighttime GM signals completion through the clock.
    self.clock.mark_nighttime_completed(self.clock._current_dt.day)  # pylint: disable=protected-access
    self.assertEqual(next_gm.pre_act(_next_gm_spec('Bo Diaz')), 'island rules')


if __name__ == '__main__':
  absltest.main()

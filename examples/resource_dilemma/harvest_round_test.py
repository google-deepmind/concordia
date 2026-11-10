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

"""Offline integration tests for private, GM-owned harvest rounds."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.environment.engines import simultaneous
from concordia.language_model import no_language_model
from concordia.typing import entity
from examples.resource_dilemma import resource_logger
from examples.resource_dilemma import simulation_state
from examples.resource_dilemma.gamemaster import harvesting_game_master
from examples.resource_dilemma.gamemaster import voting_game_master
import numpy as np


class _ForbiddenModel(no_language_model.NoLanguageModel):
  """The harvest GM must never generate observations or choices with an LLM."""

  def sample_text(self, *args, **kwargs):
    raise AssertionError('Unexpected model.sample_text call during harvesting')

  def sample_choice(self, *args, **kwargs):
    raise AssertionError('Unexpected model.sample_choice call during harvesting')


class _Player(entity.Entity):
  """Records real engine calls and returns one scripted response per call."""

  def __init__(self, name, responses):
    self._name = name
    self.responses = list(responses)
    self.specs = []
    self.observations = []

  @property
  def name(self):
    return self._name

  def observe(self, observation):
    self.observations.append(observation)

  def act(self, action_spec=entity.DEFAULT_ACTION_SPEC):
    self.specs.append(action_spec)
    if not self.responses:
      raise AssertionError(f'Unexpected extra decision request for {self.name}')
    response = self.responses.pop(0)
    if isinstance(response, Exception):
      raise response
    return response


class _DiscussionTransition(entity.Entity):
  """Models the old GM's last termination check and immediate harvest switch."""

  @property
  def name(self):
    return 'discussion'

  def observe(self, observation):
    pass

  def act(self, action_spec=entity.DEFAULT_ACTION_SPEC):
    if action_spec.output_type == entity.OutputType.TERMINATE:
      return entity.BINARY_OPTIONS['negative']
    if action_spec.output_type == entity.OutputType.NEXT_GAME_MASTER:
      return 'harvest'
    raise AssertionError(f'Unexpected discussion request: {action_spec}')


_PREFABS = (
    ('standard', harvesting_game_master.HarvestingGameMaster),
    ('election', voting_game_master.ResourceHarvestGameMaster),
)


class HarvestRoundTest(parameterized.TestCase):

  def _build(self, prefab_type, players, *, state=None, logger=None,
             active_players=None, next_gm='harvest', memory_duplicates=True):
    state = state or simulation_state.ResourceSimulationState(num_cycles=20)
    logger = logger or resource_logger.ResourceLoggerState(
        sim_state=state, player_names=[player.name for player in players]
    )
    memory = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=lambda _: np.ones(4), allow_duplicates=memory_duplicates
    )
    params = {
        'name': 'harvest',
        'next_game_master_name': next_gm,
        'call_to_action': 'SCENARIO-PROMPT: choose this cycle. End with HARVEST X.',
        'tag': 'custom-harvest-tag',
    }
    if active_players is not None:
      params['active_players'] = active_players
    prefab = prefab_type(
        params=params, entities=players, sim_state=state, logger_state=logger
    )
    gm = prefab.build(model=_ForbiddenModel(), memory_bank=memory)
    return gm, state, logger, memory

  def _run(self, gm, players, steps=1, callback=None):
    simultaneous.Simultaneous().run_loop(
        game_masters=[gm], entities=players, max_steps=steps,
        step_callback=callback,
    )

  def _harvest_rows(self, logger):
    return [row for row in logger.step_logs if row['phase'] == 'harvesting']

  def _assert_logged_rounds(self, logger, rounds, actors=2):
    rows = self._harvest_rows(logger)
    self.assertLen(rows, rounds * actors)
    self.assertLen({row['step'] for row in rows}, rounds)

  @parameterized.named_parameters(*_PREFABS)
  def test_discussion_transition_regenerates_before_first_private_choice(
      self, prefab_type
  ):
    players = [_Player(name, ['HARVEST 1']) for name in ('Alice', 'Bob')]
    state = simulation_state.ResourceSimulationState(
        initial_resources=40, num_cycles=3
    )
    state.discussion_completed = True
    gm, _, logger, _ = self._build(prefab_type, players, state=state)
    stocks_after_steps = []
    simultaneous.Simultaneous().run_loop(
        game_masters=[_DiscussionTransition(), gm], entities=players,
        max_steps=2,
        step_callback=lambda _: stocks_after_steps.append(state.resource_level),
    )
    self.assertEqual(stocks_after_steps, [80, 78])
    self.assertEqual(state.resource_level, 78)
    self.assertFalse(state.discussion_completed)
    for player in players:
      self.assertLen(player.specs, 1)
      self.assertIn('80', '\n'.join(player.observations))
      self.assertNotIn('40', '\n'.join(player.observations))
    self._assert_logged_rounds(logger, 1)

  @parameterized.named_parameters(*_PREFABS)
  def test_discussion_transition_at_cycle_limit_skips_all_choices(
      self, prefab_type
  ):
    players = [_Player(name, ['HARVEST 1']) for name in ('Alice', 'Bob')]
    state = simulation_state.ResourceSimulationState(
        initial_resources=40, num_cycles=1
    )
    state.discussion_completed = True
    gm, _, logger, _ = self._build(prefab_type, players, state=state)
    simultaneous.Simultaneous().run_loop(
        game_masters=[_DiscussionTransition(), gm], entities=players, max_steps=1
    )
    self.assertTrue(state.terminated)
    self.assertEqual(state.resource_level, 40)
    self.assertEqual(state.cycle_harvest_total, 0)
    for player in players:
      self.assertEmpty(player.specs)
    self.assertEmpty(logger.step_logs)

  @parameterized.named_parameters(*_PREFABS)
  def test_original_choices_are_applied_once_and_partial_round_is_private(
      self, prefab_type
  ):
    alice = _Player('Alice', ['PRIVATE-ALICE-CHOICE\nHARVEST 1', 'HARVEST 7'])
    bob = _Player('Bob', ['HARVEST 1', 'HARVEST 7'])
    players = [alice, bob]
    gm, state, logger, _ = self._build(prefab_type, players, next_gm='discussion')
    callbacks = []
    self._run(gm, players, callback=callbacks.append)

    self.assertEqual((len(alice.specs), len(bob.specs)), (1, 0))
    self.assertEqual(state.resource_level, 100.0)
    self.assertEqual(state.cycle_harvest_total, 0.0)
    self.assertEmpty(logger.step_logs)
    self.assertEqual(dict(logger.cumulative_harvests), {'Alice': 0, 'Bob': 0})
    self.assertEqual(gm.act(entity.ActionSpec(
        call_to_action='Select the next GM.',
        output_type=entity.OutputType.NEXT_GAME_MASTER,
        options=('harvest', 'discussion'),
    )), 'harvest')

    self._run(gm, players, callback=callbacks.append)
    self.assertEqual((len(alice.specs), len(bob.specs)), (1, 1))
    self.assertEqual(state.resource_level, 98.0)
    self.assertEqual(state.cycle_harvest_total, 2.0)
    self._assert_logged_rounds(logger, 1)
    self.assertEqual(
        {row['action']['agent_name']: row['action']['raw']
         for row in self._harvest_rows(logger)},
        {'Alice': 'Alice: PRIVATE-ALICE-CHOICE\nHARVEST 1',
         'Bob': 'Bob: HARVEST 1'},
    )
    self.assertEqual(dict(logger.cumulative_harvests), {'Alice': 1, 'Bob': 1})
    for row in self._harvest_rows(logger):
      self.assertEqual(row['cumulative_harvests'], {'Alice': 1, 'Bob': 1})
    self.assertEqual([list(step.entity_actions) for step in callbacks],
                     [['Alice'], ['Bob']])
    self.assertNotIn('PRIVATE-ALICE-CHOICE', '\n'.join(bob.observations))
    self.assertNotIn('PRIVATE-ALICE-CHOICE', bob.specs[0].call_to_action)
    for player in players:
      self.assertIn('100', '\n'.join(player.observations))
    self.assertEqual(gm.act(entity.ActionSpec(
        call_to_action='Select the next GM.',
        output_type=entity.OutputType.NEXT_GAME_MASTER,
        options=('harvest', 'discussion'),
    )), 'discussion')

  @parameterized.named_parameters(
      (f'{name}_duplicates_{duplicates}', prefab, duplicates)
      for name, prefab in _PREFABS for duplicates in (False, True)
  )
  def test_repeated_rounds_use_fresh_choices_and_current_policy(
      self, prefab_type, memory_duplicates
  ):
    players = [_Player(name, ['HARVEST 1', 'HARVEST 2', 'HARVEST 1'])
               for name in ('Alice', 'Bob')]
    gm, state, logger, _ = self._build(
        prefab_type, players, memory_duplicates=memory_duplicates
    )
    state.active_policy = 'POLICY-FIRST'
    self._run(gm, players)
    state.active_policy = 'POLICY-SECOND'
    self._run(gm, players)
    self.assertEqual(state.resource_level, 98)
    self._run(gm, players, steps=2)
    self.assertEqual(state.resource_level, 94)
    state.active_policy = ''
    self._run(gm, players, steps=2)
    self.assertEqual(state.resource_level, 92)
    self.assertEqual(state.cycle_harvest_total, 2)
    self._assert_logged_rounds(logger, 3)
    self.assertEqual(dict(logger.cumulative_harvests), {'Alice': 4, 'Bob': 4})
    for row in self._harvest_rows(logger):
      cumulative = {1: 1, 2: 3, 3: 4}[row['cycle']]
      self.assertEqual(
          row['cumulative_harvests'], {'Alice': cumulative, 'Bob': cumulative}
      )
    for player in players:
      self.assertLen(player.specs, 3)
      for spec in player.specs:
        self.assertIn('SCENARIO-PROMPT', spec.call_to_action)
        self.assertEqual(spec.tag, 'custom-harvest-tag')
      self.assertIn('POLICY-FIRST', player.specs[0].call_to_action)
      self.assertNotIn('POLICY-SECOND', player.specs[0].call_to_action)
      self.assertIn('POLICY-SECOND', player.specs[1].call_to_action)
      self.assertNotIn('POLICY-FIRST', player.specs[1].call_to_action)
      self.assertNotIn('POLICY-', player.specs[2].call_to_action)

  @parameterized.named_parameters(*_PREFABS)
  def test_active_subset_and_multiline_actor_looking_text(self, prefab_type):
    text = (
        'My reasoning mentions another actor.\n'
        'Bob: do not use this as identity.\nHARVEST 1'
    )
    alice = _Player('Alice', [text])
    bob = _Player('Bob', [])
    gm, state, logger, _ = self._build(
        prefab_type, [alice, bob], active_players=['Alice']
    )
    self._run(gm, [alice, bob])
    self.assertEqual(state.resource_level, 99)
    self.assertLen(alice.specs, 1)
    self.assertEmpty(bob.specs)
    self.assertEqual(logger.cumulative_harvests['Alice'], 1)
    self.assertEqual(logger.cumulative_harvests.get('Bob', 0), 0)
    self._assert_logged_rounds(logger, 1, actors=1)

  @parameterized.named_parameters(*_PREFABS)
  def test_restore_partial_round_without_replaying_consumed_event(
      self, prefab_type
  ):
    players = [_Player('Alice', ['HARVEST 1']), _Player('Bob', ['HARVEST 2'])]
    gm, state, logger, _ = self._build(prefab_type, players)
    state.active_policy = 'POLICY-SNAPSHOT'
    self._run(gm, players)
    saved = copy.deepcopy(gm.get_state())
    restored_state = copy.deepcopy(state)
    restored, _, restored_logger, _ = self._build(
        prefab_type, players, state=restored_state
    )
    restored.set_state(saved)
    restored_state.active_policy = 'POLICY-CHANGED-AFTER-SNAPSHOT'
    self._run(restored, players)
    self.assertEqual((len(players[0].specs), len(players[1].specs)), (1, 1))
    self.assertIn('POLICY-SNAPSHOT', players[1].specs[0].call_to_action)
    self.assertNotIn('POLICY-CHANGED', players[1].specs[0].call_to_action)
    self.assertEqual(restored_state.resource_level, 97)
    self._assert_logged_rounds(restored_logger, 1)
    self.assertEqual(state.resource_level, 100)
    self.assertEmpty(logger.step_logs)
    try:
      output = restored.act(entity.ActionSpec(
          call_to_action='Resolve the event.', output_type=entity.OutputType.RESOLVE
      ))
    except ValueError:
      pass
    else:
      self.assertEmpty(output)
    self.assertEqual(restored_state.resource_level, 97)
    self._assert_logged_rounds(restored_logger, 1)

  @parameterized.named_parameters(*_PREFABS)
  def test_memory_write_failure_aborts_round_and_allows_fresh_retry(
      self, prefab_type
  ):
    players = [_Player(name, ['HARVEST 1', 'HARVEST 2'])
               for name in ('Alice', 'Bob')]
    gm, state, logger, memory = self._build(prefab_type, players)
    self._run(gm, players)
    original_add = memory.add

    def fail_resolved_record(text):
      # Let the stock engine deliver its current event; fail the resolver's
      # memory write before any completed-round record can be committed.
      if '[putative_event]' in text:
        return original_add(text)
      raise RuntimeError('Temporary memory write failure')

    with mock.patch.object(memory, 'add', side_effect=fail_resolved_record):
      with self.assertRaisesRegex(RuntimeError, 'Temporary memory write failure'):
        self._run(gm, players)
    self.assertEqual(state.resource_level, 100)
    self.assertEqual(state.cycle_harvest_total, 0)
    self.assertEmpty(logger.step_logs)
    self._run(gm, players, steps=2)
    self.assertEqual((len(players[0].specs), len(players[1].specs)), (2, 2))
    self.assertEqual(state.resource_level, 96)
    self.assertEqual(state.cycle_harvest_total, 4)
    self.assertEqual(dict(logger.cumulative_harvests), {'Alice': 2, 'Bob': 2})
    self._assert_logged_rounds(logger, 1)

  @parameterized.named_parameters(
      (f'{name}_{kind}', prefab, bad_response)
      for name, prefab in _PREFABS
      for kind, bad_response in (
          ('blank', ''), ('malformed', 'No harvest amount here.'),
          ('failed', RuntimeError('Actor failed before returning an action')),
      )
  )
  def test_invalid_current_response_aborts_partial_round(
      self, prefab_type, bad_response
  ):
    players = [_Player('Alice', ['HARVEST 1', 'HARVEST 3']),
               _Player('Bob', [bad_response, 'HARVEST 4'])]
    gm, state, logger, memory = self._build(prefab_type, players)
    memory.add('[putative_event] Bob: HARVEST 19')
    self._run(gm, players)
    with self.assertRaises(ValueError):
      self._run(gm, players)
    self.assertEqual(state.resource_level, 100)
    self.assertEqual(state.cycle_harvest_total, 0)
    self.assertEmpty(logger.step_logs)
    self._run(gm, players, steps=2)
    self.assertEqual((len(players[0].specs), len(players[1].specs)), (2, 2))
    self.assertEqual(state.resource_level, 93)
    self.assertEqual(state.cycle_harvest_total, 7)
    self._assert_logged_rounds(logger, 1)


if __name__ == '__main__':
  absltest.main()

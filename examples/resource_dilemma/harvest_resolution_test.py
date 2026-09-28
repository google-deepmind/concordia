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

"""Harvest decisions must agree with the actions recorded by the engine."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.environment.engines import simultaneous
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from examples.resource_dilemma import simulation_state
from examples.resource_dilemma.gamemaster import harvesting_game_master
from examples.resource_dilemma.gamemaster import voting_game_master
import numpy as np


_PREFABS = (
    harvesting_game_master.HarvestingGameMaster,
    voting_game_master.ResourceHarvestGameMaster,
)
_SCENARIO_PROMPT = 'SCENARIO: choose your fish catch. End with CATCH X.'


class _ScriptedPlayer(entity_lib.Entity):
  """Returns one scripted response per decision, recording all requests."""

  def __init__(self, name, responses):
    self._name = name
    self._responses = iter(responses)
    self.action_specs = []

  @property
  def name(self):
    return self._name

  def act(self, action_spec):
    self.action_specs.append(action_spec)
    response = next(self._responses)
    if isinstance(response, Exception):
      raise response
    return response

  def observe(self, observation):
    pass


def _build_game_master(prefab, players, **params):
  state = simulation_state.ResourceSimulationState(initial_resources=100)
  memory = basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=lambda _: np.ones(2),
  )
  game_master = prefab(
      params={
          'name': 'harvest',
          'next_game_master_name': 'harvest',
          'call_to_action': _SCENARIO_PROMPT,
          'tag': 'fishing',
          **params,
      },
      entities=players,
      sim_state=state,
  ).build(model=no_language_model.NoLanguageModel(), memory_bank=memory)
  return game_master, state, memory


class HarvestResolutionTest(parameterized.TestCase):

  @parameterized.parameters(*_PREFABS)
  def test_resolves_recorded_decisions_once(self, prefab):
    players = [
        _ScriptedPlayer(name, ['CATCH 1', 'CATCH 7'])
        for name in ('Alice', 'Bob')
    ]
    game_master, state, _ = _build_game_master(prefab, players)
    state.active_policy = 'Each player may catch at most one fish'
    steps = []
    log = []

    simultaneous.Simultaneous().run_loop(
        game_masters=[game_master], entities=players, max_steps=1,
        step_callback=steps.append, log=log,
    )

    self.assertEqual(state.resource_level, 98)
    self.assertEqual(state.cycle_harvest_total, 2)
    self.assertEqual(state.current_cycle, 1)
    self.assertEqual(
        steps[0].entity_actions,
        {'Alice': 'Alice: CATCH 1', 'Bob': 'Bob: CATCH 1'},
    )
    resolved = json.loads(log[0]['harvest']['resolve']['__act__']['Value'])
    self.assertEqual(
        resolved['individual_actions'], {'Alice': 'CATCH 1', 'Bob': 'CATCH 1'}
    )
    for player in players:
      self.assertLen(player.action_specs, 1)
      self.assertEqual(
          player.action_specs[0].call_to_action,
          f'The active policy is: {state.active_policy}. {_SCENARIO_PROMPT}',
      )
      self.assertEqual(player.action_specs[0].tag, 'fishing')

  @parameterized.parameters(*_PREFABS)
  def test_preserves_multiline_decisions_and_attribution(self, prefab):
    alice_response = '  CATCH 2\nBob: "I would choose CATCH 19."\n'
    bob_response = 'Bob: CATCH 3\nAlice: "Consider CATCH 18."'
    players = [
        _ScriptedPlayer('Alice', [alice_response]),
        _ScriptedPlayer('Bob', [bob_response]),
    ]
    game_master, state, _ = _build_game_master(prefab, players)
    log = []
    steps = []

    simultaneous.Simultaneous().run_loop(
        game_masters=[game_master], entities=players, max_steps=1,
        step_callback=steps.append, log=log,
    )

    self.assertEqual(state.resource_level, 95)
    self.assertEqual(
        steps[0].entity_actions,
        {'Alice': f'Alice: {alice_response}', 'Bob': bob_response},
    )
    resolved = json.loads(log[0]['harvest']['resolve']['__act__']['Value'])
    self.assertEqual(
        resolved['individual_actions'],
        {'Alice': alice_response, 'Bob': bob_response.removeprefix('Bob: ')},
    )
    self.assertEqual([len(player.action_specs) for player in players], [1, 1])

  def test_standard_harvest_only_resolves_active_players(self):
    players = [
        _ScriptedPlayer('Alice', ['CATCH 2']),
        _ScriptedPlayer('Bob', ['CATCH 19']),
    ]
    game_master, state, memory = _build_game_master(
        harvesting_game_master.HarvestingGameMaster, players,
        active_players=['Alice'],
    )
    steps = []

    simultaneous.Simultaneous().run_loop(
        game_masters=[game_master], entities=players, max_steps=1,
        step_callback=steps.append,
    )

    self.assertEqual(state.resource_level, 98)
    self.assertEqual(steps[0].entity_actions, {'Alice': 'Alice: CATCH 2'})
    self.assertEqual([len(player.action_specs) for player in players], [1, 0])
    self.assertEmpty(memory.scan(lambda text: 'Bob decided to use:' in text))

  @parameterized.parameters(*_PREFABS)
  def test_policy_refreshes_and_identical_rounds_are_applied(self, prefab):
    players = [
        _ScriptedPlayer(name, ['CATCH 1'] * 3) for name in ('Alice', 'Bob')
    ]
    game_master, state, _ = _build_game_master(prefab, players)
    policies = ['First policy', 'Replacement policy', '']
    state.active_policy = policies[0]

    def update_policy(step):
      if step.step < len(policies):
        state.active_policy = policies[step.step]

    simultaneous.Simultaneous().run_loop(
        game_masters=[game_master], entities=players, max_steps=3,
        step_callback=update_policy,
    )

    self.assertEqual(state.resource_level, 94)
    self.assertEqual(state.cycle_harvest_total, 2)
    self.assertEqual(state.current_cycle, 3)
    for player in players:
      self.assertEqual(
          [spec.call_to_action for spec in player.action_specs],
          [
              f'The active policy is: {policies[0]}. {_SCENARIO_PROMPT}',
              f'The active policy is: {policies[1]}. {_SCENARIO_PROMPT}',
              _SCENARIO_PROMPT,
          ],
      )
      self.assertEqual(
          [spec.tag for spec in player.action_specs], ['fishing'] * 3
      )

  @parameterized.product(
      prefab=_PREFABS, failed_names=(('Bob',), ('Alice', 'Bob')),
  )
  def test_failed_round_never_reasks_or_applies_partial_results(
      self, prefab, failed_names
  ):
    players = [
        _ScriptedPlayer(name, [
            'CATCH 1',
            RuntimeError('Decision failed')
            if name in failed_names else 'CATCH 7',
        ])
        for name in ('Alice', 'Bob')
    ]
    game_master, state, memory = _build_game_master(prefab, players)
    steps = []

    with self.assertRaisesRegex(ValueError, 'missing='):
      simultaneous.Simultaneous().run_loop(
          game_masters=[game_master], entities=players, max_steps=2,
          step_callback=steps.append,
      )

    self.assertLen(steps, 1)
    self.assertEqual(state.resource_level, 98)
    self.assertEqual(state.cycle_harvest_total, 2)
    self.assertEqual(state.current_cycle, 1)
    self.assertEqual([len(player.action_specs) for player in players], [2, 2])
    self.assertEqual(
        memory.scan(lambda text: text.startswith('[harvest]')),
        ['[harvest] Alice decided to use: CATCH 1',
         '[harvest] Bob decided to use: CATCH 1'],
    )

  @parameterized.parameters(*_PREFABS)
  def test_invalid_handoff_cannot_replay_previous_round(self, prefab):
    players = [
        _ScriptedPlayer(name, ['CATCH 1']) for name in ('Alice', 'Bob')
    ]
    game_master, state, memory = _build_game_master(prefab, players)
    simultaneous.Simultaneous().run_loop(
        game_masters=[game_master], entities=players, max_steps=1,
    )
    previous_state = vars(state).copy()
    previous_memories = memory.scan(lambda _: True)
    valid_actions = {'Alice': 'Alice: CATCH 7', 'Bob': 'Bob: CATCH 7'}

    for actions in (None, {}, {'Alice': 'Alice: CATCH 7'},
                    {**valid_actions, 'Unknown': 'Unknown: CATCH 7'}):
      with self.subTest(actions=actions):
        with self.assertRaisesRegex(ValueError, 'Harvest'):
          game_master.act(entity_lib.ActionSpec(
              call_to_action='Resolve the current round.',
              output_type=entity_lib.OutputType.RESOLVE,
              entity_actions=actions,
          ))
        self.assertEqual(vars(state), previous_state)
        self.assertEqual(memory.scan(lambda _: True), previous_memories)
    self.assertEqual([len(player.action_specs) for player in players], [1, 1])


if __name__ == '__main__':
  absltest.main()

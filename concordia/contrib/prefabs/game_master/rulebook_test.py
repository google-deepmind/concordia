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

"""Tests for the rulebook game master, with seated, scripted entities."""

from collections.abc import Mapping, Sequence
from typing import Any

from absl.testing import absltest
from concordia.components.agent import human_act_component
from concordia.contrib.components.agent import scripted_input
from concordia.contrib.components.game_master import rulebook as rulebook_lib
from concordia.contrib.prefabs.entity import seated
from concordia.contrib.prefabs.game_master import rulebook as rulebook_gm
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
import numpy as np


class RaceToTen:
  """Players take turns adding 1, 2 or 3; whoever reaches 10 wins."""

  def __init__(self, players: Sequence[str]):
    self._players = list(players)
    self.total = 0
    self.turn = 0
    self.winner: str | None = None
    self._outbox: list[tuple[str, str]] = [('all', 'Race to ten begins.')]

  def actors(self) -> Sequence[str]:
    return self._players

  def next_actor(self) -> str:
    actor = self._players[self.turn % len(self._players)]
    self.turn += 1
    return actor

  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    return entity_lib.choice_action_spec(
        call_to_action=f'The total is {self.total}. How much, {actor}?',
        options=('1', '2', '3'),
    )

  def resolve(self, actor: str, action: str) -> str:
    self.total += int(action)
    if self.total >= 10 and self.winner is None:
      self.winner = actor
    event = f'{actor} adds {action}; the total is {self.total}.'
    self._outbox.append(('all', event))
    return event

  def drain_observations(self) -> Sequence[tuple[str, str]]:
    out, self._outbox = self._outbox, []
    return out

  def done(self) -> bool:
    return self.winner is not None

  def get_state(self) -> Mapping[str, Any]:
    return {'total': self.total, 'turn': self.turn, 'winner': self.winner}

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.total = int(state['total'])
    self.turn = int(state['turn'])
    self.winner = state['winner']


def _embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


def _simulation(
    rules: RaceToTen, scripts: Mapping[str, Sequence[str]]
) -> tuple[simulation_lib.Simulation, dict[str, Any]]:
  readers = {
      name: scripted_input.ScriptedInput(s) for name, s in scripts.items()
  }
  instances = [
      prefab_lib.InstanceConfig(
          prefab='seated__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': name,
              'act_component_factory': (
                  lambda order, r=reader: human_act_component.HumanActComponent(
                      r, component_order=order
                  )
              ),
          },
      )
      for name, reader in readers.items()
  ]
  instances.append(
      prefab_lib.InstanceConfig(
          prefab='rulebook__GameMaster',
          role=prefab_lib.Role.GAME_MASTER,
          params={'name': 'Referee', 'rulebook': rules},
      )
  )
  config = prefab_lib.Config(
      default_premise='',
      default_max_steps=20,
      prefabs={
          'seated__Entity': seated.Entity(),
          'rulebook__GameMaster': rulebook_gm.GameMaster(),
      },
      instances=instances,
  )
  sim = simulation_lib.Simulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
      embedder=_embedder,
      engine=sequential.Sequential(),
  )
  return sim, readers


class RulebookGameMasterTest(absltest.TestCase):

  def test_scripted_game_runs_to_the_rulebook_end(self):
    rules = RaceToTen(['Alice', 'Bob'])
    sim, readers = _simulation(
        rules, {'Alice': ['3', '3', '3'], 'Bob': ['1', '1']}
    )
    sim.play(max_steps=20)
    self.assertEqual(rules.total, 11)
    self.assertEqual(rules.winner, 'Alice')
    self.assertEqual(readers['Alice'].remaining, 0)
    # The seat saw the rulebook's observations and its action spec.
    request = readers['Bob'].requests[0]
    self.assertIn('Race to ten begins.', request.context)
    self.assertIn('The total is 3', request.action_spec.call_to_action)

  def test_results_are_kept_and_state_checkpoints(self):
    rules = RaceToTen(['Alice', 'Bob'])
    sim, _ = _simulation(rules, {'Alice': ['2', '2', '2'], 'Bob': ['2', '2']})
    sim.play(max_steps=20)
    gm = sim.get_game_masters()[0]
    resolution = gm.get_component('__resolution__')  # pyrefly: ignore[missing-attribute]
    self.assertLen(resolution.results, 5)  # pyrefly: ignore[missing-attribute]
    state = resolution.get_state()
    other = RaceToTen(['Alice', 'Bob'])
    rulebook_lib.RulebookResolution(other, None, None).set_state(state)  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(other.get_state(), rules.get_state())

  def test_simulation_checkpoint_restores_the_rulebook(self):
    rules = RaceToTen(['Alice', 'Bob'])
    sim, _ = _simulation(rules, {'Alice': ['2', '2'], 'Bob': ['2']})
    sim.play(max_steps=3)
    checkpoint = sim.make_checkpoint_data()
    restored = RaceToTen(['Alice', 'Bob'])
    other, _ = _simulation(restored, {'Alice': [], 'Bob': []})
    other.load_from_checkpoint(checkpoint)
    self.assertEqual(
        restored.get_state(), {'total': 6, 'turn': 3, 'winner': None}
    )

  def test_split_event_strips_a_repeated_name(self):
    self.assertEqual(
        rulebook_lib.split_event('Bob: Bob 2', ['Bob', 'Bobby']), ('Bob', '2')
    )
    self.assertEqual(
        rulebook_lib.split_event('Bobby: 3', ['Bob', 'Bobby']), ('Bobby', '3')
    )
    self.assertIsNone(rulebook_lib.split_event('Carol: 1', ['Bob']))

  def test_needs_a_rulebook(self):
    with self.assertRaises(ValueError):
      _simulation(None, {})  # pyrefly: ignore[bad-argument-type]


if __name__ == '__main__':
  absltest.main()

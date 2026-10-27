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

"""Tests for the player roster."""

from collections.abc import Mapping, MutableSequence
import dataclasses
import json
import threading
from typing import Any, cast

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components.game_master import player_roster
from concordia.environment.engines import asynchronous
from concordia.language_model import language_model
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal as minimal_entity
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as measurements_lib
import numpy as np

_ROSTER = player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY


def _embedder(text: str) -> np.ndarray:
  del text
  return np.ones(3)


def _params(name: str) -> dict[str, Any]:
  return {**minimal_entity.Entity().params, 'name': name}


def _roster(
    players: MutableSequence[entity_lib.Entity],
) -> player_roster.PlayerRoster:
  return player_roster.PlayerRoster(
      players=players,
      player_prefabs={'person': minimal_entity.Entity()},
      model=no_language_model.NoLanguageModel(),
      embedder=_embedder,
  )


def _first_players(*names: str) -> list[entity_lib.Entity]:
  return [
      minimal_entity.Entity(params=_params(name)).build(
          model=no_language_model.NoLanguageModel(),
          memory_bank=associative_memory.AssociativeMemoryBank(
              sentence_embedder=_embedder
          ),
      )
      for name in names
  ]


class _Rules(entity_component.ActingComponent):
  """A game master's rules for the test: all play, and Alice gives way to Cy.

  At the third event it resolves, Cy joins the game and Alice leaves it.
  """

  def __init__(self):
    super().__init__()
    self._lock = threading.Lock()
    self._resolved = 0

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    roster = self.get_entity().get_component(
        _ROSTER, type_=player_roster.PlayerRoster
    )
    kind = action_spec.output_type
    if kind == entity_lib.OutputType.NEXT_ACTING:
      return ', '.join(roster.get_player_names())
    if kind == entity_lib.OutputType.NEXT_ACTION_SPEC:
      return json.dumps({
          'call_to_action': 'What do you do?',
          'output_type': 'free',
          'options': [],
          'tag': None,
      })
    if kind == entity_lib.OutputType.MAKE_OBSERVATION:
      return 'It is a fine day.'
    if kind == entity_lib.OutputType.TERMINATE:
      return entity_lib.BINARY_OPTIONS['negative']
    if kind == entity_lib.OutputType.RESOLVE:
      with self._lock:
        self._resolved += 1
        if self._resolved == 3:
          roster.join('person', {'name': 'Cy'})
          roster.leave('Alice')
      return 'It happens.'
    if action_spec.options:
      return action_spec.options[0]
    return ''

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'resolved': self._resolved}

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._resolved = cast(int, state['resolved'])


@dataclasses.dataclass
class _RosterGameMaster(prefab_lib.Prefab):
  """A game master whose players come and go through its roster."""

  description: str = 'Rules under which players come and go.'  # pyrefly: ignore[bad-override]
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': 'rules'}
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    del memory_bank
    # What all who join share: the asynchronous engine's measurements.
    person = minimal_entity.Entity(
        params={
            **minimal_entity.Entity().params,
            'measurements': measurements_lib.ReactiveMeasurements(),
        }
    )
    roster = player_roster.PlayerRoster(
        players=cast(MutableSequence[entity_lib.Entity], self.entities),
        player_prefabs={'person': person},
        model=model,
        embedder=self.params['embedder'],
    )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=self.params['name'],
        act_component=_Rules(),
        context_components={_ROSTER: roster},
        measurements=measurements_lib.ReactiveMeasurements(),
    )


def _simulation() -> simulation_lib.Simulation:
  config = prefab_lib.Config(
      prefabs={
          'person': minimal_entity.Entity(),
          'rules': _RosterGameMaster(),
      },
      instances=[
          *[
              prefab_lib.InstanceConfig(
                  prefab='person',
                  role=prefab_lib.Role.ENTITY,
                  params={
                      **_params(name),
                      'measurements': measurements_lib.ReactiveMeasurements(),
                  },
              )
              for name in ('Alice', 'Bob')
          ],
          prefab_lib.InstanceConfig(
              prefab='rules',
              role=prefab_lib.Role.GAME_MASTER,
              params={'name': 'rules', 'embedder': _embedder},
          ),
      ],
      default_premise='',
      default_max_steps=6,
  )
  return simulation_lib.Simulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
      embedder=_embedder,
      engine=asynchronous.Asynchronous(sleep_time=0.0),
  )


def _roster_of(sim: simulation_lib.Simulation) -> player_roster.PlayerRoster:
  (game_master,) = sim.game_masters
  return cast(
      entity_agent_with_logging.EntityAgentWithLogging, game_master
  ).get_component(_ROSTER, type_=player_roster.PlayerRoster)


def _memories(player: entity_lib.Entity) -> list[str]:
  memory = cast(
      entity_agent_with_logging.EntityAgentWithLogging, player
  ).get_component('__memory__')
  return memory.get_all_memories_as_text()  # pyrefly: ignore[missing-attribute]


class PlayerRosterTest(absltest.TestCase):

  def test_players_join_and_leave_and_are_found_by_name(self):
    players = _first_players('Alice', 'Bob')
    roster = _roster(players)

    cy = roster.join('person', _params('Cy'))
    alice = roster.leave('Alice')

    self.assertEqual(alice.name, 'Alice')
    self.assertEqual([p.name for p in players], ['Bob', 'Cy'])
    self.assertEqual(roster.get_player_names(), ['Bob', 'Cy'])
    self.assertIs(roster.get_player_by_name('Cy'), cy)
    with self.assertRaises(KeyError):
      roster.get_player_by_name('Alice')
    with self.assertRaises(KeyError):
      roster.leave('Alice')

  def test_a_name_is_taken_while_its_player_plays_or_if_it_was_first(self):
    roster = _roster(_first_players('Alice'))
    roster.join('person', _params('Cy'))
    roster.leave('Alice')

    for name in ('Cy', 'Alice'):
      with self.assertRaises(ValueError):
        roster.join('person', _params(name))
    with self.assertRaises(KeyError):
      roster.join('ghost', _params('Dee'))
    roster.leave('Cy')
    self.assertEqual(roster.join('person', _params('Cy')).name, 'Cy')

  def test_a_player_has_its_prefabs_params_updated_by_its_own(self):
    roster = _roster([])

    cy = roster.join('person', {'name': 'Cy'})

    self.assertEqual(cy.name, 'Cy')
    with self.assertRaises(TypeError):
      roster.join('person', {'name': 'Dee', 'measurements': object()})
    self.assertEqual(roster.get_player_names(), ['Cy'])

  def test_a_roster_without_a_model_cannot_build_players(self):
    roster = player_roster.PlayerRoster(
        players=[], player_prefabs={'person': minimal_entity.Entity()}
    )

    with self.assertRaises(ValueError):
      roster.join('person', _params('Cy'))

  def test_its_state_restores_who_joined_and_who_left(self):
    roster = _roster(_first_players('Alice', 'Bob'))
    cy = roster.join('person', _params('Cy'))
    cy.observe('Cy was born.')
    roster.join('person', _params('Dee'))
    roster.leave('Alice')
    roster.leave('Dee')
    state = json.loads(json.dumps(roster.get_state()))

    players = _first_players('Alice', 'Bob')
    restored = _roster(players)
    restored.join('person', _params('Eve'))  # Joined after the state.
    restored.set_state(state)

    self.assertEqual([p.name for p in players], ['Bob', 'Cy'])
    self.assertEqual(state['left'], ['Alice'])
    self.assertIn(
        'Cy was born.', _memories(restored.get_player_by_name('Cy'))[-1]
    )
    self.assertEqual(restored.get_state(), roster.get_state())


class PlayerRosterSimulationTest(absltest.TestCase):

  def test_players_join_and_leave_a_running_simulation(self):
    sim = _simulation()

    sim.play()

    self.assertEqual([e.name for e in sim.entities], ['Bob', 'Cy'])
    self.assertIn('It is a fine day.', ' '.join(_memories(sim.entities[1])))
    self.assertEqual(
        _roster_of(sim).get_player_by_name('Cy'), sim.get_entities()[1]
    )

  def test_a_checkpoint_restores_the_players_of_its_time(self):
    sim = _simulation()
    sim.play()
    checkpoint = json.loads(json.dumps(sim.make_checkpoint_data()))

    restored = _simulation()
    restored.load_from_checkpoint(checkpoint)

    self.assertEqual([e.name for e in restored.entities], ['Bob', 'Cy'])
    self.assertEqual(
        _memories(restored.entities[1]), _memories(sim.entities[1])
    )
    self.assertEqual(
        _roster_of(restored).get_state(), _roster_of(sim).get_state()
    )


if __name__ == '__main__':
  absltest.main()

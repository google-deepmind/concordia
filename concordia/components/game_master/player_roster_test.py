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
from concordia.components.agent import action_spec_ignored
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import player_roster
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
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


class _WhoPlays(action_spec_ignored.ActionSpecIgnored):
  """A component conditioned on the roster, as other GM components may be."""

  def __init__(self):
    super().__init__('Who plays')

  def _make_pre_act_value(self) -> str:
    return 'Now: ' + self.get_named_component_pre_act_value(_ROSTER)


class _SayContext(entity_component.ActingComponent):
  """Acts by saying its components' contexts, in order."""

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    return ''.join(context.values())

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class _EveryPlayer(entity_component.ContextComponent):
  """Next acting: every player who plays now, as the roster names them."""

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    return (
        self.get_entity()
        .get_component(_ROSTER, type_=player_roster.PlayerRoster)
        .get_pre_act_value()
    )

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class _Turnover(entity_component.ContextComponent):
  """Resolution: whatever is done happens; at the third event, Cy joins the
  game and Alice leaves it. Everyone who plays hears of a fine day.
  """

  def __init__(self, queue: make_observation.ObservationQueue):
    super().__init__()
    self._queue = queue
    self._lock = threading.Lock()
    self._resolved = 0

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    roster = self.get_entity().get_component(
        _ROSTER, type_=player_roster.PlayerRoster
    )
    with self._lock:
      self._resolved += 1
      if self._resolved == 3:
        roster.join('person', {'name': 'Cy'})
        roster.leave('Alice')
    names = roster.get_player_names()
    self._queue.add('all', 'It is a fine day.', names)
    return 'It happens.'

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
    self.check_extra_components()
    # What all who join share: the asynchronous engine's measurements.
    person = minimal_entity.Entity(
        params={
            **minimal_entity.Entity().params,
            'measurements': measurements_lib.ReactiveMeasurements(),
        }
    )
    players = self.entities if self.entities is not None else []
    first = [player.name for player in players]
    queue = make_observation.ObservationQueue()
    components = {
        _ROSTER: player_roster.PlayerRoster(
            players=players,
            player_prefabs={'person': person},
            model=model,
            embedder=self.params['embedder'],
        ),
        next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: _EveryPlayer(),
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            next_acting.FixedActionSpec(
                entity_lib.free_action_spec(call_to_action='What do you do?')
            )
        ),
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: _Turnover(queue),
        make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            make_observation.MakeObservation(
                model=model,
                player_names=first,
                call_to_make_observation=(
                    asynchronous.DEFAULT_CALL_TO_MAKE_OBSERVATION
                ),
                external_queue=queue,
                allow_llm_fallback=False,
            )
        ),
        terminate.DEFAULT_TERMINATE_COMPONENT_KEY: terminate.Terminate(),
    }
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=self.params['name'],
        act_component=switch_act.SwitchAct(model=model, entity_names=first),
        context_components=components,
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

  def test_other_components_may_condition_on_who_plays(self):
    roster = _roster(_first_players('Alice', 'Bob'))
    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='rules',
        act_component=_SayContext(),
        context_components={_ROSTER: roster, 'who plays': _WhoPlays()},
    )

    before = game_master.act()
    roster.join('person', {'name': 'Cy'})
    roster.leave('Alice')
    after = game_master.act()

    self.assertEqual(
        before, '\nPlayers:\nAlice, Bob\nWho plays:\nNow: Alice, Bob\n'
    )
    self.assertEqual(after, '\nPlayers:\nBob, Cy\nWho plays:\nNow: Bob, Cy\n')

  def test_who_plays_is_current_within_a_step(self):
    roster = _roster(_first_players('Alice'))

    class JoinThenSay(_SayContext):

      def get_action_attempt(self, context, action_spec):
        roster.join('person', {'name': 'Cy'})  # After the roster's pre_act.
        return roster.get_pre_act_value()

    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='rules',
        act_component=JoinThenSay(),
        context_components={_ROSTER: roster},
    )

    self.assertEqual(game_master.act(), 'Alice, Cy')

  def test_the_roster_changes_the_very_list_it_is_given(self):
    with self.assertRaises(TypeError):
      player_roster.PlayerRoster(players=tuple(_first_players('Alice')))

  def test_a_state_that_cannot_be_restored_leaves_the_players_be(self):
    players = _first_players('Alice', 'Bob')
    roster = _roster(players)
    state = {
        'joined': {'Cy': {'prefab': 'ghost', 'params': {'name': 'Cy'}}},
        'left': ['Alice'],
    }

    with self.assertRaises(KeyError):
      roster.set_state(state)

    self.assertEqual([p.name for p in players], ['Alice', 'Bob'])

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

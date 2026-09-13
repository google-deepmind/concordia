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

"""Same initial case through browser/headless builds, without execution."""

import copy
import json
from unittest import mock

from concordia.agents import entity_agent
from concordia.environment.engines import sequential
from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether import game_prefab
from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import multiplayer
from concordia.examples.bellwether import researcher
from concordia.examples.bellwether import run
from concordia.prefabs.simulation import generic
import numpy as np
import pytest


def actor_states(simulation: generic.Simulation):
  states = {}
  for actor in simulation.get_entities():
    assert isinstance(actor, entity_agent.EntityAgent)
    states[actor.name] = actor.get_state()
  return states


@pytest.fixture(autouse=True)
def no_execution():
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No run')
  ), mock.patch.object(
      game_prefab.FixtureModel,
      'sample_text',
      side_effect=AssertionError('No model'),
  ):
    yield


@pytest.mark.parametrize('name', researcher.RECIPES)
@pytest.mark.parametrize('logic', ['minimal', 'basic'])
def test_headless_browser_component_parity_and_owned_state(
    tmp_path, name, logic
):
  game = game_service.Game(tmp_path, recipe=name, actor_logic=logic)
  try:
    case = researcher.prepare_case(name, lambda r: 'wait', actor_logic=logic)
    simulation = generic.Simulation(
        case.config,
        game_prefab.FixtureModel(),
        lambda _: np.ones(8),
        engine=sequential.Sequential(),
    )
    assert game.world.get_state() == case.world.get_state()

    assert actor_states(game.simulation) == actor_states(simulation)
    assert game.config.instances == case.config.instances
    assert game.player_view()['recipe'] == case.manifest
    assert game.player_view()['scenario']['opening'] == case.opening
    assert game.inbox.snapshot()['entries'][0]['text'] == case.opening
    events = game.world.data['events']
    assert sum(e['kind'] == 'opening' for e in events) == 1
    assert sum(e['kind'] == 'private_memory' for e in events) == 4
    assert game.world.observations is not case.world.observations
    # Mutating one run cannot alter another run or the public manifest source.
    game.player_view()['recipe']['fixed_roster'].clear()
    assert len(game.case.manifest['fixed_roster']) == 5
    before = copy.deepcopy(case.world.get_state())
    game.world.transfer('Generator', 'Used', 'fuel', 1)
    assert case.world.get_state() == before
  finally:
    game.close()


def test_baseline_dispute_override_preserved_and_conflicts_rejected(tmp_path):
  dispute = {'text': 'Only Nell gets this claim', 'recipients': ['Nell']}
  game = game_service.Game(tmp_path, dispute=dispute)
  try:
    assert game.world.dispute == dispute
    assert game.player_view()['scenario']['opening'] == rules.OPENING
  finally:
    game.close()
  for name in researcher.RECIPES[1:]:
    with pytest.raises(ValueError, match='only with the bellwether'):
      game_service.Game(tmp_path, recipe=name, dispute=dispute)
  with pytest.raises(ValueError, match='supported recipe'):
    game_service.Game(tmp_path, recipe='import.module')


def test_shared_recipe_injection_and_private_initial_information(tmp_path):
  game = multiplayer.SharedGame(
      tmp_path, recipe='institutional-dispute', secure=False
  )
  try:
    actors = {e.name: e for e in game.simulation.get_entities()}
    assert isinstance(actors['Nell'], entity_agent.EntityAgent)
    assert isinstance(actors['Mara'], entity_agent.EntityAgent)
    assert (
        type(actors['Nell'].get_act_component()).__name__ == 'HumanActComponent'
    )
    assert (
        type(actors['Mara'].get_act_component()).__name__
        == 'ConcatActComponent'
    )
    nell_account = next(
        i.params['account']
        for i in game.config.instances
        if i.params['name'] == 'Nell'
    )
    assert nell_account in str(game.world.view('Nell')['journal'])
    for role in ['Coordinator', 'spectator']:
      assert nell_account not in json.dumps(game.view_for(role))
    assert (
        'recollection may be'
        not in game.operations.dispatch(
            'developer',
            {
                'operation': 'game.public_account',
                'arguments': {'format': 'json'},
            },
        )['result']['content']
    )
    assert game.case.manifest['human_roles'] == ['Coordinator', 'Nell']
  finally:
    game.close()


@pytest.mark.parametrize(
    'args',
    [
        ['--recipe', 'mutual-aid'],
        [
            '--mode',
            'fixture',
            '--recipe',
            'mutual-aid',
            '--dispute-file',
            'nonexistent.json',
        ],
        ['--mode', 'fixture', '--recipe', 'arbitrary.module'],
    ],
)
def test_cli_invalid_selection_fails_before_io_or_model(args):
  with mock.patch.object(
      run.game_service, 'Game', side_effect=AssertionError('No build')
  ), mock.patch.object(
      run.ollama_model,
      'OllamaLanguageModel',
      side_effect=AssertionError('No model'),
  ):
    with pytest.raises(SystemExit) as error:
      run.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize('shared', [False, True])
def test_cli_selects_recipe_for_existing_single_and_shared_service(
    tmp_path, shared
):
  built = []

  class RecordedGame(game_service.Game):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      built.append(self)

  class RecordedShared(multiplayer.SharedGame):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      built.append(self)

  module = multiplayer if shared else game_service
  name = 'SharedGame' if shared else 'Game'
  with mock.patch.object(
      module, name, RecordedShared if shared else RecordedGame
  ), mock.patch.object(run.time, 'sleep', side_effect=KeyboardInterrupt):
    run.main(
        [
            '--mode',
            'fixture',
            '--recipe',
            'mutual-aid',
            '--editor-port',
            '0',
            '--player-port',
            '0',
            '--output',
            str(tmp_path),
        ]
        + (['--multiplayer'] if shared else [])
    )
  assert len(built) == 1
  game = built[0]
  assert game.case.manifest['recipe'] == 'mutual-aid'
  assert game.world.inventory_state()['Generator']['fuel'] == 4
  assert game.world.inventory_state()['Used']['fuel'] == 2
  assert game.developer_view()['quiescent']
  assert game.phase == 'paused'
  assert game.simulation.get_raw_log() == []


def test_default_matches_original_game_configuration_without_extra_prompts(
    tmp_path,
):
  game = game_service.Game(tmp_path)
  try:
    world = rules.StormNight(
        rules.new_inventory(), game_prefab.make_observation.ObservationQueue()
    )
    config = game_prefab.configuration(lambda r: 'wait', world)
    simulation = generic.Simulation(
        config,
        game_prefab.FixtureModel(),
        lambda _: np.ones(8),
        engine=sequential.Sequential(),
    )
    world.seed()
    assert game.config.instances == config.instances
    assert game.world.get_state() == world.get_state()
    assert actor_states(game.simulation) == actor_states(simulation)
  finally:
    game.close()

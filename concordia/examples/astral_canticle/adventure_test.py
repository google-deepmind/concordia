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

"""Human/default policy isolation and end-to-end standard-engine unit checks."""

import json
from unittest import mock

from concordia.components.agent import concat_act_component
from concordia.components.agent import human_act_component
from concordia.examples.astral_canticle import adventure
from concordia.examples.astral_canticle import human_io
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
from concordia.utils import structured_logging
import pytest


def test_human_prefab_has_no_model_calls_and_npcs_keep_llm_policy():
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  reader = mock.Mock(return_value='Examine the tuning fork')
  players, gm = adventure.build_cast(model, reader)
  assert isinstance(
      players[0].get_act_component(), human_act_component.HumanActComponent
  )
  assert all(
      isinstance(p.get_act_component(), concat_act_component.ConcatActComponent)
      for p in players[1:]
  )
  result = players[0].act(
      entity_lib.free_action_spec(call_to_action='Your move, {name}')
  )
  assert result == 'Examine the tuning fork'
  assert (
      'resonance cradle' in reader.call_args.args[0].contexts['__observation__']
  )
  model.sample_text.assert_not_called()
  model.sample_choice.assert_not_called()
  players[1].act()
  model.sample_text.assert_called_once()
  assert gm.name == adventure.GM


def test_human_gm_uses_same_component_without_llm_context_calls():
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  reader = mock.Mock(return_value='No')
  players, gm = adventure.build_cast(model, reader, role='gm')
  assert all(
      isinstance(p.get_act_component(), concat_act_component.ConcatActComponent)
      for p in players
  )
  assert (
      gm.act(
          entity_lib.ActionSpec(
              call_to_action='Finished?',
              output_type=entity_lib.OutputType.TERMINATE,
              options=('Yes', 'No'),
          )
      )
      == 'No'
  )
  model.sample_text.assert_not_called()
  model.sample_choice.assert_not_called()


@pytest.mark.parametrize('player_prefab', ('minimal', 'basic'))
def test_three_turns_use_standard_engine_and_produce_standard_log(
    tmp_path, player_prefab
):
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  model.sample_text.return_value = (
      'The resonance cradle glows. The west exit remains open.'
  )
  session = mock.Mock(return_value='Examine the resonance cradle')
  adventure.play(
      model, session, tmp_path, max_steps=3, player_prefab=player_prefab
  )
  assert session.call_count == 1  # NPCs never reach the human adapter.
  assert session.progress.call_count == 3
  assert [c.args[1] for c in session.progress.call_args_list] == [
      'Ilyra Venn',
      'Sable-9',
      'Thorn of Io',
  ]
  session.finish.assert_called_once()
  log = json.loads((tmp_path / 'simulation.json').read_text())
  assert log
  assert '"Source": "human"' in (tmp_path / 'simulation.json').read_text()
  assert (tmp_path / 'log.html').read_text().startswith('<!DOCTYPE html>')


def test_basic_selection_changes_only_ilyra_and_preserves_context_calls():
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  model.sample_text.return_value = 'examining the cradle.'
  reader = mock.Mock(return_value='LOOK')
  players, gm = adventure.build_cast(model, reader, player_prefab='basic')
  original_players, original_gm = adventure.build_cast(model, reader)
  for selected, original in zip(
      [*players[1:], gm], [*original_players[1:], original_gm]
  ):
    assert type(selected.get_act_component()) is type(
        original.get_act_component()
    )
    components = selected.get_all_context_components()
    originals = original.get_all_context_components()
    assert list(components) == list(originals)
    for key, component in components.items():
      assert type(component) is type(originals[key])
      assert component.get_state() == originals[key].get_state()
  assert players[0].act() == 'LOOK'
  assert model.sample_text.call_count == 3  # Perceptions, never an LLM action.
  assert set(reader.call_args.args[0].contexts) == {
      'Instructions',
      'Observation',
      'SelfPerception',
      'SituationPerception',
      'PersonBySituation',
      '__observation__',
      '__memory__',
      'Goal',
  }
  assert (
      players[0]
      .get_component('SituationPerception')
      .get_state()['num_memories_to_retrieve']
      == 25
  )
  assert (
      players[0].get_component('__observation__').get_state()['history_length']
      == 1_000_000
  )


def test_basic_selection_keeps_automated_player_when_human_is_gm():
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  model.sample_text.return_value = 'examining the cradle.'
  reader = mock.Mock(return_value='No')
  players, gm = adventure.build_cast(
      model, reader, role='gm', player_prefab='basic'
  )
  assert isinstance(
      gm.get_act_component(), human_act_component.HumanActComponent
  )
  assert isinstance(
      players[0].get_act_component(), concat_act_component.ConcatActComponent
  )
  players[0].act()
  assert model.sample_text.call_count == 4
  reader.assert_not_called()


def test_unknown_player_prefab_is_rejected():
  with pytest.raises(ValueError, match='Unknown player prefab'):
    adventure.build_cast(None, None, player_prefab='typo')


def test_simulation_owns_the_run_and_exports_each_completed_step(tmp_path):
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  model.sample_text.return_value = 'A final visible consequence at the cradle.'
  session = mock.Mock(return_value='LOOK')
  built = []
  builder = adventure.build_simulation

  def capture(*args, **kwargs):
    simulation = builder(*args, **kwargs)
    built.append(simulation)
    return simulation

  def saved_before_progress(step, actor):
    del actor
    log = structured_logging.SimulationLog.from_json(
        (tmp_path / 'simulation.json').read_text()
    )
    assert len(log.get_steps()) == step
    assert log.get_entity_memories(adventure.PLAYER)
    assert (tmp_path / 'log.html').read_text().startswith('<!DOCTYPE html>')

  session.progress.side_effect = saved_before_progress
  with mock.patch.object(adventure, 'build_simulation', side_effect=capture):
    adventure.play(model, session, tmp_path, max_steps=3)
  assert len(built) == 1
  simulation = built[0]
  assert type(simulation) is generic.Simulation
  assert len(simulation.get_raw_log()) == 3
  assert simulation.get_entity_prefab_config(adventure.PLAYER).prefab == (
      'human_player'
  )
  assert len(simulation.get_game_masters()) == 1
  session.add_observation.assert_called_once()
  assert (
      'final visible consequence' in session.add_observation.call_args.args[0]
  )
  session.finish.assert_called_once()
  assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', [human_io.InputClosed, RuntimeError])
def test_interrupted_input_keeps_completed_steps_without_claiming_completion(
    tmp_path, failure
):
  model = mock.Mock(wraps=no_language_model.NoLanguageModel())
  model.sample_text.return_value = 'The cradle is visible.'
  session = mock.Mock(side_effect=['LOOK', failure('Stopped at next prompt')])
  with pytest.raises(failure):
    adventure.play(model, session, tmp_path, max_steps=4)
  assert session.progress.call_count == 3
  log = structured_logging.SimulationLog.from_json(
      (tmp_path / 'simulation.json').read_text()
  )
  assert len(log.get_steps()) == 3
  assert log.get_game_master_memories()
  session.finish.assert_not_called()
  session.add_observation.assert_not_called()
  assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('role', ['player', 'gm'])
def test_real_transport_is_not_copied_and_builds_have_fresh_components(role):
  def answer():
    session.submit(session.snapshot()['pending']['id'], 'LOOK')

  session = human_io.HumanSession(role=role, on_request=answer)
  model = no_language_model.NoLanguageModel()
  a = adventure.build_simulation(model, session, role=role)
  b = adventure.build_simulation(model, session, role=role)
  cast_a = a.get_entities() + a.get_game_masters()
  cast_b = b.get_entities() + b.get_game_masters()
  for first, second in zip(cast_a, cast_b):
    for key, component in first.get_all_context_components().items():
      assert component is not second.get_component(key)
  assert a.game_master_memory_bank is not b.game_master_memory_bank
  controlled = cast_a[0] if role == 'player' else cast_a[-1]
  assert (
      controlled.act(entity_lib.free_action_spec(call_to_action='Your move'))
      == 'LOOK'
  )
  assert session.snapshot()['entries'][-1]['text'] == 'LOOK'

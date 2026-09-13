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

"""Live CLI composition with stubbed Ollama transport, not a simulation."""

import json
from unittest import mock

from concordia.environment.engines import sequential
from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether import game_prefab
from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import researcher
from concordia.examples.bellwether import run
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
import numpy as np
import pytest


@pytest.mark.parametrize('logic,expected_calls', [('minimal', 1), ('basic', 4)])
def test_component_prompts_keep_their_own_output_contract(
    tmp_path, logic, expected_calls
):
  built = []
  original_game = game_service.Game
  prompts = []
  formats = []
  models = []

  class RecordedGame(original_game):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      built.append(self)
      models.extend([kwargs['model'], kwargs['action_model']])

  def generate(**kwargs):
    prompt = kwargs['prompt']
    prompts.append(prompt)
    formats.append(kwargs.get('format'))
    assert kwargs['options']['num_predict'] <= 256
    result = (
        json.dumps({'decision': 'decline', 'speech': 'I decline this request.'})
        if kwargs.get('format') == rules.resident_response_schema()
        else 'a cautious resident considering the available observations.'
    )
    return {'response': result}

  with (
      mock.patch.object(run.ollama_model.ollama, 'Client') as client,
      mock.patch.object(game_service, 'Game', RecordedGame),
      mock.patch.object(
          generic.Simulation, 'play', side_effect=AssertionError('No engine')
      ),
  ):
    client.return_value.generate.side_effect = generate
    with mock.patch.object(run.time, 'sleep', side_effect=KeyboardInterrupt):
      run.main([
          '--mode',
          'live',
          '--resident-prefab',
          logic,
          '--editor-port',
          '0',
          '--player-port',
          '0',
          '--output',
          str(tmp_path),
      ])
    game = built[0]
    # Unit exercise of one normal actor; no Simulation.play or watch execution.
    game.world.resolve(rules.PLAYER, 'ask Nell for fuel and part')
    nell = next(x for x in game.simulation.get_entities() if x.name == 'Nell')
    nell.observe('The coordinator asks for reserve fuel and the spare part.')
    result = nell.act(
        entity_lib.free_action_spec(call_to_action=game.world.action_prompt())
    )
    assert rules.parse_resident_response(result)[0] == 'decline'
    assert json.loads(result)['decision'] == 'decline'
    assert len(prompts) == expected_calls
    assert 'Return exactly one JSON object' in prompts[-1]
    if logic == 'basic':
      assert all(
          'Return exactly one JSON object' not in p for p in prompts[:-1]
      )
      assert all('one JSON decision object' not in p for p in prompts[:-1])
      assert any('What kind of person' in p for p in prompts[:-1])
      assert any('What situation' in p for p in prompts[:-1])
    assert formats == [None] * (expected_calls - 1) + [
        rules.resident_response_schema()
    ]
    assert client.call_args_list == [mock.call(timeout=90)] * 2
    # Exhaust each real standard limiter; no extra provider request is made.
    for wrapped, limit, already_used in (
        (models[0], 192, expected_calls - 1),
        (models[1], 64, 1),
    ):
      for _ in range(limit - already_used):
        wrapped.sample_text('bounded unit request')
      before = len(prompts)
      assert wrapped.sample_text('past limit') == ''
      assert len(prompts) == before
    assert len(prompts) == 256
    assert game.simulation.get_raw_log() == []


@pytest.mark.parametrize('logic', ['minimal', 'basic'])
def test_action_model_is_runtime_only_and_human_policy_has_precedence(logic):
  action_model = mock.Mock(wraps=game_prefab.FixtureModel())
  action_model.sample_text.return_value = (
      '{"decision":"decline","speech":"No."}'
  )
  human_response = '{"decision":"counter","speech":"My own human terms."}'
  reader = mock.Mock(return_value=human_response)
  case = researcher.prepare_case(
      'bellwether',
      lambda request: 'wait',
      human_readers={'Nell': reader},
      actor_logic=logic,
      action_model=action_model,
  )
  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No engine')
  ):
    simulation = generic.Simulation(
        case.config,
        game_prefab.FixtureModel(),
        lambda text: np.ones(8),
        engine=sequential.Sequential(),
    )
    actors = {a.name: a for a in simulation.get_entities()}
    spec = entity_lib.free_action_spec(call_to_action='Return decision JSON.')
    assert actors['Nell'].act(spec) == human_response
    reader.assert_called_once()
    action_model.sample_text.assert_not_called()
    assert json.loads(actors['Mara'].act(spec))['decision'] == 'decline'
    action_model.sample_text.assert_called_once()
    # The live model is captured by the trusted build factory, not params.
    assert all('action_model' not in i.params for i in case.config.instances)

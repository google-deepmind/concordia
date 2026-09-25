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

"""CLI prefab selection reaches the runner without starting a simulation."""

from unittest import mock

from concordia.examples.astral_canticle import adventure
from concordia.examples.astral_canticle import terminal
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
import pytest


@pytest.mark.parametrize(
    'args, expected', [([], 'minimal'), (['--player-prefab', 'basic'], 'basic')]
)
@pytest.mark.parametrize('transport', ['terminal', 'web'])
def test_cli_passes_prefab_to_runner(monkeypatch, args, expected, transport):
  if transport == 'web':
    pytest.importorskip('fastapi')
    from concordia.examples.astral_canticle import web  # pylint: disable=g-import-not-at-top

    module = web
  else:
    module = terminal
  monkeypatch.setattr('sys.argv', [transport, *args])
  with mock.patch.object(adventure, 'play') as play, mock.patch.object(
      module.ollama_model, 'OllamaLanguageModel'
  ):
    if transport == 'web':
      # Invoke only the captured runner with play mocked out; no server, model
      # inference or simulation is launched by this check.
      with mock.patch.object(module, 'create_app') as create, mock.patch(
          'concordia.examples.astral_canticle.web.uvicorn.run'
      ) as serve:
        module.main()
        create.call_args.kwargs['runner']()
        assert serve.call_args.kwargs['host'] == '127.0.0.1'
    else:
      module.main()
  play.assert_called_once()
  assert play.call_args.kwargs['player_prefab'] == expected
  assert play.call_args.kwargs['role'] == 'player'


def test_terminal_prints_preassembled_string_without_dict_headings(
    capsys, monkeypatch
):
  context = (
      '\nGoal: Repair the loom.\n\nSelf: A keeper.\nSituation: A cracked'
      ' cradle.\nIntent: Inspect.'
  )
  request = human_input.HumanInputRequest(
      request_id='terminal',
      entity_name='Ilyra',
      action_spec=entity_lib.free_action_spec(call_to_action='Your move?'),
      contexts={'not-a-display-heading': 'must not be reconstructed'},
      context=context,
  )
  monkeypatch.setattr('builtins.input', lambda prompt: 'LOOK')
  assert terminal.TerminalInput()(request) == 'LOOK'
  assert capsys.readouterr().out == context + '\n\nIlyra: Your move?\n'
